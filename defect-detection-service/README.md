# 流水线缺陷检测系统 - 缺陷检测服务

## 项目概述

本模块是生产流水线实时缺陷检测系统的核心组件，负责对采集的图像进行多算法融合的缺陷检测、结果标注、分级告警。支持传统计算机视觉算法与深度学习模型的灵活组合，可按产品型号切换检测策略。

## 核心功能

### 一、多算法支持

#### 传统算法
- **边缘检测**：基于 Canny 边缘检测算法，适用于划痕、裂纹等线性缺陷
- **模板匹配**：基于灰度模板匹配，适用于缺失、错位等结构缺陷
- **灰度差分**：与标准参考图像对比，适用于脏污、污渍、气泡等区域缺陷

#### 深度学习模型
- **图像分类**：ResNet/EfficientNet 等，用于整体 OK/NG 判定
- **目标检测**：YOLO/Faster R-CNN 等，用于多类别缺陷定位与分类
- **语义分割**：U-Net/SegFormer 等，用于精确的缺陷轮廓分割

### 二、可配置检测参数

每个产品独立配置以下参数：
- **ROI 区域**：支持多区域检测，可命名和独立开关
- **灵敏度**：全局灵敏度系数 0.1-1.0
- **允许误差**：缺陷判定的像素/毫米误差阈值
- **缺陷类型开关**：按缺陷类型独立配置
  - 最小/最大面积过滤（如忽略小于 0.1mm² 的脏污）
  - 置信度阈值
  - 严重程度分级
  - 告警动作配置

### 三、实时推理加速

- **ONNX Runtime**：跨平台推理引擎，支持 CPU/GPU
- **TensorRT 加速**：NVIDIA GPU 量化优化，单图处理 ≤50ms
- **多流并行**：支持多 ROI 并行推理
- **动态批处理**：小批量推理优化

### 四、结果标注与输出

- **缺陷轮廓绘制**：精确绘制分割算法输出的缺陷轮廓
- **边界框标注**：角点增强的目标检测框
- **标签与置信度**：显示缺陷类型、严重程度、置信度、面积
- **ROI 区域标注**：蓝色边框标注检测区域
- **结果横幅**：顶部显示 OK/NG 结果、缺陷数量、推理时间
- **图例说明**：右侧显示缺陷等级颜色图例

### 五、分级告警逻辑

| 严重程度 | 说明 | 默认动作 |
|---------|------|---------|
| CRITICAL | 关键缺陷（裂纹、断裂、缺失） | NG + 停机 |
| MAJOR | 主要缺陷（深划痕、大凹痕） | NG + 剔料 |
| MINOR | 轻微缺陷（小脏污、小气泡） | 仅记录 |
| WARNING | 警告（接近阈值） | 警告提示 |

- **连续不合格告警**：连续 N 个 NG 自动停机
- **告警回调**：支持注册自定义告警处理函数
- **告警历史**：最多保存 1000 条历史记录

## 技术架构

```
┌─────────────────┐
│  消息队列消费   │─── RabbitMQ/Kafka ── 图像采集服务
└─────────────────┘
          │
          ▼
┌─────────────────┐
│  图像预处理     │  ROI 裁剪、格式转换
└─────────────────┘
          │
          ▼
┌─────────────────┐
│  算法管理器     │  按产品配置调度算法
└─────────────────┘
   ┌─────┼─────┐
   ▼     ▼     ▼
┌────┐ ┌────┐ ┌────┐     ┌──────────────┐
│边缘│ │模板│ │灰度│     │ ONNX Runtime │
│检测│ │匹配│ │差分│────▶│  TensorRT    │
└────┘ └────┘ └────┘     └──────────────┘
   │     │     │               │
   └─────┼─────┘               ▼
         ▼               ┌──────────────┐
   ┌────────────┐        │  深度学习    │
   │ 结果融合   │        │  推理引擎    │
   │ NMS 去重   │        └──────────────┘
   └────────────┘               │
          │                     │
          └───────────┬─────────┘
                      ▼
              ┌──────────────┐
              │ 结果标注模块 │  可视化标注
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ 告警管理器   │  分级告警逻辑
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ 结果发送模块 │─── 消息队列 ── 后端/PLC
              └──────────────┘
```

## 目录结构

```
defect-detection-service/
├── src/
│   ├── main.py                          # 主服务入口
│   ├── algorithm_manager.py             # 算法管理器
│   ├── result_annotator.py              # 结果标注模块
│   ├── alert_manager.py                 # 告警管理器
│   ├── algorithms/
│   │   ├── base_algorithm.py            # 传统算法基类
│   │   ├── edge_detection.py            # 边缘检测算法
│   │   ├── template_matching.py         # 模板匹配算法
│   │   └── gray_diff.py                 # 灰度差分算法
│   ├── deep_learning/
│   │   ├── base_dl.py                   # 深度学习基类
│   │   ├── classification.py            # 分类算法
│   │   ├── object_detection.py          # 目标检测算法
│   │   └── segmentation.py              # 分割算法
│   ├── inference/
│   │   └── onnx_engine.py               # ONNX 推理引擎
│   ├── messaging/
│   │   ├── message_consumer.py          # 消息消费者
│   │   └── result_producer.py           # 结果生产者
│   ├── config/
│   │   └── settings.py                  # 配置管理
│   └── utils/
│       ├── logger.py                    # 日志工具
│       └── schemas.py                   # 数据结构定义
├── config/
│   ├── config.yaml                      # 服务配置
│   └── products.yaml                    # 产品检测配置
├── models/                              # 模型文件目录
│   └── references/                      # 参考图像目录
├── logs/                                # 日志目录
├── data/                                # 数据目录
├── requirements.txt                     # Python 依赖
└── README.md                            # 本文档
```

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.2+（GPU 推理）
- TensorRT 8.6+（可选，GPU 加速）
- RabbitMQ 或 Kafka
- Docker（可选，推荐）

### 本地运行

1. **安装依赖**
```bash
cd defect-detection-service
pip install -r requirements.txt

# GPU 版本（可选）
pip install onnxruntime-gpu>=1.16.0
```

2. **修改配置**

编辑 `config/config.yaml`，根据实际情况修改消息队列连接信息。

编辑 `config/products.yaml`，配置产品检测参数、算法组合、缺陷类型过滤规则。

3. **准备模型**

将 ONNX 模型文件放入 `./models/` 目录，参考图像放入 `./models/references/` 目录。

4. **启动服务**
```bash
python src/main.py
```

### Docker 部署

```bash
# 构建镜像
docker build -t line-defect-detection .

# 启动服务
docker run -d \
  --name defect-detection \
  -p 8081:8081 \
  -p 9091:9091 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --gpus all \
  line-defect-detection
```

## API 接口

### 状态查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/status` | 服务综合状态 |
| GET | `/api/products` | 产品列表 |
| GET | `/api/product/current` | 当前产品配置 |
| GET | `/api/algorithms` | 可用算法列表 |
| GET | `/api/alerts?limit=100` | 告警历史 |
| GET | `/api/stats` | 统计信息 |
| GET | `/api/config` | 当前配置 |

### 控制操作

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/product/switch` | 切换产品型号 |
| POST | `/api/algorithm/params` | 更新算法参数 |
| POST | `/api/detect` | 同步检测单张图像 |
| POST | `/api/alerts/reset-stop-line` | 复位停机状态 |
| POST | `/api/alerts/clear-history` | 清除告警历史 |
| POST | `/api/mq/reconnect` | 重连消息队列 |
| GET | `/api/reload-config` | 重新加载配置 |

### 图像检测 API

```bash
curl -X POST http://localhost:8081/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "PROD-001",
    "image": "base64_encoded_image_data",
    "return_annotated": true
  }'
```

### 切换产品型号

```bash
curl -X POST http://localhost:8081/api/product/switch \
  -H "Content-Type: application/json" \
  -d '{"product_id": "PROD-002"}'
```

## 产品配置说明

### 产品配置字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| product_id | 产品唯一标识 | "PROD-001" |
| product_name | 产品名称 | "金属外壳部件" |
| pixel_to_mm_ratio | 像素转毫米比例 | 0.025 |
| sensitivity | 全局灵敏度 | 0.8 |
| allowed_error_mm | 允许误差（毫米） | 0.1 |
| allow_multiple_defects | 是否允许多个缺陷 | false |
| max_defects_allowed | 最大允许缺陷数 | 0 |
| inference_backend | 推理后端 | "onnx_gpu" |
| enable_tensorrt | 启用 TensorRT | true |

### 缺陷类型配置

| 字段 | 说明 | 示例值 |
|------|------|--------|
| type | 缺陷类型 | "scratch" |
| enabled | 是否启用 | true |
| min_area_mm2 | 最小过滤面积 | 0.05 |
| max_area_mm2 | 最大过滤面积 | 100.0 |
| min_confidence | 最低置信度 | 0.7 |
| severity | 严重程度 | "major" |
| alert_action | 告警动作 | "reject" |

### 告警动作类型

- `none`：无动作
- `log`：仅记录日志
- `warn`：警告提示
- `reject`：剔料
- `stop_line`：停机

## 消息队列数据格式

### 输入消息（来自图像采集服务）

```json
{
  "sequence_id": "uuid-1234",
  "timestamp": 1717234567.89,
  "product_id": "PROD-001",
  "line_id": "line-001",
  "images": [
    {
      "image_id": "img-1234",
      "camera_id": "cam-top-01",
      "camera_position": "top",
      "width": 960,
      "height": 600,
      "image_url": "http://minio/images/img-1234.jpg"
    }
  ],
  "image_data": {
    "img-1234": "base64_encoded_jpeg_bytes"
  }
}
```

### 输出消息（检测结果）

```json
{
  "detection_id": "uuid-5678",
  "sequence_id": "uuid-1234",
  "product_id": "PROD-001",
  "result": "NG",
  "defects": [
    {
      "defect_id": "uuid-9012",
      "type": "scratch",
      "severity": "major",
      "confidence": 0.92,
      "bbox": {
        "x1": 100.5, "y1": 200.3,
        "x2": 150.2, "y2": 220.8,
        "area": 994.5
      },
      "area_mm2": 0.621,
      "description": "划痕: 0.92, area: 0.621 mm²"
    }
  ],
  "total_inference_time_ms": 42.5,
  "alert_action": "reject",
  "summary": {
    "critical_defects": 0,
    "major_defects": 1,
    "minor_defects": 0,
    "total_defects": 1
  }
}
```

## 监控指标

Prometheus 指标暴露在端口 `9091`：

| 指标名称 | 说明 |
|----------|------|
| `defect_detection_total` | 检测总数 |
| `defect_detection_ok_total` | OK 数量 |
| `defect_detection_ng_total` | NG 数量 |
| `defect_inference_time_ms` | 推理时间（毫秒） |
| `defect_alert_total` | 告警总数 |
| `defect_stop_line_active` | 停机状态 |
| `defect_consecutive_ng_count` | 连续 NG 计数 |

## 性能指标

| 算法类型 | CPU 推理 | GPU 推理 | TensorRT |
|---------|---------|---------|----------|
| 边缘检测 | 5-10ms | - | - |
| 模板匹配 | 20-50ms | - | - |
| 灰度差分 | 10-30ms | - | - |
| 图像分类 (ResNet18) | 30-50ms | 5-10ms | 2-5ms |
| 目标检测 (YOLOv8s) | 100-200ms | 15-30ms | 10-20ms |
| 语义分割 (U-Net) | 150-300ms | 20-40ms | 15-25ms |

*注：基于 640x640 输入图像测试*

## 常见问题

### 1. 如何添加新的缺陷类型？

在 `src/utils/schemas.py` 的 `DefectType` 枚举中添加新类型，然后在产品配置中配置过滤规则。

### 2. 如何集成新的深度学习模型？

- 分类模型：继承 `BaseDeepLearningAlgorithm`，实现 `_postprocess` 方法
- 目标检测：修改 `ObjectDetectionAlgorithm` 的后处理逻辑适配模型输出格式
- 分割模型：修改 `SegmentationAlgorithm` 的后处理逻辑

### 3. 如何实现 PLC 通信？

在告警回调中注册 PLC 通信函数，示例：

```python
from pymodbus.client import ModbusTcpClient

def plc_reject_callback(alert):
    if alert.action == AlertAction.REJECT:
        client = ModbusTcpClient("192.168.1.100")
        client.write_coil(100, True)
        client.close()

alert_manager.register_callback(AlertAction.REJECT, plc_reject_callback)
```

## License

Proprietary - Internal Use Only
