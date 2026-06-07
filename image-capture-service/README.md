# 流水线缺陷检测系统 - 图像采集服务

## 项目概述

本模块是生产流水线实时缺陷检测系统的核心组件之一，负责多相机图像的同步采集、光源控制、图像预处理、数据缓存与消息分发。

## 核心功能

### 一、图像采集
- **多相机同步触发**：通过光电传感器检测产品到位，同时触发顶部、底部、侧方多个工业相机拍照
- **光源控制**：支持频闪、常亮模式，可编程调节亮度与色温，适配金属、玻璃、塑料等不同材质
- **图像预处理**：自动进行 Bayer 转 RGB、畸变校正、ROI 裁剪、尺寸缩放，减少传输带宽
- **本地缓存与补传**：图像先存入环形缓冲区，异步发送至消息队列；网络中断时自动落盘，恢复后补传
- **相机状态监控**：实时上报相机在线/离线、曝光参数、触发计数，异常时触发告警

## 技术架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  光电传感器触发 │────▶│  多相机同步采集 │────▶│  光源同步控制  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  图像预处理模块 │
                        │  - Bayer转RGB   │
                        │  - 畸变校正     │
                        │  - ROI裁剪      │
                        │  - 尺寸缩放     │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  环形缓冲区     │────┐
                        └─────────────────┘    │  网络正常
                                 │            ▼
                        网络异常 │    ┌─────────────────┐
                                 ▼    │  消息队列发送  │──▶ RabbitMQ/Kafka
                        ┌─────────────────┐ └─────────────────┘
                        │  本地磁盘缓存   │
                        │  - 自动补传     │
                        │  - 重试机制     │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  状态监控告警   │──▶ Prometheus / Webhook
                        └─────────────────┘
```

## 目录结构

```
image-capture-service/
├── src/
│   ├── main.py                    # 主服务入口
│   ├── capture/
│   │   ├── base_camera.py         # 相机抽象基类
│   │   ├── mock_camera.py         # 模拟相机实现
│   │   ├── basler_camera.py       # Basler相机实现
│   │   ├── hik_camera.py          # 海康相机实现
│   │   ├── camera_manager.py      # 多相机管理器
│   │   └── trigger_controller.py  # 触发器控制器
│   ├── lighting/
│   │   ├── base_controller.py     # 光源控制器抽象基类
│   │   └── light_controller.py    # Modbus光源控制器
│   ├── preprocessing/
│   │   └── image_processor.py     # 图像预处理模块
│   ├── buffer/
│   │   ├── ring_buffer.py         # 环形缓冲区
│   │   └── local_cache.py         # 本地缓存与补传
│   ├── messaging/
│   │   ├── base_producer.py       # 消息生产者抽象基类
│   │   ├── rabbitmq_producer.py   # RabbitMQ生产者
│   │   ├── kafka_producer.py      # Kafka生产者
│   │   └── message_sender.py      # 消息发送管理器
│   ├── monitoring/
│   │   └── camera_monitor.py      # 相机监控与告警
│   ├── config/
│   │   └── settings.py            # 配置管理
│   └── utils/
│       ├── logger.py              # 日志工具
│       └── schemas.py             # 数据结构定义
├── config/
│   └── config.yaml                # 配置文件
├── docker/
│   └── prometheus.yml             # Prometheus配置
├── data/                          # 数据缓存目录
├── logs/                          # 日志目录
├── Dockerfile                     # Docker镜像构建
├── docker-compose.yml             # Docker Compose部署
├── requirements.txt               # Python依赖
└── README.md                      # 本文档
```

## 快速开始

### 环境要求

- Python 3.8+
- 工业相机 SDK（Basler pypylon / 海康 MVS SDK，可选）
- RabbitMQ 或 Kafka
- Docker（可选，推荐）

### 本地运行

1. **安装依赖**
```bash
cd image-capture-service
pip install -r requirements.txt
```

2. **修改配置**

编辑 `config/config.yaml`，根据实际情况修改相机参数、光源配置、消息队列连接信息等。

3. **启动服务**
```bash
python src/main.py
```

### Docker 部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f image-capture
```

## API 接口

### 状态查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/status` | 服务综合状态 |
| GET | `/api/cameras` | 相机状态列表 |
| GET | `/api/alerts?limit=100` | 告警历史 |
| GET | `/api/stats` | 发送统计信息 |
| GET | `/api/config` | 当前配置 |

### 控制操作

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/trigger` | 手动触发一次采集 |
| GET | `/api/reload-config` | 重新加载配置文件 |
| POST | `/api/mq/reconnect` | 重连消息队列 |

### 相机参数调整

```bash
# 设置单个相机曝光
curl -X POST http://localhost:8000/api/camera/exposure \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam-top-01", "exposure_time": 5000}'

# 设置所有相机曝光
curl -X POST http://localhost:8000/api/cameras/exposure \
  -H "Content-Type: application/json" \
  -d '{"exposure_time": 5000}'

# 设置相机增益
curl -X POST http://localhost:8000/api/camera/gain \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam-top-01", "gain": 1.2}'
```

### 光源参数调整

```bash
# 设置亮度
curl -X POST http://localhost:8000/api/light/brightness \
  -H "Content-Type: application/json" \
  -d '{"channel_id": "ch-top", "brightness": 80}'

# 设置模式 (strobe/continuous/off)
curl -X POST http://localhost:8000/api/light/mode \
  -H "Content-Type: application/json" \
  -d '{"channel_id": "ch-top", "mode": "strobe"}'

# 设置色温
curl -X POST http://localhost:8000/api/light/color_temp \
  -H "Content-Type: application/json" \
  -d '{"channel_id": "ch-top", "color_temp": 5500}'

# 应用材质预设 (metal/glass/plastic)
curl -X POST http://localhost:8000/api/light/preset \
  -H "Content-Type: application/json" \
  -d '{"material": "metal"}'
```

### 预处理参数调整

```bash
curl -X POST http://localhost:8000/api/preprocessing/config \
  -H "Content-Type: application/json" \
  -d '{
    "roi": {"x": 100, "y": 100, "width": 1720, "height": 1000},
    "resize": {"width": 960, "height": 600, "interpolation": "linear"},
    "enable_undistort": true
  }'
```

## 监控指标

Prometheus 指标暴露在端口 `9090`，主要指标包括：

| 指标名称 | 说明 |
|----------|------|
| `defect_camera_status` | 相机状态 (0=离线, 1=在线, 2=错误) |
| `defect_camera_exposure_time_us` | 相机曝光时间 |
| `defect_camera_gain` | 相机增益 |
| `defect_camera_trigger_count_total` | 触发总次数 |
| `defect_capture_count_total` | 采集总次数 |
| `defect_system_cpu_percent` | 系统CPU使用率 |
| `defect_system_memory_percent` | 系统内存使用率 |

## 消息队列数据格式

### 消息结构

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
      "timestamp": 1717234567.89,
      "width": 960,
      "height": 600,
      "pixel_format": "BayerRG8",
      "trigger_count": 100,
      "metadata": {
        "temperature": 25.5,
        "exposure_time": 5000,
        "gain": 1.0,
        "preprocessing_steps": ["bayer_conversion", "undistort", "roi_crop", "resize"]
      }
    }
  ],
  "image_data": {
    "img-1234": [255, 255, ...]  // JPEG编码的图像字节数组
  }
}
```

## 相机集成说明

### Basler 相机

1. 安装 pypylon SDK：
```bash
pip install pypylon
```

2. 修改配置文件中的相机 `type` 为 `basler`，并设置正确的序列号。

### 海康相机

1. 安装海康 MVS SDK 和 Python 绑定。
2. 修改配置文件中的相机 `type` 为 `hikvision`。

### 模拟模式

默认使用模拟模式 (`type: mock`)，可在无需真实硬件的情况下测试完整流程。

## 故障排查

### 相机连接失败
- 检查相机电源和网线连接
- 确认相机IP地址与配置一致
- 验证相机序列号是否正确
- 查看日志中详细的错误信息

### 消息队列连接失败
- 检查 RabbitMQ/Kafka 服务是否正常运行
- 确认连接参数（主机、端口、用户名、密码）正确
- 检查网络防火墙设置

### 图像采集延迟过高
- 检查曝光时间设置是否过大
- 确认预处理步骤是否过于复杂
- 检查系统资源使用率（CPU、内存）
- 考虑增加环形缓冲区大小

## 后续模块规划

本项目将逐步增加以下功能模块：
1. **缺陷检测模块** - 基于 OpenCV + PyTorch/TensorFlow 的划痕、脏污识别
2. **设备联动模块** - Modbus TCP/OPC UA/MQTT 与 PLC 通信，执行剔除或停机
3. **数据存储模块** - PostgreSQL + TimescaleDB 存储检测数据，Redis 缓存
4. **后端管理服务** - Spring Boot 2.7 + Java 8 业务逻辑层
5. **前端管理端** - Vue3 + Ant Design Vue + ECharts
6. **移动端** - uni-app

## License

Proprietary - Internal Use Only
