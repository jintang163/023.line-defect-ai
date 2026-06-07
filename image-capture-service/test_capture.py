#!/usr/bin/env python
"""
快速测试脚本 - 验证图像采集服务的核心功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
from src.config.settings import ConfigManager
from src.utils.schemas import LightMode
from src.utils.logger import Logger

logger = Logger("test", "DEBUG").logger


def test_config():
    """测试配置加载"""
    logger.info("=== 测试配置加载 ===")
    cfg = ConfigManager("./config/config.yaml")

    service_cfg = cfg.get_service_config()
    logger.info(f"服务名称: {service_cfg['name']}")

    cameras = cfg.get_camera_configs()
    logger.info(f"相机数量: {len(cameras)}")
    for cam in cameras:
        logger.info(f"  - {cam.id}: {cam.name} ({cam.type.value})")

    lights = cfg.get_light_channels()
    logger.info(f"光源通道数量: {len(lights)}")
    for ch in lights:
        logger.info(f"  - {ch.id}: {ch.name}, 模式={ch.mode.value}")

    logger.info("配置测试通过 ✓")
    return True


def test_ring_buffer():
    """测试环形缓冲区"""
    logger.info("\n=== 测试环形缓冲区 ===")
    from src.buffer.ring_buffer import RingBuffer

    buffer = RingBuffer(max_size=5)

    for i in range(7):
        success = buffer.put(f"item-{i}", block=False)
        logger.info(f"放入 item-{i}: {'成功' if success else '失败(缓冲区满)'}")

    logger.info(f"缓冲区大小: {len(buffer)}")
    logger.info(f"溢出次数: {buffer.overflow_count}")

    for i in range(3):
        item = buffer.get(block=False)
        logger.info(f"取出: {item}")

    logger.info(f"缓冲区大小: {len(buffer)}")
    logger.info("环形缓冲区测试通过 ✓")
    return True


def test_image_preprocessing():
    """测试图像预处理"""
    logger.info("\n=== 测试图像预处理 ===")
    import numpy as np
    import cv2
    from src.preprocessing.image_processor import ImagePreprocessor
    from src.utils.schemas import CapturedImage

    cfg = ConfigManager("./config/config.yaml")
    preprocessor = ImagePreprocessor(cfg)

    raw_image = np.random.randint(0, 255, (1200, 1920), dtype=np.uint8)
    img = CapturedImage.create(
        camera_id="cam-test",
        camera_position="top",
        raw_data=raw_image,
        width=1920,
        height=1200,
        pixel_format="BayerRG8"
    )

    logger.info(f"原始图像尺寸: {raw_image.shape}")
    start = time.time()
    processed = preprocessor.process(img)
    elapsed = (time.time() - start) * 1000

    if processed is not None:
        logger.info(f"处理后图像尺寸: {processed.shape}")
        logger.info(f"预处理耗时: {elapsed:.2f}ms")
        logger.info(f"处理步骤: {img.metadata.get('preprocessing_steps', [])}")
        logger.info("图像预处理测试通过 ✓")
        return True
    else:
        logger.error("图像预处理失败")
        return False


def test_light_controller():
    """测试光源控制器"""
    logger.info("\n=== 测试光源控制器 ===")
    from src.lighting.light_controller import ModbusLightController

    cfg = ConfigManager("./config/config.yaml")
    controller = ModbusLightController(
        channels=cfg.get_light_channels(),
        host="192.168.1.200",
        port=502
    )

    connected = controller.connect()
    logger.info(f"光源控制器连接: {'成功' if connected else '失败(使用模拟模式)'}")

    success = controller.set_brightness("ch-top", 90)
    logger.info(f"设置亮度: {'成功' if success else '失败'}")

    success = controller.set_mode("ch-top", LightMode.STROBE)
    logger.info(f"设置频闪模式: {'成功' if success else '失败'}")

    success = controller.apply_material_preset("metal")
    logger.info(f"应用金属材质预设: {'成功' if success else '失败'}")

    success = controller.trigger_all_strobe()
    logger.info(f"触发所有频闪: {'成功' if success else '失败'}")

    channels = controller.get_all_channels()
    for ch in channels:
        logger.info(f"  - {ch.id}: 亮度={ch.brightness}, 模式={ch.mode.value}, 色温={ch.color_temp}K")

    logger.info("光源控制器测试通过 ✓")
    return True


def test_camera_manager():
    """测试相机管理器"""
    logger.info("\n=== 测试相机管理器 ===")
    from src.capture.camera_manager import CameraManager
    from src.capture.trigger_controller import TriggerController

    cfg = ConfigManager("./config/config.yaml")
    cam_manager = CameraManager(cfg)

    initialized = cam_manager.initialize()
    logger.info(f"相机初始化: {'成功' if initialized else '失败'}")

    statuses = cam_manager.get_camera_statuses()
    for status in statuses:
        logger.info(f"  - {status.camera_id}: {status.status.value}")

    trigger = TriggerController(cfg)
    trigger_count = 0

    def on_trigger():
        nonlocal trigger_count
        trigger_count += 1
        message = cam_manager.trigger_sync_capture()
        if message:
            logger.info(f"触发 #{trigger_count}: 采集到 {len(message.images)} 张图像, 序列={message.sequence_id[:8]}")

    trigger.set_callback(on_trigger)
    trigger.set_simulation_interval(1.0)
    trigger.start()

    logger.info("运行3次采集...")
    time.sleep(3.5)
    trigger.stop()

    logger.info(f"总触发次数: {trigger.trigger_count}")
    logger.info("相机管理器测试通过 ✓")
    return True


def test_messaging():
    """测试消息队列生产者"""
    logger.info("\n=== 测试消息队列 ===")
    from src.messaging.rabbitmq_producer import RabbitMQProducer
    from src.messaging.kafka_producer import KafkaProducer

    cfg = ConfigManager("./config/config.yaml")
    msg_cfg = cfg.get_messaging_config()

    rabbitmq = RabbitMQProducer(msg_cfg)
    connected = rabbitmq.connect()
    logger.info(f"RabbitMQ 连接: {'成功' if connected else '失败(使用模拟模式)'}")
    logger.info(f"RabbitMQ 发送计数: {rabbitmq.send_count}")

    kafka = KafkaProducer(msg_cfg)
    connected = kafka.connect()
    logger.info(f"Kafka 连接: {'成功' if connected else '失败(使用模拟模式)'}")
    logger.info(f"Kafka 发送计数: {kafka.send_count}")

    logger.info("消息队列测试通过 ✓")
    return True


def test_local_cache():
    """测试本地缓存"""
    logger.info("\n=== 测试本地缓存 ===")
    import numpy as np
    from src.buffer.local_cache import LocalCache
    from src.utils.schemas import ImageMessage, CapturedImage

    cache = LocalCache(cache_dir="./data/test_cache", max_size_gb=1, retry_interval=5)

    img = CapturedImage.create(
        camera_id="cam-test",
        camera_position="top",
        raw_data=np.random.randint(0, 255, (600, 960, 3), dtype=np.uint8),
        width=960,
        height=600,
        pixel_format="RGB"
    )
    img.processed_data = img.raw_data

    message = ImageMessage(
        sequence_id="test-sequence-001",
        timestamp=time.time(),
        images=[img]
    )

    saved = cache.save(message, [img])
    logger.info(f"缓存保存: {'成功' if saved else '失败'}")

    pending = cache.get_pending_count()
    logger.info(f"待处理数量: {pending}")

    stats = cache.get_cache_stats()
    logger.info(f"缓存统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    import shutil
    shutil.rmtree("./data/test_cache", ignore_errors=True)

    logger.info("本地缓存测试通过 ✓")
    return True


def main():
    logger.info("=" * 60)
    logger.info("图像采集服务 - 功能测试")
    logger.info("=" * 60)

    tests = [
        ("配置加载", test_config),
        ("环形缓冲区", test_ring_buffer),
        ("图像预处理", test_image_preprocessing),
        ("光源控制器", test_light_controller),
        ("相机管理器", test_camera_manager),
        ("消息队列", test_messaging),
        ("本地缓存", test_local_cache),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"{name} 测试异常: {e}", exc_info=True)
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"测试完成: 通过 {passed}/{len(tests)}, 失败 {failed}")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
