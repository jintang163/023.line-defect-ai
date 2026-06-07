#!/usr/bin/env python
"""
简单验证脚本 - 检查所有模块能否正常导入
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

modules = [
    ("src.utils.logger", "Logger"),
    ("src.utils.schemas", "CameraConfig, CapturedImage, ImageMessage"),
    ("src.config.settings", "ConfigManager"),
    ("src.buffer.ring_buffer", "RingBuffer"),
    ("src.buffer.local_cache", "LocalCache"),
    ("src.capture.base_camera", "BaseCamera"),
    ("src.capture.mock_camera", "MockCamera"),
    ("src.capture.basler_camera", "BaslerCamera"),
    ("src.capture.hik_camera", "HikvisionCamera"),
    ("src.capture.camera_manager", "CameraManager"),
    ("src.capture.trigger_controller", "TriggerController"),
    ("src.lighting.base_controller", "BaseLightController"),
    ("src.lighting.light_controller", "ModbusLightController"),
    ("src.messaging.base_producer", "BaseMessageProducer"),
    ("src.messaging.rabbitmq_producer", "RabbitMQProducer"),
    ("src.messaging.kafka_producer", "KafkaProducer"),
    ("src.messaging.message_sender", "MessageSender"),
    ("src.monitoring.camera_monitor", "CameraMonitor, AlertManager"),
]

print("=" * 60)
print("模块导入验证")
print("=" * 60)

success_count = 0
fail_count = 0

for module_name, class_names in modules:
    try:
        module = __import__(module_name, fromlist=['*'])
        for class_name in [c.strip() for c in class_names.split(',')]:
            if hasattr(module, class_name):
                print(f"✓ {module_name}.{class_name}")
            else:
                print(f"✗ {module_name}.{class_name} - 类不存在")
                fail_count += 1
                continue
        success_count += 1
    except Exception as e:
        print(f"✗ {module_name} - 导入失败: {e}")
        fail_count += 1

print("\n" + "=" * 60)
print(f"结果: {success_count} 成功, {fail_count} 失败")
print("=" * 60)

if fail_count == 0:
    print("\n所有模块导入成功！✓")
else:
    print(f"\n有 {fail_count} 个模块导入失败")
    sys.exit(1)
