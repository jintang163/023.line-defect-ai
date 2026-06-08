import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")

    try:
        from src.utils.logger import Logger
        print("✓ Logger imported successfully")
    except Exception as e:
        print(f"✗ Logger import failed: {e}")
        return False

    try:
        from src.utils.schemas import (
            DefectType, DefectSeverity, AlgorithmType, InferenceBackend,
            DetectionResult, AlertAction, ROI, BoundingBox, Defect,
            ProductConfig, ImageData, DetectionOutput
        )
        print("✓ Schemas imported successfully")
    except Exception as e:
        print(f"✗ Schemas import failed: {e}")
        return False

    try:
        from src.config.settings import ConfigManager
        print("✓ ConfigManager imported successfully")
    except Exception as e:
        print(f"✗ ConfigManager import failed: {e}")
        return False

    try:
        from src.algorithms.base_algorithm import BaseDetectionAlgorithm
        from src.algorithms.edge_detection import EdgeDetectionAlgorithm
        from src.algorithms.template_matching import TemplateMatchingAlgorithm
        from src.algorithms.gray_diff import GrayDiffAlgorithm
        print("✓ Traditional algorithms imported successfully")
    except Exception as e:
        print(f"✗ Traditional algorithms import failed: {e}")
        return False

    try:
        from src.deep_learning.base_dl import BaseDeepLearningAlgorithm
        from src.deep_learning.classification import ClassificationAlgorithm
        from src.deep_learning.object_detection import ObjectDetectionAlgorithm
        from src.deep_learning.segmentation import SegmentationAlgorithm
        print("✓ Deep learning algorithms imported successfully")
    except Exception as e:
        print(f"✗ Deep learning algorithms import failed: {e}")
        return False

    try:
        from src.inference.onnx_engine import ONNXInferenceEngine
        print("✓ ONNX Inference Engine imported successfully")
    except Exception as e:
        print(f"✗ ONNX Inference Engine import failed: {e}")
        return False

    try:
        from src.algorithm_manager import AlgorithmManager
        print("✓ AlgorithmManager imported successfully")
    except Exception as e:
        print(f"✗ AlgorithmManager import failed: {e}")
        return False

    try:
        from src.result_annotator import ResultAnnotator
        print("✓ ResultAnnotator imported successfully")
    except Exception as e:
        print(f"✗ ResultAnnotator import failed: {e}")
        return False

    try:
        from src.alert_manager import AlertManager
        print("✓ AlertManager imported successfully")
    except Exception as e:
        print(f"✗ AlertManager import failed: {e}")
        return False

    try:
        from src.messaging.message_consumer import MessageConsumer
        from src.messaging.result_producer import ResultProducer
        print("✓ Messaging components imported successfully")
    except Exception as e:
        print(f"✗ Messaging components import failed: {e}")
        return False

    print("\nAll imports successful! ✓")
    return True

def test_enums():
    print("\nTesting enums...")

    from src.utils.schemas import DefectType, DefectSeverity, AlgorithmType, DetectionResult

    assert DefectType.SCRATCH.value == "scratch"
    assert DefectSeverity.CRITICAL.value == "critical"
    assert AlgorithmType.EDGE_DETECTION.value == "edge_detection"
    assert DetectionResult.OK.value == "OK"

    print("✓ All enums work correctly")
    return True

def test_data_classes():
    print("\nTesting data classes...")

    from src.utils.schemas import ROI, BoundingBox, Defect, DetectionOutput, ProductConfig
    from src.utils.schemas import DefectType, DefectSeverity

    roi = ROI(x=100, y=200, width=500, height=300, name="test_roi")
    assert roi.to_tuple() == (100, 200, 500, 300)
    print("✓ ROI works correctly")

    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
    assert bbox.width == 100
    assert bbox.height == 50
    assert bbox.area == 5000
    print("✓ BoundingBox works correctly")

    import numpy as np
    defect = Defect.create(
        defect_type=DefectType.SCRATCH,
        severity=DefectSeverity.MAJOR,
        confidence=0.95,
        bbox=bbox,
        area_mm2=0.125
    )
    assert defect.defect_id is not None
    assert defect.type == DefectType.SCRATCH
    assert defect.confidence == 0.95
    print("✓ Defect works correctly")

    output = DetectionOutput.create(
        sequence_id="test-seq-123",
        product_id="PROD-001",
        defects=[defect]
    )
    assert output.result == DetectionResult.OK
    assert len(output.defects) == 1
    assert len(output.critical_defects) == 0
    assert len(output.major_defects) == 1
    print("✓ DetectionOutput works correctly")

    print("✓ All data classes work correctly")
    return True

def main():
    print("=" * 60)
    print("Defect Detection Service - Import Verification")
    print("=" * 60)
    print()

    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_enums()
    all_passed &= test_data_classes()

    print()
    print("=" * 60)
    if all_passed:
        print("✓ All verification tests passed!")
        return 0
    else:
        print("✗ Some verification tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
