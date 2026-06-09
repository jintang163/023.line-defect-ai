import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.utils.logger import Logger

logger = Logger("param_adjuster", "INFO", "./logs/defect-detection.log").logger


class ParamAdjuster:
    _KNOWN_PRODUCT_PARAMS = {
        "sensitivity", "allowed_error_mm", "allow_multiple_defects",
        "max_defects_allowed", "pixel_to_mm_ratio", "inference_timeout_ms",
        "gpu_device_id", "enable_tensorrt", "product_name",
    }

    def __init__(self, algorithm_manager, config_manager):
        self._algorithm_manager = algorithm_manager
        self._config_manager = config_manager
        self._change_log: List[Dict] = []
        self._next_id: int = 1

    def adjust_product_param(
        self,
        product_id: str,
        param_path: str,
        new_value: Any,
        operator: str = "system",
    ) -> bool:
        if product_id not in self._algorithm_manager._products:
            logger.error(f"Product not found: {product_id}")
            return False

        product_config = self._algorithm_manager._products[product_id]
        old_value = self._get_param_value(product_config, param_path)
        old_value_serializable = old_value.value if hasattr(old_value, "value") else old_value

        try:
            self._set_param_value(product_config, param_path, new_value)
        except Exception as e:
            logger.error(f"Failed to set param {param_path} for product {product_id}: {e}")
            return False

        new_value_serializable = new_value
        if hasattr(new_value, "value"):
            new_value_serializable = new_value.value

        self._add_change_log({
            "timestamp": datetime.now().isoformat(),
            "operator": operator,
            "param_path": param_path,
            "old_value": old_value_serializable,
            "new_value": new_value_serializable,
            "product_id": product_id,
        })

        logger.info(
            f"Param adjusted: product={product_id} path={param_path} "
            f"old={old_value} new={new_value} operator={operator}"
        )
        return True

    def adjust_threshold(
        self,
        product_id: str,
        defect_type: str,
        field: str,
        new_value: Any,
        operator: str = "system",
    ) -> bool:
        if product_id not in self._algorithm_manager._products:
            logger.error(f"Product not found: {product_id}")
            return False

        product_config = self._algorithm_manager._products[product_id]

        defect_config = None
        for dt in product_config.defect_types:
            if dt.type.value == defect_type:
                defect_config = dt
                break

        if defect_config is None:
            logger.error(f"Defect type {defect_type} not found for product {product_id}")
            return False

        valid_fields = {
            "min_area_mm2", "max_area_mm2", "min_confidence",
            "severity", "alert_action",
        }
        if field not in valid_fields:
            logger.error(f"Invalid threshold field: {field}")
            return False

        old_value = getattr(defect_config, field)

        if field == "severity":
            from src.utils.schemas import DefectSeverity
            new_value = DefectSeverity(new_value)
        elif field == "alert_action":
            from src.utils.schemas import AlertAction
            new_value = AlertAction(new_value)

        setattr(defect_config, field, new_value)

        self._add_change_log({
            "timestamp": datetime.now().isoformat(),
            "operator": operator,
            "param_path": f"defect_types.{defect_type}.{field}",
            "old_value": old_value.value if hasattr(old_value, "value") else old_value,
            "new_value": new_value.value if hasattr(new_value, "value") else new_value,
            "product_id": product_id,
        })

        logger.info(
            f"Threshold adjusted: product={product_id} defect_type={defect_type} "
            f"field={field} old={old_value} new={new_value} operator={operator}"
        )
        return True

    def get_product_params(self, product_id: str) -> Optional[Dict]:
        if product_id not in self._algorithm_manager._products:
            logger.error(f"Product not found: {product_id}")
            return None

        product_config = self._algorithm_manager._products[product_id]
        return product_config.to_dict()

    def get_change_log(self, limit: int = 100) -> List[Dict]:
        return list(reversed(self._change_log[-limit:]))

    def rollback_change(self, change_id: int) -> bool:
        entry = None
        entry_index = None
        for i, log in enumerate(self._change_log):
            if log.get("id") == change_id:
                entry = log
                entry_index = i
                break

        if entry is None:
            logger.error(f"Change log entry not found: id={change_id}")
            return False

        product_id = entry["product_id"]
        param_path = entry["param_path"]
        old_value = entry["old_value"]

        if product_id not in self._algorithm_manager._products:
            logger.error(f"Product not found for rollback: {product_id}")
            return False

        product_config = self._algorithm_manager._products[product_id]

        try:
            self._set_param_value(product_config, param_path, old_value)
        except Exception as e:
            logger.error(f"Failed to rollback param {param_path}: {e}")
            return False

        self._change_log.pop(entry_index)

        logger.info(
            f"Rolled back change id={change_id}: product={product_id} "
            f"path={param_path} restored={old_value}"
        )
        return True

    def _add_change_log(self, entry: Dict):
        entry["id"] = self._next_id
        self._next_id += 1
        self._change_log.append(entry)
        if len(self._change_log) > 1000:
            self._change_log = self._change_log[-1000:]

    def _get_param_value(self, product_config, param_path: str) -> Any:
        parts = param_path.split(".")

        if len(parts) == 1:
            val = getattr(product_config, parts[0])
            return val.value if hasattr(val, "value") else val

        if parts[0] == "rois" and len(parts) >= 3:
            index = int(parts[1])
            field = parts[2]
            return getattr(product_config.rois[index], field)

        if parts[0] == "algorithms" and len(parts) >= 3:
            index = int(parts[1])
            if parts[2] == "params" and len(parts) >= 4:
                key = ".".join(parts[3:])
                return product_config.algorithms[index].params.get(key)
            val = getattr(product_config.algorithms[index], parts[2])
            return val.value if hasattr(val, "value") else val

        if parts[0] == "defect_types" and len(parts) >= 3:
            type_name = parts[1]
            field = parts[2]
            for dt in product_config.defect_types:
                if dt.type.value == type_name:
                    val = getattr(dt, field)
                    return val.value if hasattr(val, "value") else val

        obj = product_config
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                obj = getattr(obj, part)
        return obj.value if hasattr(obj, "value") else obj

    def _set_param_value(self, product_config, param_path: str, value: Any):
        parts = param_path.split(".")

        if len(parts) == 1:
            setattr(product_config, parts[0], value)
            return

        if parts[0] == "rois" and len(parts) >= 3:
            index = int(parts[1])
            field = parts[2]
            setattr(product_config.rois[index], field, value)
            return

        if parts[0] == "algorithms" and len(parts) >= 3:
            index = int(parts[1])
            if parts[2] == "params" and len(parts) >= 4:
                key = ".".join(parts[3:])
                product_config.algorithms[index].params[key] = value
                return
            setattr(product_config.algorithms[index], parts[2], value)
            return

        if parts[0] == "defect_types" and len(parts) >= 3:
            type_name = parts[1]
            field = parts[2]
            for dt in product_config.defect_types:
                if dt.type.value == type_name:
                    if field == "severity":
                        from src.utils.schemas import DefectSeverity
                        value = DefectSeverity(value)
                    elif field == "alert_action":
                        from src.utils.schemas import AlertAction
                        value = AlertAction(value)
                    setattr(dt, field, value)
                    return

        obj = product_config
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                obj = getattr(obj, part)

        final_part = parts[-1]
        if isinstance(obj, dict):
            obj[final_part] = value
        else:
            setattr(obj, final_part, value)
