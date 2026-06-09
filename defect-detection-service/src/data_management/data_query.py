import base64
from typing import Dict, Any, Optional, List

from src.utils.logger import Logger

logger = Logger("data_query", "INFO", "./logs/defect-detection.log").logger


class DataQuery:
    def __init__(self, record_store, image_archive):
        self.record_store = record_store
        self.image_archive = image_archive

    def search(self, product_id=None, product_model=None, start_time=None,
               end_time=None, defect_type=None, result=None,
               limit=50, offset=0) -> Dict[str, Any]:
        records, total_count = self.record_store.query_records(
            product_id=product_id,
            start_time=start_time,
            end_time=end_time,
            result=result,
            defect_type=defect_type,
            limit=limit,
            offset=offset
        )

        if product_model:
            records = [r for r in records if r.get("product_model") == product_model]

        for record in records:
            thumbnail_base64 = None
            try:
                thumbnail_path = record.get("thumbnail_path", "")
                if thumbnail_path:
                    thumbnail_data = self.image_archive.get_image(thumbnail_path)
                    if thumbnail_data is not None:
                        thumbnail_base64 = base64.b64encode(thumbnail_data).decode("utf-8")
                else:
                    det_id = record.get("detection_id", "")
                    pid = record.get("product_id", "")
                    if det_id and pid:
                        thumbnail_data = self.image_archive.get_thumbnail(det_id, pid)
                        if thumbnail_data is not None:
                            thumbnail_base64 = base64.b64encode(thumbnail_data).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to load thumbnail for {record.get('detection_id')}: {e}")
            record["thumbnail_base64"] = thumbnail_base64

        return {
            "records": records,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }

    def get_record_detail(self, detection_id: str) -> Optional[Dict]:
        record = self.record_store.get_record_by_detection_id(detection_id)
        if record is None:
            return None

        original_image_base64 = None
        annotated_image_base64 = None

        try:
            original_path = record.get("original_image_path", "")
            if original_path:
                original_data = self.image_archive.get_image(original_path)
                if original_data is not None:
                    original_image_base64 = base64.b64encode(original_data).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to load original image for {detection_id}: {e}")

        try:
            annotated_path = record.get("annotated_image_path", "")
            if annotated_path:
                annotated_data = self.image_archive.get_image(annotated_path)
                if annotated_data is not None:
                    annotated_image_base64 = base64.b64encode(annotated_data).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to load annotated image for {detection_id}: {e}")

        record["original_image_base64"] = original_image_base64
        record["annotated_image_base64"] = annotated_image_base64
        return record

    def get_search_suggestions(self) -> Dict[str, Any]:
        product_ids = set()
        product_models = set()
        defect_types = set()

        try:
            records, _ = self.record_store.query_records(limit=10000)
            for record in records:
                pid = record.get("product_id", "")
                pm = record.get("product_model", "")
                dt = record.get("defect_types", "")

                if pid:
                    product_ids.add(pid)
                if pm:
                    product_models.add(pm)
                if dt:
                    for d in dt.split(","):
                        d = d.strip()
                        if d:
                            defect_types.add(d)
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")

        return {
            "product_ids": sorted(list(product_ids)),
            "product_models": sorted(list(product_models)),
            "defect_types": sorted(list(defect_types))
        }
