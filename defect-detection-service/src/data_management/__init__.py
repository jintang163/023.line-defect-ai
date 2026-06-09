from typing import Dict, Any, Optional, List
import json
import cv2
import numpy as np

from src.utils.logger import Logger
from src.utils.schemas import (
    DetectionRecord, DetectionOutput, DetectionResult, ProductConfig
)

from src.data_management.record_store import RecordStore
from src.data_management.image_archive import ImageArchive
from src.data_management.data_query import DataQuery
from src.data_management.data_export import DataExport
from src.data_management.data_statistics import DataStatistics

logger = Logger("data_management", "INFO", "./logs/defect-detection.log").logger


class DataManagementManager:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)

        self._record_store: Optional[RecordStore] = None
        self._image_archive: Optional[ImageArchive] = None
        self._data_query: Optional[DataQuery] = None
        self._data_export: Optional[DataExport] = None
        self._data_statistics: Optional[DataStatistics] = None

        if not self._enabled:
            logger.info("Data management module is disabled")
            return

        self._init_components()

    def _init_components(self):
        record_config = self._config.get("record_storage", {})
        self._record_store = RecordStore(record_config)
        logger.info(f"Record store enabled: {record_config.get('enable', False)}")

        archive_config = self._config.get("image_archive", {})
        self._image_archive = ImageArchive(archive_config)
        logger.info(f"Image archive enabled: {archive_config.get('enable', False)}")

        self._data_query = DataQuery(self._record_store, self._image_archive)
        self._data_statistics = DataStatistics(self._record_store)

        export_config = self._config.get("export", {})
        self._data_export = DataExport(
            self._record_store, self._image_archive, export_config
        )

        logger.info("Data management manager initialized")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def record_store(self) -> Optional[RecordStore]:
        return self._record_store

    @property
    def image_archive(self) -> Optional[ImageArchive]:
        return self._image_archive

    @property
    def data_query(self) -> Optional[DataQuery]:
        return self._data_query

    @property
    def data_export(self) -> Optional[DataExport]:
        return self._data_export

    @property
    def data_statistics(self) -> Optional[DataStatistics]:
        return self._data_statistics

    def save_detection_result(self, detection_output: DetectionOutput,
                              product_config: Optional[ProductConfig] = None,
                              original_image: Optional[np.ndarray] = None,
                              annotated_image: Optional[np.ndarray] = None) -> bool:
        if not self._enabled:
            return False

        try:
            product_id = detection_output.product_id
            product_name = product_config.product_name if product_config else ""
            product_batch = detection_output.metadata.get("product_batch", "")
            product_model = detection_output.metadata.get("product_model", product_id)
            model_version = detection_output.metadata.get("model_version", "1.0.0")

            defect_types_list = [d.type.value for d in detection_output.defects]
            defect_types = ",".join(defect_types_list)
            defect_count = len(detection_output.defects)

            defects_detail = json.dumps(
                [d.to_dict() for d in detection_output.defects],
                ensure_ascii=False
            )

            original_image_path = ""
            annotated_image_path = ""
            thumbnail_path = ""

            if self._image_archive and self._image_archive._enabled:
                if original_image is not None and annotated_image is not None:
                    _, orig_encoded = cv2.imencode(
                        ".jpg", original_image,
                        [cv2.IMWRITE_JPEG_QUALITY, self._image_archive._compression_quality]
                    )
                    _, ann_encoded = cv2.imencode(
                        ".jpg", annotated_image,
                        [cv2.IMWRITE_JPEG_QUALITY, self._image_archive._compression_quality]
                    )

                    paths = self._image_archive.archive_image(
                        detection_id=detection_output.detection_id,
                        original_image=orig_encoded.tobytes(),
                        annotated_image=ann_encoded.tobytes(),
                        product_id=product_id,
                        timestamp=detection_output.timestamp
                    )
                    original_image_path = paths.get("original_path", "")
                    annotated_image_path = paths.get("annotated_path", "")
                    thumbnail_path = paths.get("thumbnail_path", "")

            record = DetectionRecord.create(
                detection_id=detection_output.detection_id,
                sequence_id=detection_output.sequence_id,
                product_id=product_id,
                product_name=product_name,
                product_batch=product_batch,
                product_model=product_model,
                result=detection_output.result,
                defect_types=defect_types,
                defect_count=defect_count,
                inference_time_ms=detection_output.total_inference_time_ms,
                model_version=model_version,
                line_id=detection_output.line_id,
                station_id=detection_output.station_id,
                camera_id=detection_output.image_data.camera_id if detection_output.image_data else "",
                original_image_path=original_image_path,
                annotated_image_path=annotated_image_path,
                thumbnail_path=thumbnail_path,
                defects_detail=defects_detail,
                metadata=detection_output.metadata
            )

            if self._record_store:
                success = self._record_store.save_record(record)
                if success:
                    logger.debug(f"Detection record saved: {detection_output.detection_id}")
                return success

            return True
        except Exception as e:
            logger.error(f"Failed to save detection result: {e}", exc_info=True)
            return False

    def search(self, product_id=None, product_model=None, start_time=None,
               end_time=None, defect_type=None, result=None,
               limit=50, offset=0) -> Dict[str, Any]:
        if not self._data_query:
            return {"records": [], "total_count": 0, "limit": limit, "offset": offset}
        return self._data_query.search(
            product_id=product_id, product_model=product_model,
            start_time=start_time, end_time=end_time,
            defect_type=defect_type, result=result,
            limit=limit, offset=offset
        )

    def get_record_detail(self, detection_id: str) -> Optional[Dict]:
        if not self._data_query:
            return None
        return self._data_query.get_record_detail(detection_id)

    def get_search_suggestions(self) -> Dict[str, Any]:
        if not self._data_query:
            return {"product_ids": [], "product_models": [], "defect_types": []}
        return self._data_query.get_search_suggestions()

    def export_images_zip(self, detection_ids: List[str],
                          output_filename: str = None) -> Optional[str]:
        if not self._data_export:
            return None
        return self._data_export.export_images_zip(detection_ids, output_filename)

    def export_records_excel(self, detection_ids: List[str] = None,
                             product_id=None, start_time=None, end_time=None,
                             result=None, defect_type=None,
                             output_filename: str = None) -> Optional[str]:
        if not self._data_export:
            return None
        return self._data_export.export_records_excel(
            detection_ids=detection_ids, product_id=product_id,
            start_time=start_time, end_time=end_time,
            result=result, defect_type=defect_type,
            output_filename=output_filename
        )

    def export_all(self, detection_ids: List[str] = None,
                   product_id=None, start_time=None, end_time=None,
                   result=None, defect_type=None,
                   output_filename: str = None) -> Optional[str]:
        if not self._data_export:
            return None
        return self._data_export.export_all(
            detection_ids=detection_ids, product_id=product_id,
            start_time=start_time, end_time=end_time,
            result=result, defect_type=defect_type,
            output_filename=output_filename
        )

    def get_yield_trend(self, product_id=None, start_time=None, end_time=None,
                        interval="hour") -> Dict[str, Any]:
        if not self._data_statistics:
            return {"trend": [], "product_id": product_id, "interval": interval}
        return self._data_statistics.get_yield_trend(
            product_id=product_id, start_time=start_time,
            end_time=end_time, interval=interval
        )

    def get_defect_distribution(self, product_id=None, start_time=None,
                                end_time=None) -> Dict[str, Any]:
        if not self._data_statistics:
            return {"distribution": [], "total_defects": 0, "product_id": product_id}
        return self._data_statistics.get_defect_distribution(
            product_id=product_id, start_time=start_time, end_time=end_time
        )

    def get_product_defect_ranking(self, start_time=None, end_time=None,
                                   top_n=10) -> Dict[str, Any]:
        if not self._data_statistics:
            return {"ranking": [], "top_n": top_n}
        return self._data_statistics.get_product_defect_ranking(
            start_time=start_time, end_time=end_time, top_n=top_n
        )

    def get_overview(self, start_time=None, end_time=None) -> Dict[str, Any]:
        if not self._data_statistics:
            return {}
        return self._data_statistics.get_overview(
            start_time=start_time, end_time=end_time
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "record_store_enabled": self._record_store is not None and self._record_store._enabled,
            "image_archive_enabled": self._image_archive is not None and self._image_archive._enabled,
            "tables_count": len(self._record_store.get_tables()) if self._record_store and self._record_store._enabled else 0
        }

    def close(self):
        if self._record_store:
            self._record_store.close()
        if self._image_archive:
            self._image_archive.close()
        logger.info("Data management manager closed")
