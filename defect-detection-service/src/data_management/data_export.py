import os
import time
import csv
import zipfile
import uuid
import json
import threading
from typing import List, Optional, Dict, Any

from src.utils.logger import Logger

logger = Logger("data_export", "INFO", "./logs/defect-detection.log").logger


class DataExport:
    def __init__(self, record_store, image_archive, config: dict = None):
        self._lock = threading.RLock()
        self.record_store = record_store
        self.image_archive = image_archive
        if config is None:
            config = {}
        self.temp_dir = config.get("temp_dir", "./data/export_temp")
        self.max_export_records = config.get("max_export_records", 10000)
        os.makedirs(self.temp_dir, exist_ok=True)

    def export_images_zip(self, detection_ids: List[str], output_filename: str = None) -> Optional[str]:
        with self._lock:
            if not detection_ids:
                logger.warning("No detection_ids provided for image export")
                return None

            if len(detection_ids) > self.max_export_records:
                logger.warning(f"Detection IDs count {len(detection_ids)} exceeds max, truncating")
                detection_ids = detection_ids[:self.max_export_records]

            if output_filename is None:
                output_filename = f"images_export_{uuid.uuid4().hex[:8]}.zip"
            if not output_filename.endswith(".zip"):
                output_filename += ".zip"

            output_path = os.path.join(self.temp_dir, output_filename)

            try:
                with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for det_id in detection_ids:
                        try:
                            record = self.record_store.get_record_by_detection_id(det_id)
                            if record is None:
                                logger.warning(f"Record not found for detection_id: {det_id}")
                                continue

                            product_id = record.get("product_id", "unknown")
                            timestamp = record.get("timestamp", 0)
                            from datetime import datetime
                            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "unknown_date"
                            zip_folder = f"{product_id}/{date_str}/{det_id}/"

                            original_path = record.get("original_image_path", "")
                            annotated_path = record.get("annotated_image_path", "")

                            if original_path:
                                image_data = self.image_archive.get_image(original_path)
                                if image_data:
                                    zf.writestr(zip_folder + "original.jpg", image_data)

                            if annotated_path:
                                image_data = self.image_archive.get_image(annotated_path)
                                if image_data:
                                    zf.writestr(zip_folder + "annotated.jpg", image_data)
                        except Exception as e:
                            logger.error(f"Error exporting images for detection_id {det_id}: {e}")
                            continue

                logger.info(f"Images zip exported to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to create images zip: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None

    def export_records_excel(self, detection_ids: List[str] = None, product_id=None,
                             start_time=None, end_time=None, result=None,
                             defect_type=None, output_filename: str = None) -> Optional[str]:
        with self._lock:
            if output_filename is None:
                output_filename = f"records_export_{uuid.uuid4().hex[:8]}.csv"
            if not output_filename.endswith(".csv"):
                output_filename += ".csv"

            output_path = os.path.join(self.temp_dir, output_filename)

            try:
                if detection_ids:
                    records = []
                    for det_id in detection_ids:
                        rec = self.record_store.get_record_by_detection_id(det_id)
                        if rec:
                            records.append(rec)
                else:
                    records, _ = self.record_store.query_records(
                        product_id=product_id,
                        start_time=start_time,
                        end_time=end_time,
                        result=result,
                        defect_type=defect_type,
                        limit=self.max_export_records
                    )

                if len(records) > self.max_export_records:
                    logger.warning(f"Records count {len(records)} exceeds max, truncating")
                    records = records[:self.max_export_records]

                headers = [
                    "detection_id", "sequence_id", "product_id", "product_name",
                    "product_batch", "product_model", "result", "defect_types",
                    "defect_count", "inference_time_ms", "model_version",
                    "timestamp", "line_id", "station_id", "camera_id"
                ]

                with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                    writer.writeheader()
                    for rec in records:
                        row = {}
                        for h in headers:
                            val = rec.get(h, "")
                            if isinstance(val, (list, dict)):
                                val = json.dumps(val, ensure_ascii=False)
                            row[h] = val
                        writer.writerow(row)

                logger.info(f"Records CSV exported to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to export records CSV: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None

    def export_all(self, detection_ids: List[str] = None, product_id=None,
                   start_time=None, end_time=None, result=None,
                   defect_type=None, output_filename: str = None) -> Optional[str]:
        with self._lock:
            if output_filename is None:
                output_filename = f"full_export_{uuid.uuid4().hex[:8]}.zip"
            if not output_filename.endswith(".zip"):
                output_filename += ".zip"

            output_path = os.path.join(self.temp_dir, output_filename)

            try:
                if detection_ids:
                    records = []
                    for det_id in detection_ids:
                        rec = self.record_store.get_record_by_detection_id(det_id)
                        if rec:
                            records.append(rec)
                else:
                    records, _ = self.record_store.query_records(
                        product_id=product_id,
                        start_time=start_time,
                        end_time=end_time,
                        result=result,
                        defect_type=defect_type,
                        limit=self.max_export_records
                    )

                if len(records) > self.max_export_records:
                    logger.warning(f"Records count {len(records)} exceeds max, truncating")
                    records = records[:self.max_export_records]

                headers = [
                    "detection_id", "sequence_id", "product_id", "product_name",
                    "product_batch", "product_model", "result", "defect_types",
                    "defect_count", "inference_time_ms", "model_version",
                    "timestamp", "line_id", "station_id", "camera_id"
                ]

                csv_path = os.path.join(self.temp_dir, f"_temp_records_{uuid.uuid4().hex[:8]}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                    writer.writeheader()
                    for rec in records:
                        row = {}
                        for h in headers:
                            val = rec.get(h, "")
                            if isinstance(val, (list, dict)):
                                val = json.dumps(val, ensure_ascii=False)
                            row[h] = val
                        writer.writerow(row)

                with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(csv_path, "records.csv")

                    for rec in records:
                        try:
                            det_id = rec.get("detection_id", "")
                            product_id_val = rec.get("product_id", "unknown")
                            timestamp = rec.get("timestamp", 0)
                            from datetime import datetime
                            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "unknown_date"
                            zip_folder = f"images/{product_id_val}/{date_str}/{det_id}/"

                            original_path = rec.get("original_image_path", "")
                            annotated_path = rec.get("annotated_image_path", "")

                            if original_path:
                                image_data = self.image_archive.get_image(original_path)
                                if image_data:
                                    zf.writestr(zip_folder + "original.jpg", image_data)

                            if annotated_path:
                                image_data = self.image_archive.get_image(annotated_path)
                                if image_data:
                                    zf.writestr(zip_folder + "annotated.jpg", image_data)
                        except Exception as e:
                            logger.error(f"Error adding images for detection_id {det_id}: {e}")
                            continue

                if os.path.exists(csv_path):
                    os.remove(csv_path)

                logger.info(f"Full export created at {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to create full export: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None

    def cleanup_temp(self):
        with self._lock:
            now = time.time()
            if not os.path.exists(self.temp_dir):
                return
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        file_age = now - os.path.getmtime(filepath)
                        if file_age > 3600:
                            os.remove(filepath)
                            logger.info(f"Cleaned up temp file: {filepath}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {filepath}: {e}")
