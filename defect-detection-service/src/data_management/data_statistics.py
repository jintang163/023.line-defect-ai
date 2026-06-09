from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Optional, List

from src.utils.logger import Logger

logger = Logger("data_statistics", "INFO", "./logs/defect-detection.log").logger


class DataStatistics:
    def __init__(self, record_store):
        self.record_store = record_store

    def get_yield_trend(self, product_id=None, start_time=None, end_time=None,
                        interval="hour") -> Dict[str, Any]:
        records, _ = self.record_store.query_records(
            product_id=product_id, start_time=start_time, end_time=end_time,
            limit=100000
        )

        grouped: Dict[str, List] = defaultdict(list)
        for record in records:
            timestamp = record.get("timestamp", 0)
            if not timestamp:
                continue
            try:
                dt = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                continue

            if interval == "hour":
                period = dt.strftime("%Y-%m-%d %H:00")
            elif interval == "day":
                period = dt.strftime("%Y-%m-%d")
            elif interval == "shift":
                shift_index = dt.hour // 8
                shift_start = shift_index * 8
                period = dt.strftime("%Y-%m-%d ") + f"{shift_start:02d}:00"
            else:
                period = dt.strftime("%Y-%m-%d %H:00")
            grouped[period].append(record)

        trend = []
        for period in sorted(grouped.keys()):
            group_records = grouped[period]
            total = len(group_records)
            ok = sum(1 for r in group_records if r.get("result") == "OK")
            ng = total - ok
            yield_rate = round((ok / total) * 100, 2) if total > 0 else 0.0
            trend.append({
                "period": period,
                "total": total,
                "ok": ok,
                "ng": ng,
                "yield_rate": yield_rate
            })

        return {
            "trend": trend,
            "product_id": product_id,
            "interval": interval
        }

    def get_defect_distribution(self, product_id=None, start_time=None,
                                end_time=None) -> Dict[str, Any]:
        records, _ = self.record_store.query_records(
            product_id=product_id, start_time=start_time, end_time=end_time,
            result="NG", limit=100000
        )

        defect_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            defect_types_str = record.get("defect_types", "")
            if defect_types_str:
                for dt in defect_types_str.split(","):
                    dt = dt.strip()
                    if dt:
                        defect_counts[dt] += 1

        total_defects = sum(defect_counts.values())
        distribution = []
        for defect_type, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = round((count / total_defects) * 100, 1) if total_defects > 0 else 0.0
            distribution.append({
                "defect_type": defect_type,
                "count": count,
                "percentage": percentage
            })

        return {
            "distribution": distribution,
            "total_defects": total_defects,
            "product_id": product_id
        }

    def get_product_defect_ranking(self, start_time=None, end_time=None,
                                   top_n=10) -> Dict[str, Any]:
        records, _ = self.record_store.query_records(
            start_time=start_time, end_time=end_time, limit=100000
        )

        product_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "ng_count": 0, "product_name": ""}
        )
        for record in records:
            pid = record.get("product_id", "")
            product_data[pid]["total"] += 1
            product_data[pid]["product_name"] = record.get("product_name", pid)
            if record.get("result") == "NG":
                product_data[pid]["ng_count"] += 1

        ranking = []
        for pid, data in product_data.items():
            defect_rate = round((data["ng_count"] / data["total"]) * 100, 2) if data["total"] > 0 else 0.0
            ranking.append({
                "product_id": pid,
                "product_name": data["product_name"],
                "total": data["total"],
                "ng_count": data["ng_count"],
                "defect_rate": defect_rate
            })
        ranking.sort(key=lambda x: x["ng_count"], reverse=True)
        ranking = ranking[:top_n]

        return {
            "ranking": ranking,
            "top_n": top_n
        }

    def get_overview(self, start_time=None, end_time=None) -> Dict[str, Any]:
        records, _ = self.record_store.query_records(
            start_time=start_time, end_time=end_time, limit=100000
        )

        total_records = len(records)
        ok_count = sum(1 for r in records if r.get("result") == "OK")
        ng_count = total_records - ok_count
        yield_rate = round((ok_count / total_records) * 100, 2) if total_records > 0 else 0.0

        defect_counts: Dict[str, int] = defaultdict(int)
        inference_times = []
        for record in records:
            if record.get("result") == "NG":
                defect_types_str = record.get("defect_types", "")
                if defect_types_str:
                    for dt in defect_types_str.split(","):
                        dt = dt.strip()
                        if dt:
                            defect_counts[dt] += 1
            inference_time = record.get("inference_time_ms")
            if inference_time is not None:
                try:
                    inference_times.append(float(inference_time))
                except (ValueError, TypeError):
                    pass

        top_defect_type = ""
        if defect_counts:
            top_defect_type = max(defect_counts, key=defect_counts.get)

        avg_inference_time_ms = round(
            sum(inference_times) / len(inference_times), 2
        ) if inference_times else 0.0

        return {
            "total_records": total_records,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "yield_rate": yield_rate,
            "top_defect_type": top_defect_type,
            "avg_inference_time_ms": avg_inference_time_ms
        }
