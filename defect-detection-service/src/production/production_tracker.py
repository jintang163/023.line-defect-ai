from typing import Dict, Any, Optional, List, Tuple, Callable
import threading
import time
from collections import defaultdict, deque

from src.utils.schemas import (
    YieldSnapshot, ProductionStats, DetectionOutput,
    DetectionResult, Defect, ProductConfig
)
from src.utils.logger import Logger

logger = Logger().logger


class ProductionTracker:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._snapshot_interval_count = config.get("snapshot_interval_count", 1000)
        self._max_snapshots_per_product = config.get("max_snapshots_per_product", 1000)

        self._stats_lock = threading.RLock()
        self._product_stats: Dict[str, ProductionStats] = {}
        self._product_snapshots: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._max_snapshots_per_product)
        )
        self._period_start_times: Dict[str, float] = {}

        self._snapshot_callbacks: List[Callable[[YieldSnapshot], None]] = []

        logger.info(f"ProductionTracker 已初始化，快照间隔: {self._snapshot_interval_count} 件")
        logger.info("连续NG急停保护由 AlertManager 统一处理")

    def _get_or_create_stats(self, product_id: str) -> ProductionStats:
        if product_id not in self._product_stats:
            current_time = time.time()
            self._product_stats[product_id] = ProductionStats(
                product_id=product_id,
                current_batch_start=current_time,
                last_snapshot_time=current_time
            )
            self._period_start_times[product_id] = current_time
            logger.info(f"为产品 {product_id} 创建新的生产统计记录")
        return self._product_stats[product_id]

    def _update_defect_distribution(self, stats: ProductionStats, defects: List[Defect]):
        for defect in defects:
            defect_type = defect.type.value
            stats.defect_distribution[defect_type] = stats.defect_distribution.get(defect_type, 0) + 1

    def _should_create_snapshot(self, stats: ProductionStats) -> bool:
        return stats.total_count > 0 and stats.total_count % self._snapshot_interval_count == 0

    def _create_snapshot(self, stats: ProductionStats) -> YieldSnapshot:
        period_start = self._period_start_times.get(stats.product_id, stats.current_batch_start)
        period_end = time.time()

        snapshot = YieldSnapshot.create(
            product_id=stats.product_id,
            total_count=stats.total_count,
            ok_count=stats.ok_count,
            ng_count=stats.ng_count,
            period_start=period_start,
            period_end=period_end,
            defect_distribution=dict(stats.defect_distribution)
        )

        self._period_start_times[stats.product_id] = period_end
        stats.last_snapshot_time = period_end

        logger.info(
            f"生成良率快照 - 产品: {stats.product_id}, "
            f"总数: {stats.total_count}, 良率: {snapshot.yield_rate:.2f}%"
        )

        return snapshot

    def _notify_snapshot_callbacks(self, snapshot: YieldSnapshot):
        for callback in self._snapshot_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"快照回调执行失败: {e}", exc_info=True)

    def process_result(
        self,
        detection_output: DetectionOutput,
        product_config: Optional[ProductConfig]
    ) -> Tuple[bool, Optional[YieldSnapshot]]:
        product_id = detection_output.product_id
        is_emergency_stop = False
        snapshot = None

        with self._stats_lock:
            stats = self._get_or_create_stats(product_id)

            stats.total_count += 1

            if detection_output.result == DetectionResult.OK:
                stats.ok_count += 1
                stats.consecutive_ng_count = 0
            else:
                stats.ng_count += 1
                stats.consecutive_ng_count += 1

                if stats.consecutive_ng_count > stats.max_consecutive_ng:
                    stats.max_consecutive_ng = stats.consecutive_ng_count

            self._update_defect_distribution(stats, detection_output.defects)

            if self._should_create_snapshot(stats):
                snapshot = self._create_snapshot(stats)
                self._product_snapshots[product_id].append(snapshot)
                self._notify_snapshot_callbacks(snapshot)

        return is_emergency_stop, snapshot

    def reset_stats(self, product_id: str):
        with self._stats_lock:
            if product_id in self._product_stats:
                current_time = time.time()
                old_stats = self._product_stats[product_id]

                if old_stats.total_count > 0:
                    final_snapshot = YieldSnapshot.create(
                        product_id=product_id,
                        total_count=old_stats.total_count,
                        ok_count=old_stats.ok_count,
                        ng_count=old_stats.ng_count,
                        period_start=old_stats.current_batch_start,
                        period_end=current_time,
                        defect_distribution=dict(old_stats.defect_distribution),
                        details={"reset_reason": "product_switch"}
                    )
                    self._product_snapshots[product_id].append(final_snapshot)
                    self._notify_snapshot_callbacks(final_snapshot)

                self._product_stats[product_id] = ProductionStats(
                    product_id=product_id,
                    current_batch_start=current_time,
                    last_snapshot_time=current_time
                )
                self._period_start_times[product_id] = current_time

                logger.info(f"产品 {product_id} 统计数据已重置")
            else:
                logger.warning(f"尝试重置不存在的产品统计: {product_id}")

    def get_stats(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        with self._stats_lock:
            if product_id:
                stats = self._get_or_create_stats(product_id)
                result = stats.to_dict()
                result["current_run_time_seconds"] = time.time() - stats.current_batch_start
                return result
            else:
                result = {}
                for pid, stats in self._product_stats.items():
                    stats_dict = stats.to_dict()
                    stats_dict["current_run_time_seconds"] = time.time() - stats.current_batch_start
                    result[pid] = stats_dict
                return result

    def get_snapshots(
        self,
        product_id: Optional[str] = None,
        limit: int = 100
    ) -> List[YieldSnapshot]:
        with self._stats_lock:
            if product_id:
                snapshots = list(self._product_snapshots.get(product_id, []))
                return snapshots[-limit:]
            else:
                all_snapshots = []
                for snapshots in self._product_snapshots.values():
                    all_snapshots.extend(snapshots)
                all_snapshots.sort(key=lambda s: s.timestamp)
                return all_snapshots[-limit:]

    def register_snapshot_callback(self, callback: Callable[[YieldSnapshot], None]):
        if callback not in self._snapshot_callbacks:
            self._snapshot_callbacks.append(callback)
            logger.info("已注册快照上传回调")

    def get_defect_distribution(
        self,
        product_id: Optional[str] = None
    ) -> Dict[str, int]:
        with self._stats_lock:
            if product_id:
                stats = self._get_or_create_stats(product_id)
                return dict(stats.defect_distribution)
            else:
                total_distribution: Dict[str, int] = defaultdict(int)
                for stats in self._product_stats.values():
                    for defect_type, count in stats.defect_distribution.items():
                        total_distribution[defect_type] += count
                return dict(total_distribution)

    @property
    def snapshot_interval_count(self) -> int:
        return self._snapshot_interval_count
