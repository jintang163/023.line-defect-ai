from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from src.utils.schemas import (
    Defect, BoundingBox, Point, DefectSeverity, DefectType,
    DetectionOutput, ProductConfig, ROI
)
from src.utils.logger import Logger

logger = Logger().logger


class ResultAnnotator:
    def __init__(self):
        self._colors: Dict[DefectSeverity, Tuple[int, int, int]] = {
            DefectSeverity.CRITICAL: (0, 0, 255),
            DefectSeverity.MAJOR: (0, 165, 255),
            DefectSeverity.MINOR: (0, 255, 255),
            DefectSeverity.WARNING: (0, 255, 0)
        }

        self._defect_labels: Dict[DefectType, str] = {
            DefectType.SCRATCH: "划痕",
            DefectType.DIRT: "脏污",
            DefectType.DENT: "凹痕",
            DefectType.CRACK: "裂纹",
            DefectType.MISSING: "缺失",
            DefectType.STAIN: "污渍",
            DefectType.DEFORMATION: "变形",
            DefectType.BUBBLE: "气泡",
            DefectType.UNKNOWN: "未知"
        }

        self._line_thickness = 2
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 1

    def annotate(self, image: np.ndarray, defects: List[Defect],
                 product_config: Optional[ProductConfig] = None,
                 show_roi: bool = True, show_confidence: bool = True,
                 show_area: bool = True, show_severity: bool = True) -> np.ndarray:
        if image is None or image.size == 0:
            logger.warning("Empty image provided for annotation")
            return image

        annotated = image.copy()

        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        if product_config and show_roi:
            annotated = self._draw_rois(annotated, product_config.rois)

        for defect in defects:
            annotated = self._draw_defect(
                annotated, defect,
                show_confidence=show_confidence,
                show_area=show_area,
                show_severity=show_severity
            )

        return annotated

    def _draw_rois(self, image: np.ndarray, rois: List[ROI]) -> np.ndarray:
        for roi in rois:
            if not roi.enabled:
                continue

            x, y, w, h = roi.to_tuple()
            color = (255, 0, 0)
            thickness = 2

            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

            if roi.name:
                label = roi.name
                label_size = cv2.getTextSize(label, self._font, self._font_scale, self._font_thickness)[0]
                cv2.rectangle(
                    image,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0] + 10, y),
                    color, -1
                )
                cv2.putText(
                    image, label,
                    (x + 5, y - 5),
                    self._font, self._font_scale,
                    (255, 255, 255), self._font_thickness,
                    cv2.LINE_AA
                )

        return image

    def _draw_defect(self, image: np.ndarray, defect: Defect,
                     show_confidence: bool = True,
                     show_area: bool = True,
                     show_severity: bool = True) -> np.ndarray:
        color = self._colors.get(defect.severity, (0, 255, 0))
        thickness = self._get_thickness_by_severity(defect.severity)

        if defect.contour and len(defect.contour) >= 3:
            image = self._draw_contour(image, defect.contour, color, thickness)
        elif defect.bbox:
            image = self._draw_bbox(image, defect.bbox, color, thickness)

        label_parts = []
        defect_label = self._defect_labels.get(defect.type, defect.type.value)
        label_parts.append(defect_label)

        if show_severity:
            severity_map = {
                DefectSeverity.CRITICAL: "严重",
                DefectSeverity.MAJOR: "主要",
                DefectSeverity.MINOR: "轻微",
                DefectSeverity.WARNING: "警告"
            }
            label_parts.append(severity_map.get(defect.severity, ""))

        if show_confidence:
            label_parts.append(f"{defect.confidence:.2f}")

        if show_area and defect.area_mm2 > 0:
            label_parts.append(f"{defect.area_mm2:.3f}mm²")

        label = " | ".join([p for p in label_parts if p])

        image = self._draw_label(
            image, defect.bbox, label, color
        )

        return image

    def _draw_contour(self, image: np.ndarray, contour: List[Point],
                      color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        points = np.array([[int(p.x), int(p.y)] for p in contour], dtype=np.int32)
        points = points.reshape((-1, 1, 2))

        cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)

        overlay = image.copy()
        alpha = 0.3
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        return image

    def _draw_bbox(self, image: np.ndarray, bbox: BoundingBox,
                   color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        line_length = 20
        corner_thickness = thickness + 1
        corners = [
            ((x1, y1), (x1 + line_length, y1), (x1, y1 + line_length)),
            ((x2, y1), (x2 - line_length, y1), (x2, y1 + line_length)),
            ((x1, y2), (x1 + line_length, y2), (x1, y2 - line_length)),
            ((x2, y2), (x2 - line_length, y2), (x2, y2 - line_length))
        ]

        for origin, h_line, v_line in corners:
            cv2.line(image, origin, h_line, color, corner_thickness, cv2.LINE_AA)
            cv2.line(image, origin, v_line, color, corner_thickness, cv2.LINE_AA)

        return image

    def _draw_label(self, image: np.ndarray, bbox: BoundingBox, label: str,
                    color: Tuple[int, int, int]) -> np.ndarray:
        if not label:
            return image

        label_size = cv2.getTextSize(
            label, self._font, self._font_scale, self._font_thickness
        )[0]

        label_width = label_size[0] + 10
        label_height = label_size[1] + 8

        x = int(bbox.x1)
        y = int(bbox.y1)

        if y - label_height < 0:
            y = int(bbox.y2) + label_height
        else:
            y = y - 2

        if x + label_width > image.shape[1]:
            x = image.shape[1] - label_width

        cv2.rectangle(
            image,
            (x, y - label_height),
            (x + label_width, y),
            color, -1, cv2.LINE_AA
        )

        cv2.putText(
            image, label,
            (x + 5, y - 2),
            self._font, self._font_scale,
            (255, 255, 255), self._font_thickness,
            cv2.LINE_AA
        )

        return image

    def _get_thickness_by_severity(self, severity: DefectSeverity) -> int:
        thickness_map = {
            DefectSeverity.CRITICAL: 4,
            DefectSeverity.MAJOR: 3,
            DefectSeverity.MINOR: 2,
            DefectSeverity.WARNING: 1
        }
        return thickness_map.get(severity, self._line_thickness)

    def draw_result_banner(self, image: np.ndarray, result: str,
                           total_defects: int, inference_time: float) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        banner_height = 50
        banner = np.zeros((banner_height, image.shape[1], 3), dtype=np.uint8)

        if result == "OK":
            banner[:] = (0, 128, 0)
            result_color = (0, 255, 0)
        else:
            banner[:] = (0, 0, 128)
            result_color = (0, 0, 255)

        result_text = f"检测结果: {result}"
        defects_text = f"缺陷数量: {total_defects}"
        time_text = f"推理时间: {inference_time:.1f}ms"

        cv2.putText(banner, result_text, (20, 30), self._font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(banner, defects_text, (250, 30), self._font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, time_text, (450, 30), self._font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return np.vstack([banner, image])

    def draw_legend(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        legend_width = 180
        legend_height = 120
        legend_x = image.shape[1] - legend_width - 10
        legend_y = 70

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (50, 50, 50), -1
        )
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(image, "缺陷等级", (legend_x + 10, legend_y + 25),
                    self._font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        severities = [
            (DefectSeverity.CRITICAL, "严重"),
            (DefectSeverity.MAJOR, "主要"),
            (DefectSeverity.MINOR, "轻微"),
            (DefectSeverity.WARNING, "警告")
        ]

        for i, (severity, label) in enumerate(severities):
            color = self._colors[severity]
            y = legend_y + 45 + i * 20

            cv2.rectangle(image, (legend_x + 10, y - 8), (legend_x + 25, y + 4), color, -1)
            cv2.putText(image, label, (legend_x + 35, y),
                        self._font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def create_composite_view(self, original_image: np.ndarray, annotated_image: np.ndarray,
                              detection_output: DetectionOutput,
                              heatmap: Optional[np.ndarray] = None) -> np.ndarray:
        result_banner = self.draw_result_banner(
            annotated_image,
            detection_output.result.value,
            len(detection_output.defects),
            detection_output.total_inference_time_ms
        )

        result_banner = self.draw_legend(result_banner)

        if heatmap is not None:
            h, w = original_image.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            if len(heatmap_resized.shape) == 2:
                heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            combined = np.hstack([result_banner, np.vstack([
                np.zeros((50, w, 3), dtype=np.uint8),
                heatmap_resized
            ])])
            return combined

        return result_banner

    def set_defect_label(self, defect_type: DefectType, label: str):
        self._defect_labels[defect_type] = label

    def set_severity_color(self, severity: DefectSeverity, color: Tuple[int, int, int]):
        self._colors[severity] = color

    def set_line_thickness(self, thickness: int):
        self._line_thickness = max(1, thickness)

    def set_font_scale(self, scale: float):
        self._font_scale = max(0.3, scale)
