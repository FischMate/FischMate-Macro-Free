from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainedVisionDetection:
    class_name: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class TrainedVisionFrame:
    available: bool
    ran_inference: bool
    raw: tuple[TrainedVisionDetection, ...]
    trusted: tuple[TrainedVisionDetection, ...]
    inference_ms: float
    error: str = ""

    def detections(
        self,
        class_name: str,
        *,
        trusted: bool = False,
    ) -> list[TrainedVisionDetection]:
        source = self.trusted if trusted else self.raw
        return [item for item in source if item.class_name == class_name]


@dataclass
class _SingleTrack:
    detection: TrainedVisionDetection | None = None
    confirmations: int = 0
    trusted: bool = False
    timestamp_ms: float | None = None


class TrainedRodVisionAssist:
    """Optional screen-pixel ONNX assist, hard-gated to two special rods.

    It never emits controller commands and never reads the Roblox process. The
    owning rod detector remains responsible for rail geometry, continuity, and
    deciding whether an observation is safe to use.
    """

    ALLOWED_DETECTOR_IDS = frozenset({"noiseform", "pinions_aria"})
    PROFILE_CLASSES = {
        "noiseform": frozenset({"bar", "stick", "cue", "tile"}),
        "pinions_aria": frozenset({"bar", "stick", "note"}),
    }
    MAXIMUM_PER_CLASS = {"bar": 1, "stick": 1, "note": 12, "cue": 1, "tile": 6}

    def __init__(
        self,
        profile: dict,
        detector_id: str,
        config_key: str = "trained_vision",
    ):
        if detector_id not in self.ALLOWED_DETECTOR_IDS:
            raise ValueError(f"Trained rod vision is forbidden for {detector_id!r}")
        self.detector_id = detector_id
        self.config_key = config_key
        self.config = dict(profile.get("detection", {}).get(config_key, {}))
        configured_classes = self.config.get("enabled_classes")
        self.enabled_classes = (
            self.PROFILE_CLASSES[detector_id]
            if configured_classes is None
            else self.PROFILE_CLASSES[detector_id].intersection(
                str(value) for value in configured_classes
            )
        )
        self.enabled = bool(self.config.get("enabled", False))
        self.network = None
        self.classes: list[str] = []
        self.input_width = 0
        self.input_height = 0
        self.stride = 8
        self.crop_normalized: tuple[float, float, float, float] | None = None
        self.error = ""
        self.last_inference_timestamp_ms: float | None = None
        self.last_frame = TrainedVisionFrame(False, False, (), (), 0.0, "disabled")
        self.single_tracks = {name: _SingleTrack() for name in ("bar", "stick", "cue")}
        if not self.enabled:
            return
        try:
            relative_model = Path(
                str(
                    self.config.get(
                        "model",
                        "assets/rod_vision/pinions_noiseform_v2.onnx",
                    )
                )
            )
            model_path = relative_model if relative_model.is_absolute() else _PROJECT_ROOT / relative_model
            metadata_path = model_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_ids = frozenset(str(value) for value in metadata["allowed_detector_ids"])
            if not model_ids or not model_ids.issubset(self.ALLOWED_DETECTOR_IDS):
                raise RuntimeError("model metadata violates the two-rod safety boundary")
            if detector_id not in model_ids:
                raise RuntimeError(f"model is not authorized for {detector_id}")
            self.classes = [str(value) for value in metadata["classes"]]
            self.input_width = int(metadata["input_width"])
            self.input_height = int(metadata["input_height"])
            self.stride = int(metadata["output_stride"])
            if metadata.get("crop_normalized") is not None:
                values = tuple(float(value) for value in metadata["crop_normalized"])
                if len(values) != 4:
                    raise RuntimeError("invalid trained-vision crop metadata")
                self.crop_normalized = values
            self.network = cv2.dnn.readNetFromONNX(str(model_path))
            self.last_frame = TrainedVisionFrame(True, False, (), (), 0.0)
        except Exception as exc:  # Optional assist must never prevent launch.
            self.error = f"{type(exc).__name__}: {exc}"
            self.network = None
            self.last_frame = TrainedVisionFrame(False, False, (), (), 0.0, self.error)

    @property
    def available(self) -> bool:
        return self.enabled and self.network is not None

    def reset(self) -> None:
        self.last_inference_timestamp_ms = None
        self.single_tracks = {name: _SingleTrack() for name in ("bar", "stick", "cue")}
        self.last_frame = TrainedVisionFrame(
            self.available,
            False,
            (),
            (),
            0.0,
            self.error,
        )

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: float) -> TrainedVisionFrame:
        if not self.available:
            return self.last_frame
        interval_ms = float(self.config.get("interval_ms", 50.0))
        if (
            self.last_inference_timestamp_ms is not None
            and timestamp_ms - self.last_inference_timestamp_ms < interval_ms
        ):
            return TrainedVisionFrame(
                True,
                False,
                self.last_frame.raw,
                self.last_frame.trusted,
                self.last_frame.inference_ms,
            )

        started = time.perf_counter()
        try:
            raw = tuple(self._infer(frame_bgr))
            trusted = tuple(self._update_trust(raw, timestamp_ms, frame_bgr.shape[1]))
            inference_ms = (time.perf_counter() - started) * 1000.0
            self.last_inference_timestamp_ms = timestamp_ms
            self.last_frame = TrainedVisionFrame(
                True,
                True,
                raw,
                trusted,
                inference_ms,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.last_frame = TrainedVisionFrame(
                False,
                True,
                (),
                (),
                (time.perf_counter() - started) * 1000.0,
                self.error,
            )
        return self.last_frame

    def _infer(self, frame_bgr: np.ndarray) -> list[TrainedVisionDetection]:
        assert self.network is not None
        frame_height, frame_width = frame_bgr.shape[:2]
        crop_left = 0
        crop_top = 0
        crop_right = frame_width
        crop_bottom = frame_height
        if self.crop_normalized is not None:
            crop_left = round(frame_width * self.crop_normalized[0])
            crop_top = round(frame_height * self.crop_normalized[1])
            crop_right = round(frame_width * self.crop_normalized[2])
            crop_bottom = round(frame_height * self.crop_normalized[3])
        source = frame_bgr[crop_top:crop_bottom, crop_left:crop_right]
        if source.size == 0:
            raise RuntimeError("trained-vision crop is empty")
        source_height, source_width = source.shape[:2]
        resized = cv2.resize(
            source,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_AREA,
        )
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, swapRB=True)
        self.network.setInput(blob)
        heatmap_logits, sizes, offsets = self.network.forward(
            ["heatmap_logits", "sizes", "offsets"]
        )
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(heatmap_logits[0], -30.0, 30.0))
        )
        thresholds = dict(self.config.get("thresholds", {}))
        scale_x = source_width / self.input_width
        scale_y = source_height / self.input_height
        allowed = self.enabled_classes
        detections: list[TrainedVisionDetection] = []
        for class_index, class_name in enumerate(self.classes):
            if class_name not in allowed:
                continue
            channel = probabilities[class_index]
            threshold = float(thresholds.get(class_name, 0.35))
            local_maximum = cv2.dilate(channel, np.ones((3, 3), np.uint8))
            points = np.argwhere(
                (channel >= threshold) & (channel >= local_maximum - 1e-7)
            )
            if points.size == 0:
                continue
            points = sorted(
                ((int(y), int(x)) for y, x in points),
                key=lambda point: float(channel[point]),
                reverse=True,
            )[: max(20, self.MAXIMUM_PER_CLASS[class_name] * 4)]
            candidates: list[TrainedVisionDetection] = []
            xywh: list[list[int]] = []
            scores: list[float] = []
            for grid_y, grid_x in points:
                confidence = float(channel[grid_y, grid_x])
                width_grid = max(
                    0.5,
                    float(sizes[0, class_index * 2, grid_y, grid_x]),
                )
                height_grid = max(
                    0.5,
                    float(sizes[0, class_index * 2 + 1, grid_y, grid_x]),
                )
                center_x = (
                    grid_x + float(offsets[0, class_index * 2, grid_y, grid_x])
                ) * self.stride
                center_y = (
                    grid_y
                    + float(offsets[0, class_index * 2 + 1, grid_y, grid_x])
                ) * self.stride
                box_width = width_grid * self.stride
                box_height = height_grid * self.stride
                left = max(
                    crop_left,
                    crop_left + round((center_x - box_width / 2) * scale_x),
                )
                top = max(
                    crop_top,
                    crop_top + round((center_y - box_height / 2) * scale_y),
                )
                right = min(
                    crop_right,
                    crop_left + round((center_x + box_width / 2) * scale_x),
                )
                bottom = min(
                    crop_bottom,
                    crop_top + round((center_y + box_height / 2) * scale_y),
                )
                if right <= left or bottom <= top:
                    continue
                candidate = TrainedVisionDetection(
                    class_name,
                    confidence,
                    left,
                    top,
                    right,
                    bottom,
                )
                candidates.append(candidate)
                xywh.append([left, top, right - left, bottom - top])
                scores.append(confidence)
            if not candidates:
                continue
            kept = cv2.dnn.NMSBoxes(xywh, scores, threshold, 0.35)
            kept_indices = [int(value) for value in np.asarray(kept).reshape(-1)]
            detections.extend(
                [candidates[index] for index in kept_indices][
                    : self.MAXIMUM_PER_CLASS[class_name]
                ]
            )
        return detections

    def _update_trust(
        self,
        detections: tuple[TrainedVisionDetection, ...],
        timestamp_ms: float,
        frame_width: int,
    ) -> list[TrainedVisionDetection]:
        trusted = [
            item for item in detections if item.class_name in {"note", "tile"}
        ]
        tolerance_by_class = {"bar": 0.12, "stick": 0.06, "cue": 0.08}
        for class_name, track in self.single_tracks.items():
            confirmations_required = int(
                self.config.get(
                    f"{class_name}_confirmations",
                    self.config.get("confirmations", 2),
                )
            )
            candidates = [item for item in detections if item.class_name == class_name]
            candidate = max(candidates, key=lambda item: item.confidence, default=None)
            if candidate is None:
                track.detection = None
                track.confirmations = 0
                track.trusted = False
                track.timestamp_ms = timestamp_ms
                continue
            tolerance = frame_width * float(
                self.config.get(
                    f"{class_name}_continuity_ratio",
                    tolerance_by_class[class_name],
                )
            )
            if (
                track.detection is not None
                and abs(candidate.center_x - track.detection.center_x) <= tolerance
            ):
                track.confirmations += 1
            else:
                track.confirmations = 1
            track.detection = candidate
            track.timestamp_ms = timestamp_ms
            track.trusted = track.confirmations >= confirmations_required
            if track.trusted:
                trusted.append(candidate)
        return trusted
