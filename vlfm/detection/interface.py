"""Target detection behind one interface (design doc 3.13).

A detector takes the RGB image (and optionally the simulator's observations) and
returns the target's detection boxes and one mask per box.

    vlm: VLFM's detectors (YOLOv7 / GroundingDINO) and Mobile SAM, as the policy calls
        them today.
    gt: ground truth from HM3D's semantic annotations, to separate the effect of
        cooperation from detection errors. Reserved for phase 6.

The policies do not go through this interface yet: switching them over changes the
detection path, which needs a regression run on the GPU. That is done together with
the gt mode.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vlfm.vlm.detections import ObjectDetections

DETECTORS = ("vlm", "gt")


class TargetDetector(ABC):
    @abstractmethod
    def detect(
        self, rgb: np.ndarray, observations: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObjectDetections, List[np.ndarray]]:
        """Detections of the current target in `rgb`, and a mask (uint8, rgb's height
        and width) for each detection box."""


class VLMDetector(TargetDetector):
    """VLFM's detector and segmenter, called the way BaseObjectNavPolicy calls them."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def detect(
        self, rgb: np.ndarray, observations: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObjectDetections, List[np.ndarray]]:
        detections = self._policy._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        masks = []
        for box in detections.boxes:
            box_px = np.asarray(box) * np.array([width, height, width, height])
            masks.append(self._policy._mobile_sam.segment_bbox(rgb, box_px.tolist()))
        return detections, masks


class GroundTruthDetector(TargetDetector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("The ground-truth detector is planned for phase 6")

    def detect(
        self, rgb: np.ndarray, observations: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObjectDetections, List[np.ndarray]]:
        raise NotImplementedError


def make_detector(kind: str, policy: Any) -> TargetDetector:
    if kind == "vlm":
        return VLMDetector(policy)
    if kind == "gt":
        return GroundTruthDetector(policy)
    raise ValueError(f"Unknown detector {kind!r}; expected one of {DETECTORS}")
