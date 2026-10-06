from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest
import torch

from vlfm.detection.interface import GroundTruthDetector, VLMDetector, make_detector
from vlfm.vlm.detections import ObjectDetections


def fake_policy(boxes: List[List[float]]) -> Any:
    """Detector and segmenter stand-ins, recording the boxes passed to the segmenter."""
    detections = ObjectDetections(
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.ones(len(boxes)),
        ["chair"] * len(boxes),
        image_source=None,
        fmt="xyxy",
    )
    segmented: List[List[float]] = []

    def segment_bbox(rgb: np.ndarray, box: List[float]) -> np.ndarray:
        segmented.append(box)
        return np.ones(rgb.shape[:2], dtype=np.uint8)

    return SimpleNamespace(
        _get_object_detections=lambda rgb: detections,
        _mobile_sam=SimpleNamespace(segment_bbox=segment_bbox),
        segmented=segmented,
    )


def test_vlm_detector_segments_each_box_in_pixels() -> None:
    policy = fake_policy([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, masks = make_detector("vlm", policy).detect(rgb)
    assert len(masks) == 2 and masks[0].shape == (480, 640)
    # Boxes are normalized; the segmenter gets pixels, as in BaseObjectNavPolicy
    assert policy.segmented[0] == pytest.approx([0.0, 0.0, 320.0, 240.0])
    assert policy.segmented[1] == pytest.approx([320.0, 240.0, 640.0, 480.0])


def test_vlm_detector_without_detections() -> None:
    detections, masks = VLMDetector(fake_policy([])).detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert masks == [] and len(detections.phrases) == 0


def test_ground_truth_detector_is_reserved() -> None:
    with pytest.raises(NotImplementedError):
        make_detector("gt", None)
    with pytest.raises(NotImplementedError):
        GroundTruthDetector()


def test_unknown_detector() -> None:
    with pytest.raises(ValueError):
        make_detector("lidar", None)
