from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from src.evaluation.xai_common import (
    ExplanationResult,
    make_baseline_tensor,
    normalize_map,
    predict_probabilities,
)

logger = logging.getLogger("perturbation_xai")
_LOGGED_MEDIAPIPE_WARNINGS: set[str] = set()


def create_grid_segments(height: int, width: int, rows: int = 5, cols: int = 5) -> np.ndarray:
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    segments = np.zeros((height, width), dtype=np.int32)
    y_edges = np.linspace(0, height, rows + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, cols + 1, dtype=np.int32)

    seg_id = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            y0, y1 = y_edges[row_idx], y_edges[row_idx + 1]
            x0, x1 = x_edges[col_idx], x_edges[col_idx + 1]
            segments[y0:y1, x0:x1] = seg_id
            seg_id += 1
    return segments


def _normalize_segmentation_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image

    min_val = float(image.min())
    max_val = float(image.max())
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def _reindex_segments(segments: np.ndarray) -> np.ndarray:
    unique_ids = np.unique(segments)
    return np.searchsorted(unique_ids, segments).astype(np.int32)


def create_slic_segments(
    image_tensor: torch.Tensor,
    n_segments: int = 25,
    compactness: float = 10.0,
    sigma: float = 1.0,
    start_label: int = 0,
) -> np.ndarray:
    try:
        from skimage.segmentation import slic
    except ImportError as exc:
        raise ImportError(
            "SLIC segmentation requires scikit-image. Install it with `pip install scikit-image`."
        ) from exc

    image = image_tensor.detach().cpu().float().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")

    if image.shape[0] == 1:
        segmentation_image = image[0]
    else:
        segmentation_image = np.moveaxis(image, 0, -1)

    segmentation_image = _normalize_segmentation_image(segmentation_image)
    channel_axis = None if segmentation_image.ndim == 2 else -1
    segments = slic(
        segmentation_image,
        n_segments=max(2, int(n_segments)),
        compactness=max(float(compactness), 1e-6),
        sigma=max(float(sigma), 0.0),
        start_label=int(start_label),
        channel_axis=channel_axis,
    )
    return _reindex_segments(segments)


def create_template_face_segments(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = (xx + 0.5) / max(width, 1)
    y_norm = (yy + 0.5) / max(height, 1)

    face = (
        ((x_norm - 0.5) / 0.34) ** 2
        + ((y_norm - 0.52) / 0.42) ** 2
    ) <= 1.0

    left_eye = (
        ((x_norm - 0.32) / 0.11) ** 2
        + ((y_norm - 0.33) / 0.06) ** 2
    ) <= 1.0
    right_eye = (
        ((x_norm - 0.68) / 0.11) ** 2
        + ((y_norm - 0.33) / 0.06) ** 2
    ) <= 1.0
    left_eyebrow = (
        (x_norm >= 0.18)
        & (x_norm <= 0.45)
        & (y_norm >= 0.18)
        & (y_norm <= 0.28)
    )
    right_eyebrow = (
        (x_norm >= 0.55)
        & (x_norm <= 0.82)
        & (y_norm >= 0.18)
        & (y_norm <= 0.28)
    )
    nose = (
        ((x_norm - 0.5) / 0.10) ** 2
        + ((y_norm - 0.54) / 0.14) ** 2
    ) <= 1.0
    mouth = (
        ((x_norm - 0.5) / 0.18) ** 2
        + ((y_norm - 0.73) / 0.09) ** 2
    ) <= 1.0
    left_cheek = (
        ((x_norm - 0.28) / 0.14) ** 2
        + ((y_norm - 0.58) / 0.16) ** 2
    ) <= 1.0
    right_cheek = (
        ((x_norm - 0.72) / 0.14) ** 2
        + ((y_norm - 0.58) / 0.16) ** 2
    ) <= 1.0
    forehead = (
        (x_norm >= 0.26)
        & (x_norm <= 0.74)
        & (y_norm >= 0.10)
        & (y_norm <= 0.24)
    )
    chin = (
        (x_norm >= 0.32)
        & (x_norm <= 0.68)
        & (y_norm >= 0.82)
        & (y_norm <= 0.92)
    )

    region_masks = {
        "background": ~face,
        "forehead": forehead & face,
        "left_eyebrow": left_eyebrow & face,
        "right_eyebrow": right_eyebrow & face,
        "left_eye": left_eye & face,
        "right_eye": right_eye & face,
        "nose": nose & face,
        "left_cheek": left_cheek & face,
        "right_cheek": right_cheek & face,
        "mouth": mouth & face,
        "chin": chin & face,
    }
    occupied = np.zeros((height, width), dtype=bool)
    for region_name, mask in region_masks.items():
        if region_name == "background":
            continue
        occupied |= mask
    region_masks["face_skin"] = face & ~occupied

    region_order = [
        "background",
        "forehead",
        "left_eyebrow",
        "right_eyebrow",
        "left_eye",
        "right_eye",
        "nose",
        "left_cheek",
        "right_cheek",
        "mouth",
        "chin",
        "face_skin",
    ]
    return _segments_from_region_masks(region_masks, region_order)


def _tensor_to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    image = np.moveaxis(image, 0, -1)
    image = _normalize_segmentation_image(image)
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _get_mediapipe_face_mesh(
    refine_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "MediaPipe segmentation requires `mediapipe`. Install it with `pip install mediapipe`."
        ) from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        raise RuntimeError("Installed mediapipe package does not expose the legacy `solutions.face_mesh` API.")

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=bool(refine_landmarks),
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )


def _legacy_mediapipe_available() -> bool:
    try:
        _get_mediapipe_face_mesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return True
    except Exception:
        return False


@lru_cache(maxsize=8)
def _get_mediapipe_tasks_face_landmarker(
    model_asset_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
):
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Installed mediapipe package does not provide the Face Landmarker tasks API."
        ) from exc

    model_path = Path(model_asset_path)
    if not model_asset_path:
        raise RuntimeError(
            "This mediapipe build exposes only the `tasks` API. Set "
            "`xai.perturbation_methods.mediapipe_model_asset_path` to a local `face_landmarker` "
            "`.task` model file, or install a mediapipe build that exposes `solutions.face_mesh`."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe face landmarker model not found: {model_path}")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=float(min_detection_confidence),
        min_face_presence_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def _connections_to_indices(connections) -> list[int]:
    indices: set[int] = set()
    for connection in connections:
        if hasattr(connection, "start") and hasattr(connection, "end"):
            indices.add(int(connection.start))
            indices.add(int(connection.end))
        else:
            start, end = connection
            indices.add(int(start))
            indices.add(int(end))
    return sorted(indices)


def _landmark_points_to_pixels(
    face_landmarks,
    indices: list[int],
    width: int,
    height: int,
) -> np.ndarray:
    points = []
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)

    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(round(float(np.clip(landmark.x, 0.0, 1.0)) * max_x))
        y = int(round(float(np.clip(landmark.y, 0.0, 1.0)) * max_y))
        points.append((x, y))

    if not points:
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(np.asarray(points, dtype=np.int32), axis=0)


def _polygon_mask_from_points(
    points: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "MediaPipe segmentation requires `opencv-python`. Install it with `pip install opencv-python`."
        ) from exc

    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 1)
    elif len(points) == 2:
        thickness = max(1, min(height, width) // 64)
        cv2.line(mask, tuple(points[0]), tuple(points[1]), 1, thickness=thickness)
    elif len(points) == 1:
        radius = max(1, min(height, width) // 128)
        cv2.circle(mask, tuple(points[0]), radius=radius, color=1, thickness=-1)
    return mask.astype(bool)


def _build_mediapipe_region_masks(
    image_tensor: torch.Tensor,
    *,
    model_asset_path: str,
    refine_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[dict[str, np.ndarray] | None, str]:
    if _legacy_mediapipe_available():
        return _build_mediapipe_region_masks_legacy(
            image_tensor,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    return _build_mediapipe_region_masks_tasks(
        image_tensor,
        model_asset_path=model_asset_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _build_mediapipe_region_masks_legacy(
    image_tensor: torch.Tensor,
    *,
    refine_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[dict[str, np.ndarray] | None, str]:
    import mediapipe as mp

    rgb_image = _tensor_to_rgb_uint8(image_tensor)
    height, width = rgb_image.shape[:2]
    face_mesh = _get_mediapipe_face_mesh(
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None, "solutions"

    face_landmarks = results.multi_face_landmarks[0]
    face_mesh_solution = mp.solutions.face_mesh
    region_connections = {
        "mouth": face_mesh_solution.FACEMESH_LIPS,
        "left_eye": face_mesh_solution.FACEMESH_LEFT_EYE,
        "right_eye": face_mesh_solution.FACEMESH_RIGHT_EYE,
        "left_eyebrow": face_mesh_solution.FACEMESH_LEFT_EYEBROW,
        "right_eyebrow": face_mesh_solution.FACEMESH_RIGHT_EYEBROW,
        "nose": face_mesh_solution.FACEMESH_NOSE,
        "face": face_mesh_solution.FACEMESH_FACE_OVAL,
    }

    region_masks: dict[str, np.ndarray] = {}
    for region_name, connections in region_connections.items():
        points = _landmark_points_to_pixels(
            face_landmarks,
            _connections_to_indices(connections),
            width,
            height,
        )
        region_masks[region_name] = _polygon_mask_from_points(points, height, width)

    feature_names = [
        "mouth",
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
        "nose",
    ]
    feature_union = np.zeros((height, width), dtype=bool)
    for feature_name in feature_names:
        feature_union |= region_masks[feature_name]

    face_mask = region_masks["face"]
    region_masks["face_skin"] = face_mask & ~feature_union
    region_masks["eyes"] = region_masks["left_eye"] | region_masks["right_eye"]
    region_masks["eyebrows"] = region_masks["left_eyebrow"] | region_masks["right_eyebrow"]
    region_masks["background"] = ~face_mask
    return region_masks, "solutions"


def _build_mediapipe_region_masks_tasks(
    image_tensor: torch.Tensor,
    *,
    model_asset_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[dict[str, np.ndarray] | None, str]:
    import mediapipe as mp
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

    rgb_image = _tensor_to_rgb_uint8(image_tensor)
    height, width = rgb_image.shape[:2]
    landmarker = _get_mediapipe_tasks_face_landmarker(
        model_asset_path=model_asset_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = landmarker.detect(mp_image)
    if not results.face_landmarks:
        return None, "tasks"

    face_landmarks = results.face_landmarks[0]
    region_connections = {
        "mouth": FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
        "left_eye": FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
        "right_eye": FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
        "left_eyebrow": FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
        "right_eyebrow": FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        "nose": FaceLandmarksConnections.FACE_LANDMARKS_NOSE,
        "face": FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
    }

    region_masks: dict[str, np.ndarray] = {}
    for region_name, connections in region_connections.items():
        points = _landmark_points_to_pixels(
            face_landmarks,
            _connections_to_indices(connections),
            width,
            height,
        )
        region_masks[region_name] = _polygon_mask_from_points(points, height, width)

    feature_names = [
        "mouth",
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
        "nose",
    ]
    feature_union = np.zeros((height, width), dtype=bool)
    for feature_name in feature_names:
        feature_union |= region_masks[feature_name]

    face_mask = region_masks["face"]
    region_masks["face_skin"] = face_mask & ~feature_union
    region_masks["eyes"] = region_masks["left_eye"] | region_masks["right_eye"]
    region_masks["eyebrows"] = region_masks["left_eyebrow"] | region_masks["right_eyebrow"]
    region_masks["background"] = ~face_mask
    return region_masks, "tasks"


def _segments_from_region_masks(
    region_masks: dict[str, np.ndarray],
    region_order: list[str],
) -> np.ndarray:
    background_shape = next(iter(region_masks.values())).shape
    segments = np.zeros(background_shape, dtype=np.int32)
    for seg_id, region_name in enumerate(region_order):
        mask = region_masks.get(region_name)
        if mask is None:
            continue
        segments[mask] = seg_id
    return _reindex_segments(segments)


def create_mediapipe_segments(
    image_tensor: torch.Tensor,
    config: "PerturbationExplainerConfig",
) -> tuple[np.ndarray, dict[str, object]]:
    region_order = [
        "background",
        "face_skin",
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
        "nose",
        "mouth",
    ]
    fallback_method = str(config.mediapipe_fallback_method).lower()
    if fallback_method == "mediapipe":
        fallback_method = "grid"
    fallback_reason = None

    try:
        region_masks, backend_name = _build_mediapipe_region_masks(
            image_tensor,
            model_asset_path=config.mediapipe_model_asset_path,
            refine_landmarks=config.mediapipe_refine_landmarks,
            min_detection_confidence=config.mediapipe_min_detection_confidence,
            min_tracking_confidence=config.mediapipe_min_tracking_confidence,
        )
        if region_masks is None:
            fallback_reason = "no_face_detected"
            raise RuntimeError("MediaPipe Face Mesh did not detect a face in the input image.")
        segments = _segments_from_region_masks(region_masks, region_order)
        return segments, {
            "requested_segmentation_method": "mediapipe",
            "segmentation_method": "mediapipe",
            "segmentation_fallback": None,
            "mediapipe_backend": backend_name,
            "mediapipe_region_order": region_order,
        }
    except Exception as exc:
        if fallback_method == "none":
            raise
        warning_message = f"MediaPipe segmentation failed ({exc}); falling back to {fallback_method}."
        if warning_message not in _LOGGED_MEDIAPIPE_WARNINGS:
            logger.warning(warning_message)
            _LOGGED_MEDIAPIPE_WARNINGS.add(warning_message)
        segments, fallback_info = _create_segments(
            image_tensor,
            config,
            requested_method=fallback_method,
        )
        fallback_info = dict(fallback_info)
        fallback_info.update(
            {
                "requested_segmentation_method": "mediapipe",
                "segmentation_fallback": fallback_reason or exc.__class__.__name__,
                "mediapipe_region_order": region_order,
            }
        )
        return segments, fallback_info


def create_segments(
    image_tensor: torch.Tensor,
    config: "PerturbationExplainerConfig",
) -> tuple[np.ndarray, dict[str, object]]:
    return _create_segments(image_tensor, config, requested_method=str(config.segmentation_method).lower())


def _create_segments(
    image_tensor: torch.Tensor,
    config: "PerturbationExplainerConfig",
    *,
    requested_method: str,
) -> tuple[np.ndarray, dict[str, object]]:
    _, height, width = image_tensor.shape
    segmentation_method = str(requested_method).lower()

    if segmentation_method == "grid":
        return (
            create_grid_segments(height, width, rows=config.rows, cols=config.cols),
            {
                "requested_segmentation_method": segmentation_method,
                "segmentation_method": "grid",
                "segmentation_fallback": None,
            },
        )
    if segmentation_method == "template":
        return (
            create_template_face_segments(height, width),
            {
                "requested_segmentation_method": segmentation_method,
                "segmentation_method": "template",
                "segmentation_fallback": None,
            },
        )
    if segmentation_method == "slic":
        return (
            create_slic_segments(
                image_tensor,
                n_segments=config.slic_n_segments,
                compactness=config.slic_compactness,
                sigma=config.slic_sigma,
            ),
            {
                "requested_segmentation_method": segmentation_method,
                "segmentation_method": "slic",
                "segmentation_fallback": None,
            },
        )
    if segmentation_method == "mediapipe":
        return create_mediapipe_segments(image_tensor, config)
    raise ValueError(f"Unsupported segmentation method: {requested_method}")


def coalition_to_tensor_mask(
    coalition: np.ndarray,
    segments: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    mask = coalition[segments]
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def apply_coalition(
    image_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    coalition: np.ndarray,
    segments: np.ndarray,
) -> torch.Tensor:
    mask = coalition_to_tensor_mask(coalition, segments, image_tensor.device).unsqueeze(0)
    return image_tensor * mask + baseline_tensor * (1.0 - mask)


def sample_random_coalitions(
    num_segments: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coalitions = rng.integers(0, 2, size=(num_samples, num_segments), endpoint=False).astype(np.float32)
    anchors = [
        np.zeros((1, num_segments), dtype=np.float32),
        np.ones((1, num_segments), dtype=np.float32),
    ]
    return np.concatenate(anchors + [coalitions], axis=0)


def evaluate_coalitions(
    model: nn.Module,
    image_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    segments: np.ndarray,
    coalitions: np.ndarray,
    target_class: int,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    scores = []
    batch_tensors = []

    for coalition in coalitions:
        perturbed = apply_coalition(image_tensor, baseline_tensor, coalition, segments)
        batch_tensors.append(perturbed)
        if len(batch_tensors) >= batch_size:
            batch = torch.stack(batch_tensors, dim=0)
            probs = predict_probabilities(model, batch, device).numpy()
            scores.extend(probs[:, target_class].tolist())
            batch_tensors = []

    if batch_tensors:
        batch = torch.stack(batch_tensors, dim=0)
        probs = predict_probabilities(model, batch, device).numpy()
        scores.extend(probs[:, target_class].tolist())

    return np.asarray(scores, dtype=np.float32)


def lime_kernel_weights(coalitions: np.ndarray, kernel_width: float) -> np.ndarray:
    distances = np.sqrt(((1.0 - coalitions) ** 2).sum(axis=1))
    return np.exp(-(distances ** 2) / max(kernel_width ** 2, 1e-8))


def kernel_shap_weights(coalitions: np.ndarray) -> np.ndarray:
    num_segments = coalitions.shape[1]
    weights = np.zeros(coalitions.shape[0], dtype=np.float64)

    for idx, coalition in enumerate(coalitions):
        size = int(coalition.sum())
        if size == 0 or size == num_segments:
            weights[idx] = 1_000.0
            continue
        denom = math.comb(num_segments, size) * size * (num_segments - size)
        weights[idx] = (num_segments - 1) / max(float(denom), 1e-8)
    return weights.astype(np.float32)


def segment_scores_to_heatmap(
    segment_scores: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    return segment_scores[segments]


@dataclass
class PerturbationExplainerConfig:
    segmentation_method: str = "grid"
    rows: int = 5
    cols: int = 5
    slic_n_segments: int = 25
    slic_compactness: float = 10.0
    slic_sigma: float = 1.0
    mediapipe_model_asset_path: str = ""
    mediapipe_refine_landmarks: bool = True
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    mediapipe_fallback_method: str = "grid"
    num_samples: int = 200
    batch_size: int = 32
    baseline_strategy: str = "blur"
    blur_kernel_size: int = 11
    ridge_alpha: float = 1.0
    kernel_width: float = 5.0
    random_seed: int = 42


class _BasePerturbationExplainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: PerturbationExplainerConfig,
    ):
        self.model = model
        self.device = device
        self.config = config

    def _fit_segment_scores(
        self,
        coalitions: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> ExplanationResult:
        image_tensor = image_tensor.to(self.device)
        probs = predict_probabilities(self.model, image_tensor, self.device).squeeze(0).numpy()
        predicted_class = int(probs.argmax())
        if target_class is None:
            target_class = predicted_class
        confidence = float(probs[target_class])

        segments, segmentation_info = create_segments(image_tensor, self.config)
        baseline = make_baseline_tensor(
            image_tensor,
            strategy=self.config.baseline_strategy,
            blur_kernel_size=self.config.blur_kernel_size,
        )

        rng = np.random.default_rng(self.config.random_seed)
        coalitions = sample_random_coalitions(
            num_segments=int(segments.max()) + 1,
            num_samples=self.config.num_samples,
            rng=rng,
        )
        scores = evaluate_coalitions(
            self.model,
            image_tensor,
            baseline,
            segments,
            coalitions,
            target_class,
            self.device,
            batch_size=self.config.batch_size,
        )
        segment_scores = self._fit_segment_scores(coalitions, scores)
        raw_heatmap = segment_scores_to_heatmap(segment_scores, segments)
        positive_heatmap = np.maximum(raw_heatmap, 0.0)
        if np.allclose(positive_heatmap.max(), 0.0):
            positive_heatmap = np.abs(raw_heatmap)
        heatmap = normalize_map(positive_heatmap)

        return ExplanationResult(
            method=self.name,
            heatmap=heatmap,
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            metadata=self._build_metadata(segments, segment_scores, segmentation_info),
        )

    def _build_metadata(
        self,
        segments: np.ndarray,
        segment_scores: np.ndarray,
        segmentation_info: dict[str, object],
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "segment_scores": segment_scores.tolist(),
            "num_segments": int(segments.max()) + 1,
            "num_samples": self.config.num_samples,
        }
        metadata.update(segmentation_info)

        used_method = str(metadata.get("segmentation_method", self.config.segmentation_method)).lower()
        if used_method == "grid":
            metadata.update(
                {
                    "rows": self.config.rows,
                    "cols": self.config.cols,
                }
            )
        elif used_method == "template":
            metadata.update(
                {
                    "template_family": "aligned_face_regions_v1",
                }
            )
        elif used_method == "slic":
            metadata.update(
                {
                    "slic_n_segments": self.config.slic_n_segments,
                    "slic_compactness": self.config.slic_compactness,
                    "slic_sigma": self.config.slic_sigma,
                }
            )
        elif used_method == "mediapipe":
            metadata.update(
                {
                    "mediapipe_model_asset_path": self.config.mediapipe_model_asset_path,
                    "mediapipe_refine_landmarks": self.config.mediapipe_refine_landmarks,
                    "mediapipe_min_detection_confidence": self.config.mediapipe_min_detection_confidence,
                    "mediapipe_min_tracking_confidence": self.config.mediapipe_min_tracking_confidence,
                    "mediapipe_fallback_method": self.config.mediapipe_fallback_method,
                }
            )
        return metadata


class LIMEExplainer(_BasePerturbationExplainer):
    name = "lime"

    def _fit_segment_scores(
        self,
        coalitions: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        weights = lime_kernel_weights(coalitions, kernel_width=self.config.kernel_width)
        regressor = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        regressor.fit(coalitions, scores, sample_weight=weights)
        return regressor.coef_.astype(np.float32)


class KernelSHAPExplainer(_BasePerturbationExplainer):
    name = "shap"

    def _fit_segment_scores(
        self,
        coalitions: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        weights = kernel_shap_weights(coalitions)
        regressor = Ridge(alpha=max(self.config.ridge_alpha, 1e-4), fit_intercept=True)
        regressor.fit(coalitions, scores, sample_weight=weights)
        return regressor.coef_.astype(np.float32)
