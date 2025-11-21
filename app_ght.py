"""
🔩 Screw Finder — Simple Template Matching (Fixed Mode)
Finds a reference image (ScrewSquare.png) inside a scene image (MasterImage.png)
using multi-scale template matching. All settings are hardcoded for fast testing.
"""

import io
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# ------------------------------
# Configuration System
# ------------------------------
APP_DIR = Path(__file__).parent.resolve()
SCENE_PATH = APP_DIR / "MasterImage.png"
REF_DIR = APP_DIR / "refs"

def generate_default_config(ref_path: Path, img_array: np.ndarray) -> Dict[str, Any]:
    """
    Generate default configuration based on template properties.
    Auto-detects horizontal/vertical from filename or dimensions.
    """
    h, w = img_array.shape[:2]
    name = ref_path.stem
    is_horiz = "horiz" in name.lower() or (w >= h)
    
    if is_horiz:
        # Horizontal template (wide profile)
        return {
            "version": "1.0",
            "template_name": name,
            "description": f"Horizontal template {w}×{h}px (auto-generated)",
            "detection": {
                "similarity_threshold": 0.70,
                "scale_mode": "range",
                "scale_min": 0.80,
                "scale_max": 1.20,
                "target_width_min": 100.0,
                "target_width_max": 400.0,
                "scale_steps": 21,
                "rotations": [0],
                "max_peaks_per_pass": 10,
                "global_max_detections": 150
            },
            "verification": {
                "density_max": 0.03,
                "template_edge_dt_max": 0.50,
                "aspect_ratio_tolerance_log": 0.08,
                "min_scale": None,
                "vertical_tolerance_deg": None,
                "border_margin_factor": 0.03,
                "pad_ratio": 0.12
            },
            "nms": {
                "iou_threshold": 0.85,
                "cluster_iou": 0.85,
                "top_k_after_nms": 15,
                "suppress_w_factor": 1.0,
                "suppress_h_factor": 1.0
            },
            "preprocessing": {
                "canny_low": 30,
                "canny_high": 120,
                "gaussian_blur": 0,
                "clahe_enabled": False,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": [8, 8]
            },
            "advanced": {
                "profile": "horizontal",
                "is_horizontal": True,
                "scene_max_width": 0,
                "distance_transform_type": "L2",
                "interpolation_upscale": "CUBIC",
                "interpolation_downscale": "AREA"
            }
        }
    else:
        # Vertical template (tall profile)
        return {
            "version": "1.0",
            "template_name": name,
            "description": f"Vertical template {w}×{h}px (auto-generated)",
            "detection": {
                "similarity_threshold": 0.52,
                "scale_mode": "target_width",
                "scale_min": 0.80,
                "scale_max": 1.20,
                "target_width_min": 38.0,
                "target_width_max": 72.0,
                "scale_steps": 21,
                "rotations": [-5, 0, 5],
                "max_peaks_per_pass": 12,
                "global_max_detections": 150
            },
            "verification": {
                "density_max": 1.0,
                "template_edge_dt_max": 0.75,
                "aspect_ratio_tolerance_log": 0.10,
                "min_scale": 0.90,
                "vertical_tolerance_deg": 7.0,
                "border_margin_factor": 0.03,
                "pad_ratio": 0.12
            },
            "nms": {
                "iou_threshold": 0.97,
                "cluster_iou": 0.85,
                "top_k_after_nms": 15,
                "suppress_w_factor": 0.6,
                "suppress_h_factor": 0.5
            },
            "preprocessing": {
                "canny_low": 30,
                "canny_high": 120,
                "gaussian_blur": 0,
                "clahe_enabled": False,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": [8, 8]
            },
            "advanced": {
                "profile": "tower",
                "is_horizontal": False,
                "scene_max_width": 0,
                "distance_transform_type": "L2",
                "interpolation_upscale": "CUBIC",
                "interpolation_downscale": "AREA"
            }
        }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that config has all required fields."""
    required_sections = ["detection", "verification", "nms", "preprocessing", "advanced"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate key parameters
    det = config["detection"]
    if "similarity_threshold" not in det or not (0 <= det["similarity_threshold"] <= 1):
        raise ValueError("Invalid similarity_threshold (must be 0-1)")
    if "scale_steps" not in det or det["scale_steps"] < 1:
        raise ValueError("Invalid scale_steps (must be >= 1)")
    
    return True

def load_or_create_config(ref_path: Path, img_array: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Load config from JSON file matching template name.
    If not found, generate and save default config.
    Returns None if config is invalid (with warning).
    """
    config_path = ref_path.with_suffix('.json')
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            validate_config(config)
            return config
        except json.JSONDecodeError as e:
            st.warning(f"⚠️ Invalid JSON in {config_path.name}: {e}. Skipping template.")
            return None
        except ValueError as e:
            st.warning(f"⚠️ Config validation failed for {config_path.name}: {e}. Skipping template.")
            return None
        except Exception as e:
            st.warning(f"⚠️ Error loading {config_path.name}: {e}. Skipping template.")
            return None
    else:
        # Generate and save default config
        config = generate_default_config(ref_path, img_array)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            st.info(f"✨ Created default config: {config_path.name}")
        except Exception as e:
            st.warning(f"⚠️ Could not save default config {config_path.name}: {e}")
        return config


# ------------------------------
# Legacy Constants (will be removed after full config system)
# ------------------------------
# Choose template by profile
DEFAULT_TEMPLATE_PATH = APP_DIR / "ScrewSquare.png"
TOWER_TEMPLATE_PATH = APP_DIR / "Tower.png"
TEMPLATE_PATH = TOWER_TEMPLATE_PATH if "tower" in str("tower" if 'PROFILE' not in globals() else PROFILE).lower() else DEFAULT_TEMPLATE_PATH
TEMPLATE_NAME = TEMPLATE_PATH.name

# Profile: "default" for symbols, "tower" for tall slender objects
PROFILE = "tower"

# Reference folder (for multiple template refs)
REF_DIR = APP_DIR / "refs"
try:
    REF_DIR.mkdir(exist_ok=True)
except Exception:
    pass

# Peak suppression factors (fraction of template size) for local maxima picking
SUPPRESS_W_FACTOR = 1.0
SUPPRESS_H_FACTOR = 1.0
if PROFILE == "tower":
    # Allow nearby neighbors horizontally/vertically (towers close together)
    SUPPRESS_W_FACTOR = 0.5
    SUPPRESS_H_FACTOR = 0.5
    # Slight tilt allowance
    ROTATIONS = [-5, 0, 5]
    # Gather more peaks when searching for multiple screws
    MAX_PEAKS_PER_PASS = 12

# Preprocessing
GAUSSIAN_BLUR = 3  # 0 disables; must be odd if >0
CONVERT_TO_GRAYSCALE = True

# Speed control
SCENE_MAX_WIDTH = 0  # 0 = keep original size (no downscaling)

# Matching (masked NCC on grayscale using the PNG alpha as mask)
SCALES = np.linspace(0.03, 0.80, 32).tolist()  # widened scale band
ROTATIONS = [0]  # upright only
TM_METHOD = cv2.TM_CCORR_NORMED  # kept for fallback if needed
THRESHOLD = 0.65  # relaxed similarity threshold to ensure peaks are collected
MAX_PEAKS_PER_PASS = 8   # per-scale peaks (allow more instances per scale)
GLOBAL_MAX_DETECTIONS = 100  # cap total raw candidates before NMS
TOP_K_AFTER_NMS = 15  # shortlist before final geometric check

# Non-maximum suppression (NMS)
NMS_IOU_THRESHOLD = 0.6  # IoU threshold to merge boxes (default)
NMS_IOU_THRESHOLD_TOWER = 0.97  # tower profile: allow near-duplicates (even IoU≈0.93); cluster will choose best


# ------------------------------
# Utils
# ------------------------------
def draw_detections(base_img: np.ndarray, detections: List[dict], color: tuple[int,int,int], name: str) -> np.ndarray:
    out = base_img.copy()
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} #{i}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out

def rotate_affine_bgr_mask(bgr: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate template and mask by angle within the same canvas size."""
    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    rot_m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    img_r = cv2.warpAffine(bgr, rot_m, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    mask_r = cv2.warpAffine(mask, rot_m, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, mask_r

def resize_with_aspect_ratio(img: np.ndarray, target_width: int) -> Tuple[np.ndarray, float]:
    if target_width <= 0 or img.shape[1] <= target_width:
        return img, 1.0
    scale = target_width / float(img.shape[1])
    new_h = int(round(img.shape[0] * scale))
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_preprocessing(img: np.ndarray) -> np.ndarray:
    proc = img.copy()
    if CONVERT_TO_GRAYSCALE:
        proc = to_gray(proc)
    if GAUSSIAN_BLUR and GAUSSIAN_BLUR > 0 and GAUSSIAN_BLUR % 2 == 1:
        proc = cv2.GaussianBlur(proc, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)
    return proc


def build_template_from_png(template_rgba_or_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    From an input template (possibly RGBA), build a cropped BGR template and a binary mask (uint8 0/255)
    derived from the alpha channel if present; otherwise from near-white suppression.
    Returns (template_bgr_cropped, mask_cropped).
    """
    img = template_rgba_or_bgr
    if img is None:
        raise ValueError("Empty template")
    # Extract alpha if present
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        mask = (a > 200).astype(np.uint8) * 255
        templ_bgr = cv2.merge([b, g, r])
    else:
        templ_bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = to_gray(templ_bgr)
        # Treat near-white as background
        mask = (gray < 245).astype(np.uint8) * 255
    # Clean & crop
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if cv2.countNonZero(mask) == 0:
        # Fallback to full mask
        mask = np.ones(templ_bgr.shape[:2], dtype=np.uint8) * 255
    ys, xs = np.where(mask > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    templ_bgr_c = templ_bgr[y1:y2, x1:x2].copy()
    mask_c = mask[y1:y2, x1:x2].copy()
    # Keep mask as-is (avoid erosion to preserve thin edges)
    return templ_bgr_c, mask_c


def nms_iou(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    boxes: (N, 4) as [x1, y1, x2, y2]
    scores: (N,)
    returns: indices kept
    """
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def collect_peaks(result: np.ndarray, threshold: float, tmpl_w: int, tmpl_h: int, max_peaks: int, suppress_w: float = 1.0, suppress_h: float = 1.0) -> List[Tuple[int, int, float]]:
    """Greedy peak picking to avoid flood of points: repeatedly take global max and suppress its neighborhood."""
    peaks: List[Tuple[int, int, float]] = []
    res = result.copy()
    for _ in range(max_peaks):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < threshold:
            break
        peaks.append((int(max_loc[0]), int(max_loc[1]), float(max_val)))
        # Suppress a window smaller than the template to allow neighboring instances
        sw = max(1, int(round(tmpl_w * suppress_w)))
        sh = max(1, int(round(tmpl_h * suppress_h)))
        x0 = max(0, max_loc[0] - sw // 2)
        y0 = max(0, max_loc[1] - sh // 2)
        x1 = min(res.shape[1], max_loc[0] + sw // 2)
        y1 = min(res.shape[0], max_loc[1] + sh // 2)
        res[y0:y1, x0:x1] = 0.0
    return peaks


def match_multiscale(scene_img: np.ndarray, template_img: np.ndarray, template_mask: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, List[dict], dict]:
    """
    Distance-transform based edge matching with mask awareness.
    Returns: (annotated_full_res_image, detections) where detection is
    {x1,y1,x2,y2,score,scale,angle}
    config: Per-template configuration dictionary with all detection/verification parameters
    """
    # Extract config parameters
    det_cfg = config["detection"]
    ver_cfg = config["verification"]
    nms_cfg = config["nms"]
    prep_cfg = config["preprocessing"]
    adv_cfg = config["advanced"]
    
    # Detection parameters
    similarity_threshold = float(det_cfg["similarity_threshold"])
    scale_mode = det_cfg.get("scale_mode", "range")
    rotations = det_cfg["rotations"]
    max_peaks_per_pass = int(det_cfg["max_peaks_per_pass"])
    global_max_detections = int(det_cfg["global_max_detections"])
    
    # Compute scales based on mode
    if scale_mode == "target_width":
        tpl_w = max(1, template_img.shape[1])
        tgt_min_w = float(det_cfg["target_width_min"])
        tgt_max_w = float(det_cfg["target_width_max"])
        lower_scale = max(0.03, tgt_min_w / float(tpl_w))
        upper_scale = min(1.50, tgt_max_w / float(tpl_w))
        scales = np.linspace(lower_scale, upper_scale, int(det_cfg["scale_steps"])).tolist()
    else:  # "range"
        scales = np.linspace(float(det_cfg["scale_min"]), float(det_cfg["scale_max"]), int(det_cfg["scale_steps"])).tolist()
    
    # Verification parameters
    density_max = float(ver_cfg["density_max"])
    template_edge_dt_max = float(ver_cfg["template_edge_dt_max"])
    ar_tol_log = float(ver_cfg["aspect_ratio_tolerance_log"])
    min_scale_tower = ver_cfg.get("min_scale")
    if min_scale_tower is not None:
        min_scale_tower = float(min_scale_tower)
    vertical_tol_deg = ver_cfg.get("vertical_tolerance_deg")
    if vertical_tol_deg is not None:
        vertical_tol_deg = float(vertical_tol_deg)
    border_margin_factor = float(ver_cfg.get("border_margin_factor", 0.03))
    pad_ratio = float(ver_cfg.get("pad_ratio", 0.12))
    
    # NMS parameters
    nms_iou_threshold = float(nms_cfg["iou_threshold"])
    cluster_iou = float(nms_cfg["cluster_iou"])
    top_k_after_nms = int(nms_cfg["top_k_after_nms"])
    suppress_w_factor = float(nms_cfg["suppress_w_factor"])
    suppress_h_factor = float(nms_cfg["suppress_h_factor"])
    
    # Preprocessing parameters
    canny_low = int(prep_cfg["canny_low"])
    canny_high = int(prep_cfg["canny_high"])
    
    # Advanced parameters
    is_horizontal = bool(adv_cfg["is_horizontal"])
    profile = adv_cfg.get("profile", "default")
    scene_max_width = int(adv_cfg.get("scene_max_width", 0))
    # Previews are displayed separately; do processing copies here
    scene_full = scene_img.copy()
    # Resize scene and compute grayscale + edges + distance transform
    scene_resized_color, s_scale = resize_with_aspect_ratio(scene_full, scene_max_width)
    # Save the downscaled scene so it can be reviewed
    try:
        down_path = APP_DIR / "MasterImage.downscaled.png"
        cv2.imwrite(str(down_path), scene_resized_color)
    except Exception:
        pass
    scene_gray = to_gray(scene_resized_color)  # No global preprocessing now
    scene_edges = cv2.Canny(scene_gray, canny_low, canny_high)
    # Distance to nearest edge: invert so edges=0
    scene_dt = cv2.distanceTransform(255 - scene_edges, cv2.DIST_L2, 3).astype(np.float32)

    debug_info: dict = {}
    debug_info["scene_shape"] = tuple(scene_full.shape[:2])
    debug_info["scene_downscaled_shape"] = tuple(scene_resized_color.shape[:2])
    debug_info["scale_factor"] = float(s_scale)
    debug_info["rotations"] = rotations[:]
    debug_info["scales"] = scales[:]
    debug_info["template_shape"] = tuple(template_img.shape[:2])
    debug_info["template_mask_nonzero"] = int(cv2.countNonZero(template_mask))
    debug_info["passes"] = []
    debug_info["profile"] = profile
    debug_info["suppress_factors"] = {"w": suppress_w_factor, "h": suppress_h_factor}

    candidates: List[dict] = []
    # Track the single best response across all passes to use as a fallback
    best_fallback = None  # (score, x, y, tw, th, scale, angle)
    # Define a border margin to avoid spurious border alignments
    H_full, W_full = scene_gray.shape[:2]
    border_margin = max(6, int(round(border_margin_factor * min(H_full, W_full))))
    for angle in rotations:
        # Rotate template if needed
        if angle != 0:
            (h, w) = template_img.shape[:2]
            # For 90° rotations, swap canvas dimensions
            abs_angle = abs(angle) % 360
            if abs_angle == 90 or abs_angle == 270:
                # Swap dimensions for 90° rotation
                new_w, new_h = h, w
                # For 90° rotation, center should be at (h/2, w/2) in original coordinates
                # After rotation, the center in new canvas is at (new_w/2, new_h/2)
                center = (w / 2.0, h / 2.0)
                rot_m = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Adjust translation for new canvas size: shift to center of new canvas
                rot_m[0, 2] += (new_w - w) / 2.0
                rot_m[1, 2] += (new_h - h) / 2.0
            else:
                # Keep original dimensions for small rotations
                new_w, new_h = w, h
                center = (w / 2.0, h / 2.0)
                rot_m = cv2.getRotationMatrix2D(center, angle, 1.0)
            tpl_rot = cv2.warpAffine(template_img, rot_m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            mask_rot = cv2.warpAffine(template_mask, rot_m, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            tpl_rot = template_img
            mask_rot = template_mask

        # Edge-based matching across scales
        for scale in scales:
            tw = max(8, int(round(tpl_rot.shape[1] * scale)))
            th = max(8, int(round(tpl_rot.shape[0] * scale)))
            if tw >= scene_gray.shape[1] or th >= scene_gray.shape[0]:
                continue
            tpl_s = cv2.resize(tpl_rot, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
            mask_s = cv2.resize(mask_rot, (tw, th), interpolation=cv2.INTER_NEAREST)
            if cv2.countNonZero(mask_s) < 10:
                continue
            # Build template edge kernel (0/1) only inside mask
            tpl_gray = to_gray(tpl_s)
            tpl_edges = cv2.Canny(tpl_gray, canny_low, canny_high)
            tpl_edges = cv2.bitwise_and(tpl_edges, tpl_edges, mask=mask_s)
            kernel = (tpl_edges > 0).astype(np.float32)
            edge_count = float(kernel.sum())
            if edge_count < 10:
                continue
            # Sum DT under kernel with filter2D
            sum_dt = cv2.filter2D(scene_dt, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            mean_dt = sum_dt / (edge_count + 1e-6)
            # Convert to similarity in [0,1]
            sim = 1.0 / (1.0 + mean_dt)
            # Track global best for fallback (clamp to valid placement region)
            _, max_val, _, max_loc = cv2.minMaxLoc(sim)
            # convert center -> top-left and clamp
            max_px = max(0, min(int(max_loc[0]) - tw // 2, scene_gray.shape[1] - tw))
            max_py = max(0, min(int(max_loc[1]) - th // 2, scene_gray.shape[0] - th))
            if best_fallback is None or float(max_val) > best_fallback[0]:
                best_fallback = (float(max_val), max_px, max_py, tw, th, float(scale), float(angle))
            # Record per-pass summary
            debug_info["passes"].append({
                "angle": float(angle),
                "scale": float(scale),
                "tw": int(tw),
                "th": int(th),
                "max_similarity": float(max_val),
                "edge_count": int(edge_count)
            })
            # Use similarity threshold from config
            peaks = collect_peaks(sim, similarity_threshold, tw, th, max_peaks_per_pass, suppress_w_factor, suppress_h_factor)
            for (px, py, score) in peaks:
                # Convert center -> top-left and clamp
                x1_r = max(0, min(px - tw // 2, scene_gray.shape[1] - tw))
                y1_r = max(0, min(py - th // 2, scene_gray.shape[0] - th))
                # Ensure bottom-right is derived from corrected top-left
                x2_r = x1_r + tw
                y2_r = y1_r + th
                # Map back to full resolution
                if s_scale != 1.0:
                    x1 = int(round(x1_r / s_scale))
                    y1 = int(round(y1_r / s_scale))
                    x2 = int(round(x2_r / s_scale))
                    y2 = int(round(y2_r / s_scale))
                else:
                    x1, y1, x2, y2 = x1_r, y1_r, x2_r, y2_r
                # Border margin gate: drop boxes touching the image border
                if x1 < border_margin or y1 < border_margin or x2 > scene_gray.shape[1] - border_margin or y2 > scene_gray.shape[0] - border_margin:
                    continue
                candidates.append(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score, "scale": scale, "angle": float(angle)}
                )
                if len(candidates) >= global_max_detections:
                    break
            if len(candidates) >= global_max_detections:
                break
        if len(candidates) >= global_max_detections:
            break

    debug_info["candidates_before_nms"] = int(len(candidates))
    # Keep a copy of raw candidates before NMS for debugging
    debug_info["raw_candidates"] = [
        {
            "box": [int(c["x1"]), int(c["y1"]), int(c["x2"]), int(c["y2"])],
            "score": float(c.get("score", 0.0)),
            "scale": float(c.get("scale", 0.0)),
            "angle": float(c.get("angle", 0.0)),
        }
        for c in candidates
    ]
    if not candidates and best_fallback is not None:
        # Add one fallback candidate (top response overall), mapped to full resolution
        score, px, py, tw, th, s_used, ang_used = best_fallback
        if s_scale != 1.0:
            x1 = int(round(px / s_scale))
            y1 = int(round(py / s_scale))
            x2 = int(round((px + tw) / s_scale))
            y2 = int(round((py + th) / s_scale))
        else:
            x1, y1, x2, y2 = px, py, px + tw, py + th
        candidates.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score, "scale": s_used, "angle": ang_used})
        debug_info["fallback"] = {
            "similarity": float(score),
            "scale": float(s_used),
            "angle": float(ang_used),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        }

    if not candidates:
        # Provide previews for debugging
        try:
            dt_disp = cv2.normalize(scene_dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            debug_info["preview_edges"] = scene_edges
            debug_info["preview_dt"] = dt_disp
        except Exception:
            pass
        return scene_full, [], debug_info

    # NMS
    boxes = np.array([[c["x1"], c["y1"], c["x2"], c["y2"]] for c in candidates], dtype=np.int32)
    scores = np.array([c["score"] for c in candidates], dtype=np.float32)
    keep = nms_iou(boxes, scores, nms_iou_threshold)
    kept_all = [candidates[i] for i in keep]
    # Cluster duplicates (high IoU) and keep a single best per location
    def _iou(b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        ix1, iy1 = max(x11, x21), max(y11, y21)
        ix2, iy2 = min(x12, x22), min(y12, y22)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        a1 = (x12 - x11 + 1) * (y12 - y11 + 1)
        a2 = (x22 - x21 + 1) * (y22 - y21 + 1)
        denom = float(a1 + a2 - inter + 1e-6)
        return float(inter) / denom if denom > 0 else 0.0
    kept_sorted = sorted(kept_all, key=lambda d: d["score"], reverse=True)
    distinct = []
    for d in kept_sorted:
        b = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        if all(_iou(b, (int(e["x1"]), int(e["y1"]), int(e["x2"]), int(e["y2"]))) < cluster_iou for e in distinct):
            distinct.append(d)
    # Final shortlist
    distinct_sorted = sorted(distinct, key=lambda d: d["score"], reverse=True)
    kept = distinct_sorted[:top_k_after_nms]
    debug_info["nms_iou_threshold_used"] = float(nms_iou_threshold)
    debug_info["topk_cutoff_score"] = float(kept[-1]["score"]) if kept else None
    debug_info["candidates_after_nms"] = int(len(kept))
    debug_info["nms_candidates"] = [
        {
            "box": [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])],
            "score": float(d.get("score", 0.0)),
            "scale": float(d.get("scale", 0.0)),
            "angle": float(d.get("angle", 0.0)),
        }
        for d in kept
    ]

    # Final geometric filter (simplified, no polygon requirement):
    # 1) Aspect ratio near the template's aspect
    # 2) Low interior edge density (triangle interior should be mostly blank)
    # 3) Boundary alignment via DT (edges near scene edges)
    # Note: aspect ratio should be width/height of the DETECTION BOX
    # For detection boxes, we compare against the template's aspect ratio
    # When template is rotated 90°/270°, dimensions are swapped
    tpl_ar = max(1e-6, template_img.shape[1] / float(template_img.shape[0]))
    # Recompute edges/DT in case we are in a new scope
    scene_gray_full = scene_gray
    scene_edges_full = scene_edges
    scene_dt_full = scene_dt
    H, W = scene_edges_full.shape
    verification_details = []
    accepted_details = []
    passed = []
    # Verification thresholds from config (already extracted above)
    boundary_dt_max = 3.0  # Fixed for now
    border_margin_verify = max(6, int(round(border_margin_factor * min(H, W))))
    debug_info["verification_thresholds"] = {
        "similarity_threshold": float(similarity_threshold),
        "density_max": float(density_max),
        "boundary_dt_max": float(boundary_dt_max),
        "template_edge_dt_max": float(template_edge_dt_max),
        "pad_ratio": float(pad_ratio),
        "aspect_ratio_tol_log": float(ar_tol_log),
        "vertical_tol_deg": float(vertical_tol_deg) if vertical_tol_deg is not None else None,
        "min_scale_tower": float(min_scale_tower) if min_scale_tower is not None else None,
        "border_margin": int(border_margin_verify),
    }
    for det in kept:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ar = w / float(h)
        det_angle = float(det.get("angle", 0.0))
        # Adjust expected template aspect ratio based on rotation
        abs_angle = abs(det_angle) % 360
        if abs_angle == 90 or abs_angle == 270:
            # For 90°/270° rotations, swap template dimensions
            tpl_ar_check = max(1e-6, template_img.shape[0] / float(template_img.shape[1]))
        else:
            # For 0° or small rotations, use original aspect ratio
            tpl_ar_check = tpl_ar
        det_detail = {
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "size": f"{w}×{h}",
            "score": float(det.get("score", 0.0)),
            "scale": float(det.get("scale", 0.0)),
            "angle": det_angle,
        }
        # Border margin gate during verification as well
        if x1 < border_margin_verify or y1 < border_margin_verify or x2 > W - border_margin_verify or y2 > H - border_margin_verify:
            det_detail["reject"] = "border_margin"
            verification_details.append(det_detail)
            continue
        if abs(np.log((ar + 1e-6) / tpl_ar_check)) > ar_tol_log:  # tighter ≈ ±10% tolerance
            det_detail["reject"] = f"aspect_ratio ar={ar:.3f} tpl={tpl_ar_check:.3f} log_diff={abs(np.log((ar + 1e-6) / tpl_ar_check)):.3f}>{ar_tol_log:.2f}"
            verification_details.append(det_detail)
            continue
        # enlarge interior pad to ignore boundary bleed
        pad = max(2, int(round(pad_ratio * min(w, h))))
        xi0 = max(0, x1 + pad)
        yi0 = max(0, y1 + pad)
        xi1 = min(W, x2 - pad)
        yi1 = min(H, y2 - pad)
        if xi1 <= xi0 or yi1 <= yi0:
            det_detail["reject"] = "invalid_interior"
            verification_details.append(det_detail)
            continue
        interior = scene_edges_full[yi0:yi1, xi0:xi1]
        density = float(np.count_nonzero(interior)) / float(interior.size + 1e-6)
        if density > density_max:
            det_detail["reject"] = f"density={density:.4f}>{density_max:.4f} (max)"
            det_detail["density"] = float(density)
            verification_details.append(det_detail)
            continue
        # Boundary alignment using template-edge kernel at this candidate's scale
        s_used = float(det.get("scale", 1.0))
        ang_used = float(det.get("angle", 0.0))
        tw = max(1, int(round(template_img.shape[1] * s_used)))
        th = max(1, int(round(template_img.shape[0] * s_used)))
        if x1 + tw > W or y1 + th > H:
            det_detail["reject"] = "out_of_bounds"
            verification_details.append(det_detail)
            continue
        # Tower profile: enforce minimum scale (only for vertical templates, not horizontal)
        if not is_horizontal and profile == "tower" and (min_scale_tower is not None) and (s_used < min_scale_tower):
            det_detail["reject"] = f"scale={s_used:.3f}<{min_scale_tower:.2f} (min_req, tower_vertical_only)"
            verification_details.append(det_detail)
            continue
        tpl_s = cv2.resize(template_img, (tw, th), interpolation=cv2.INTER_AREA if s_used < 1.0 else cv2.INTER_CUBIC)
        mask_s = cv2.resize(template_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        tpl_gray = to_gray(apply_preprocessing(tpl_s))
        tpl_edges = cv2.Canny(tpl_gray, 30, 120)
        tpl_edges = cv2.bitwise_and(tpl_edges, tpl_edges, mask=mask_s)
        kernel = (tpl_edges > 0).astype(np.float32)
        ksum = float(kernel.sum())
        if ksum < 10:
            det_detail["reject"] = "kernel_empty"
            verification_details.append(det_detail)
            continue
        dt_patch = scene_dt_full[y1:y1 + th, x1:x1 + tw]
        mean_dt_tpl = float((dt_patch * kernel).sum() / (ksum + 1e-6))
        if mean_dt_tpl > template_edge_dt_max:
            det_detail["reject"] = f"tpl_boundary_dt={mean_dt_tpl:.2f}>{template_edge_dt_max:.2f} (max)"
            det_detail["tpl_boundary_dt"] = float(mean_dt_tpl)
            verification_details.append(det_detail)
            continue
        det_detail["density"] = float(density)
        det_detail["tpl_boundary_dt"] = float(mean_dt_tpl)

        # Verticalness check for tower profile: principal axis near vertical (only for vertical templates)
        if not is_horizontal and profile == "tower" and vertical_tol_deg is not None:
            patch_edges = scene_edges_full[y1:y2, x1:x2]
            ys, xs = np.where(patch_edges > 0)
            if xs.size >= 30:
                xs_f = xs.astype(np.float32)
                ys_f = ys.astype(np.float32)
                xs_f -= xs_f.mean()
                ys_f -= ys_f.mean()
                cov = np.cov(np.vstack([xs_f, ys_f]))
                vals, vecs = np.linalg.eigh(cov)
                v = vecs[:, int(np.argmax(vals))]
                angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
                # normalize to [0, 180)
                if angle_deg < 0:
                    angle_deg += 180.0
                dev_vertical = abs(90.0 - (angle_deg % 180.0))
                det_detail["principal_angle"] = angle_deg
                det_detail["vertical_dev"] = dev_vertical
                if dev_vertical > vertical_tol_deg:
                    det_detail["reject"] = f"vertical_dev={dev_vertical:.1f}°>{vertical_tol_deg:.1f}° (max, principal_angle={angle_deg:.1f}°)"
                    verification_details.append(det_detail)
                    continue
            else:
                det_detail["principal_angle"] = None
                det_detail["vertical_dev"] = None

        passed.append((float(det.get("score", 0.0)), det))
        accepted_details.append(det_detail)
    if passed:
        # prefer highest similarity score
        passed.sort(key=lambda x: x[0], reverse=True)
        kept = [d for _, d in passed[:TOP_K_AFTER_NMS]]
    else:
        # fall back to top-K by score if verification removed all
        kept = kept[:TOP_K_AFTER_NMS]
    debug_info["candidates_after_verification"] = int(len(kept))
    debug_info["accepted_details"] = accepted_details
    debug_info["verification_details"] = verification_details
    debug_info["final_detections"] = [
        {
            "box": [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])],
            "score": float(d.get("score", 0.0)),
            "scale": float(d.get("scale", 0.0)),
            "angle": float(d.get("angle", 0.0)),
        }
        for d in kept
    ]

    # Build reasons for each raw candidate (for display before NMS)
    kept_boxes = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])) for d in kept }
    kept_all_boxes = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])) for d in kept_all }
    distinct_boxes = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])) for d in distinct_sorted }
    accepted_boxes = { tuple(ad["box"]) for ad in accepted_details }
    reject_map = { tuple(vd["box"]): vd.get("reject", "rejected") for vd in verification_details }
    # Rank map (by score) for kept_all to reference "suppressed by #rank"
    kept_all_sorted = sorted(kept_all, key=lambda d: d["score"], reverse=True)
    rank_map = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])): i + 1 for i, d in enumerate(kept_all_sorted) }
    kept_all_scores = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])): float(d.get("score", 0.0)) for d in kept_all_sorted }
    distinct_rank_map = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])): i + 1 for i, d in enumerate(distinct_sorted) }
    distinct_scores = { (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])): float(d.get("score", 0.0)) for d in distinct_sorted }
    topk_cutoff = float(kept[-1]["score"]) if kept else None
    def _iou(b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        ix1 = max(x11, x21); iy1 = max(y11, y21)
        ix2 = min(x12, x22); iy2 = min(y12, y22)
        iw = max(0, ix2 - ix1 + 1); ih = max(0, iy2 - iy1 + 1)
        inter = iw * ih
        a1 = (x12 - x11 + 1) * (y12 - y11 + 1)
        a2 = (x22 - x21 + 1) * (y22 - y21 + 1)
        denom = float(a1 + a2 - inter + 1e-6)
        return float(inter) / denom if denom > 0 else 0.0
    annotated_raw = []
    for c in candidates:
        box = (int(c["x1"]), int(c["y1"]), int(c["x2"]), int(c["y2"]))
        score = float(c.get("score", 0.0))
        dt_est = max(0.0, (1.0 / max(score, 1e-6)) - 1.0)
        if box not in kept_all_boxes:
            # Not present after NMS: suppressed by a higher-score overlapping kept box
            # Find best-IoU kept_all box
            best_rank = None; best_iou = 0.0; best_score = None
            for kb in kept_all_boxes:
                iou_val = _iou(box, kb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_rank = rank_map.get(kb)
                    best_score = kept_all_scores.get(kb)
            if best_iou > 0:
                reason = f"nms_by_#{best_rank}_iou={best_iou:.2f} (kept_score={best_score:.2f} vs {score:.2f})"
            else:
                reason = "nms_suppressed"
        elif box not in distinct_boxes and box in kept_all_boxes:
            # Removed by clustering into a stronger representative
            best_rank = None; best_iou = 0.0; rep_score = None
            for db in distinct_boxes:
                iou_val = _iou(box, db)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_rank = distinct_rank_map.get(db)
                    rep_score = distinct_scores.get(db)
            if best_rank is not None:
                reason = f"cluster_by_#{best_rank}_iou={best_iou:.2f} (rep_score={rep_score:.2f} vs {score:.2f})"
            else:
                reason = "clustered"
        elif box in distinct_boxes and box not in kept_boxes:
            # Survived clustering but cut by top-K
            rnk = distinct_rank_map.get(box)
            if topk_cutoff is not None:
                delta = topk_cutoff - score
                reason = f"dropped_topK rank={rnk} cutoff={topk_cutoff:.2f} Δ={delta:.2f}"
            else:
                reason = f"dropped_topK rank={rnk}"
        elif box in accepted_boxes:
            reason = "accepted"
        elif box in reject_map:
            reason = reject_map[box]
        else:
            reason = "unknown"
        annotated_raw.append({
            "box": [box[0], box[1], box[2], box[3]],
            "score": score,
            "dt_est": round(dt_est, 3),
            "scale": float(c.get("scale", 0.0)),
            "angle": float(c.get("angle", 0.0)),
            "reason": reason,
        })
    debug_info["raw_candidates_annotated"] = annotated_raw

    # Draw on a copy
    annotated = scene_full.copy()
    for i, det in enumerate(kept, start=1):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        # Accepted candidates: draw in red for visibility
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"#{i} s={det['score']:.2f} x{det['scale']:.2f} θ={int(det['angle'])}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    try:
        dt_disp = cv2.normalize(scene_dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug_info["preview_edges"] = scene_edges
        debug_info["preview_dt"] = dt_disp
    except Exception:
        pass

    return annotated, kept, debug_info


# ------------------------------
# App
# ------------------------------
st.set_page_config(page_title="Screw Finder — Fixed Template Matching", page_icon="🔩", layout="wide")

st.title("🔩 Screw Finder — Fixed Template Matching")
st.caption("Hardcoded settings. Scene from MasterImage.png. References are taken from refs/.")

left, right = st.columns([1, 2])

with left:
    st.subheader("📋 Fixed Settings")
    st.markdown("**Files**")
    st.markdown(f"- Scene: `{str(SCENE_PATH)}`")
    st.markdown(f"- Profile: `{PROFILE}`")

    st.markdown("**Preprocessing**")
    st.markdown(f"- Convert to Grayscale: `{CONVERT_TO_GRAYSCALE}`")
    st.markdown(f"- Gaussian Blur: `{GAUSSIAN_BLUR}`")
    st.markdown("**Edge Extraction**")
    st.markdown("- Canny thresholds: `50–150`")

    st.markdown("**Speed**")
    st.markdown(f"- Scene Max Width: `{SCENE_MAX_WIDTH}` px")

    st.markdown("**Matching (Edge NCC)**")
    st.markdown(f"- Base scales: `{SCALES[0]:.2f}` … `{SCALES[-1]:.2f}` ({len(SCALES)} steps)")
    st.markdown(f"- Rotations (deg): `{ROTATIONS}`")
    st.markdown(f"- Method: distance-transform similarity on edges (1/(1+mean distance))")
    st.markdown(f"- Similarity threshold: `{THRESHOLD:.2f}`")
    st.markdown(f"- Peaks per scale: `{MAX_PEAKS_PER_PASS}`, Shortlist top-K: `{TOP_K_AFTER_NMS}`")
    if PROFILE == "tower":
        st.markdown("- Final geometric checks: AR ±10%, verticalness ±7°, template-edge DT ≤ 2.0")
    else:
        st.markdown("- Final geometric checks: AR ±10%, low interior edges, template-edge DT ≤ 1.5")
    st.markdown(f"- Max peaks per pass: `{MAX_PEAKS_PER_PASS}`")
    st.markdown(f"- Global max detections: `{GLOBAL_MAX_DETECTIONS}`")

    st.markdown("**NMS**")
    st.markdown(f"- IoU threshold: `{NMS_IOU_THRESHOLD}`")

with right:
    # Load and preview images
    scene_bgr = cv2.imread(str(SCENE_PATH), cv2.IMREAD_COLOR)
    template_rgba = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_UNCHANGED)

    if scene_bgr is None:
        st.error(f"Scene file not found or unreadable: {SCENE_PATH}")
        st.stop()
    # Primary template is no longer used; only refs/ will be scanned

    st.subheader("📷 Input Images")
    p1, p2 = st.columns(2)
    with p1:
        st.image(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB), caption=f"Scene ({SCENE_PATH.name}) — {scene_bgr.shape[1]}×{scene_bgr.shape[0]}", use_container_width=True)
    with p2:
        st.write("References will be loaded from `refs/` below.")

    # Reference templates list with 150x150 framed thumbnails + checkbox
    st.markdown("**Reference images (from `refs/`) — select to include:**")
    ref_paths = sorted([p for p in REF_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
    if ref_paths:
        cols = st.columns(4)
        for idx, p in enumerate(ref_paths):
            with cols[idx % 4]:
                rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if rgba is None:
                    st.caption(f"{p.name} (unreadable)")
                    continue
                # Build thumbnail on 150×150 canvas
                if rgba.ndim == 3 and rgba.shape[2] == 4:
                    b, g, r, a = cv2.split(rgba)
                    alpha_f = (a.astype(np.float32) / 255.0)[..., None]
                    rgb = cv2.merge([b, g, r]).astype(np.float32)
                    white = np.full_like(rgb, 255.0)
                    prev = (rgb * alpha_f + white * (1.0 - alpha_f)).astype(np.uint8)
                else:
                    prev = rgba if rgba.ndim == 3 else cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
                ph, pw = prev.shape[:2]
                scale = min(150.0 / max(ph, 1), 150.0 / max(pw, 1))
                new_w = max(1, int(round(pw * scale)))
                new_h = max(1, int(round(ph * scale)))
                thumb = cv2.resize(prev, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.full((150, 150, 3), 240, dtype=np.uint8)
                off_x = (150 - new_w) // 2
                off_y = (150 - new_h) // 2
                canvas[off_y:off_y + new_h, off_x:off_x + new_w] = thumb
                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"{p.name}", width=150, use_container_width=False)
                st.checkbox("Include", value=False, key=f"include_ref_{idx}")
                
                # Load and display config
                config = load_or_create_config(p, rgba)
                if config:
                    det_cfg = config["detection"]
                    ver_cfg = config["verification"]
                    nms_cfg = config["nms"]
                    st.markdown(f"**Config:** `{p.stem}.json`")
                    st.table({
                        "Threshold": det_cfg["similarity_threshold"],
                        "Scale": f"{det_cfg.get('scale_min', 0.8):.2f}-{det_cfg.get('scale_max', 1.2):.2f}" if det_cfg.get("scale_mode") == "range" else f"Width {det_cfg.get('target_width_min', 0):.0f}-{det_cfg.get('target_width_max', 0):.0f}",
                        "Rotations": str(det_cfg["rotations"]),
                        "Edge DT": ver_cfg["template_edge_dt_max"],
                        "Density": ver_cfg["density_max"],
                        "NMS IoU": nms_cfg["iou_threshold"]
                    })
                else:
                    st.warning("⚠️ Config failed to load")

    st.divider()
    if st.button("▶️ Process", type="primary", key="process_button"):
        # Create progress bar and scrolling status log
        progress_bar = st.progress(0)
        status_log_container = st.empty()
        status_log = []
        update_counter = [0]  # Use list to avoid nonlocal issues
        
        def update_status(message: str):
            """Append a message to the status log and update the textbox"""
            status_log.append(message)
            update_counter[0] += 1
            # Use unique key based on update counter to avoid duplicate key errors
            status_log_container.text_area("Progress Log", value="\n".join(status_log), height=200, disabled=True, key=f"progress_log_{update_counter[0]}")
        
        try:
            # Step 1: Build list of items with configs
            update_status("📋 Step 1/4: Loading templates and configurations...")
            progress_bar.progress(0.1)
            items = []
            palette = [(0,165,255), (0,255,255), (255,0,255), (0,255,0), (255,128,0), (128,0,255)]
            pal_idx = 0
            for idx, p in enumerate(ref_paths):
                include = st.session_state.get(f"include_ref_{idx}", True)
                if not include:
                    continue
                rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if rgba is None:
                    continue
                # Load config for this template
                config = load_or_create_config(p, rgba)
                if config is None:
                    continue  # Skip if config is invalid
                color = palette[pal_idx % len(palette)]; pal_idx += 1
                items.append((p.name, rgba, config, color))

            if not items:
                update_status("⚠️ No templates selected for processing.")
                progress_bar.progress(1.0)
                st.stop()

            # Step 2: Process each template
            grouped = []
            combined = scene_bgr.copy()
            debug_info = None
            total_templates = len(items)
            
            for template_idx, (name, rgba, config, color) in enumerate(items):
                progress_pct = 0.2 + (template_idx / total_templates) * 0.7
                update_status(f"🔍 Step 2/4: Processing template {template_idx + 1}/{total_templates}: {name}...")
                progress_bar.progress(progress_pct)
                
                tpl_bgr_c, tpl_mask_c = build_template_from_png(rgba)
                
                # Check if we should split rotations into separate batches
                rotations = config["detection"].get("rotations", [0])
                split_rotations = len(rotations) > 1
                
                if split_rotations:
                    # Run detection separately for each rotation and combine results
                    all_detections = []
                    all_debug_info = []
                    total_rotations = len(rotations)
                    for rot_idx, rot_angle in enumerate(rotations):
                        rot_progress = progress_pct + (rot_idx / total_rotations) * (0.7 / total_templates)
                        update_status(f"🔄 Processing {name} at {rot_angle}° rotation ({rot_idx + 1}/{total_rotations})...")
                        progress_bar.progress(rot_progress)
                        
                        # Create a config copy with single rotation
                        rot_config = config.copy()
                        rot_config["detection"] = config["detection"].copy()
                        rot_config["detection"]["rotations"] = [rot_angle]
                        
                        # Run detection for this rotation
                        annotated_rot, detections_rot, debug_rot = match_multiscale(
                            scene_bgr, tpl_bgr_c, tpl_mask_c, rot_config
                        )
                        
                        all_detections.extend(detections_rot)
                        if debug_rot:
                            all_debug_info.append(debug_rot)
                        
                        # Update status with detection count
                        update_status(f"✅ {name} at {rot_angle}°: Found {len(detections_rot)} detection(s)")
                    
                    # Combine debug info (use first one as primary)
                    combined_debug = all_debug_info[0] if all_debug_info else {}
                    if len(all_debug_info) > 1:
                        # Merge passes from all rotations
                        combined_passes = []
                        for dbg in all_debug_info:
                            combined_passes.extend(dbg.get("passes", []))
                        combined_debug["passes"] = combined_passes
                        # Merge other debug info
                        combined_debug["candidates_before_nms"] = sum(dbg.get("candidates_before_nms", 0) for dbg in all_debug_info)
                    
                    detections_i = all_detections
                    debug_i = combined_debug
                    
                    # Update status with combined detection count
                    update_status(f"✅ {name}: Combined {len(detections_i)} detection(s) from {total_rotations} rotation(s)")
                else:
                    # Run detection normally (single rotation or no rotation)
                    update_status(f"🔍 Running detection for {name}...")
                    annotated_i, detections_i, debug_i = match_multiscale(
                        scene_bgr, tpl_bgr_c, tpl_mask_c, config
                    )
                    # Update status with detection count
                    update_status(f"✅ {name}: Found {len(detections_i)} detection(s)")

                grouped.append({"name": name, "detections": detections_i, "debug": debug_i or {}, "color": color, "config": config})
                combined = draw_detections(combined, detections_i, color, name)
                if debug_info is None:
                    debug_info = debug_i or {}
            
            # Step 3: Finalize results
            total_detections = sum(len(g["detections"]) for g in grouped)
            update_status(f"✅ Step 3/4: Finalizing results... ({total_detections} total detection(s) across {total_templates} template(s))")
            progress_bar.progress(0.9)
            
            # Step 4: Complete
            update_status(f"✅ Step 4/4: Complete! Found {total_detections} detection(s) in total")
            progress_bar.progress(1.0)
            
        except Exception as e:
            update_status(f"❌ Error during processing: {e}")
            progress_bar.progress(1.0)
            raise

        total = sum(len(g["detections"]) for g in grouped)
        st.success(f"Found {total} match(es) after NMS across {len(grouped)} template(s).")
        st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), caption="Annotated Matches (All Templates)", use_container_width=True)

        # Plot raw candidates before NMS
        rcands = (debug_info or {}).get("raw_candidates_annotated") or (debug_info or {}).get("raw_candidates")
        if rcands:
            # Draw top-N by score to keep the view readable
            top_rcands = sorted(rcands, key=lambda d: d.get("score", 0.0), reverse=True)[:150]
            raw_vis = scene_bgr.copy()
            H, W = raw_vis.shape[:2]
            def _clamp_box(b):
                x1, y1, x2, y2 = [int(v) for v in b]
                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(0, min(W - 1, x2))
                y2 = max(0, min(H - 1, y2))
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                return x1, y1, x2, y2
            for idx, c in enumerate(top_rcands, start=1):
                x1, y1, x2, y2 = _clamp_box(c["box"])
                color = (0, 165, 255)  # orange
                cv2.rectangle(raw_vis, (x1, y1), (x2, y2), color, 2)
                label = f"{idx}:{c['score']:.2f}"
                cv2.putText(raw_vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            st.image(cv2.cvtColor(raw_vis, cv2.COLOR_BGR2RGB), caption=f"Raw Candidates (before NMS) — showing {len(top_rcands)}", use_container_width=True)

        # Plot after-NMS (before verification) candidates for side-by-side comparison
        nms_cands = (debug_info or {}).get("nms_candidates")
        if nms_cands:
            nms_vis = scene_bgr.copy()
            H, W = nms_vis.shape[:2]
            for idx, c in enumerate(nms_cands, start=1):
                x1, y1, x2, y2 = _clamp_box(c["box"])
                cv2.rectangle(nms_vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
                label = f"{idx}:{c['score']:.2f}"
                cv2.putText(nms_vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            st.image(cv2.cvtColor(nms_vis, cv2.COLOR_BGR2RGB), caption=f"Candidates after NMS (before verification) — {len(nms_cands)}", use_container_width=True)

        # List detections
        # Grouped detections
        if grouped:
            for g in grouped:
                if g["detections"]:
                    st.markdown(f"**Detections for {g['name']}:**")
                    st.table([
                        {"box": [int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2'])],
                         "score": float(d['score']), "scale": float(d['scale']), "angle": float(d['angle'])}
                        for d in g["detections"]
                    ])
        # Legacy single list if present
        if False and detections:
            st.markdown("**Detections:**")
            for i, det in enumerate(detections, start=1):
                st.markdown(
                    f"- #{i}: box=({det['x1']}, {det['y1']})–({det['x2']}, {det['y2']}), "
                    f"score={det['score']:.3f}, scale={det['scale']:.3f}, angle={det['angle']:.1f}°"
                )
            # Tabular final detections
            st.markdown("**Final detections (tabular):**")
            st.table([
                {
                    "box": [int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2'])],
                    "score": float(d['score']),
                    "scale": float(d['scale']),
                    "angle": float(d['angle']),
                } for d in detections
            ])
            # Download JSON
            import json as _json
            final_json = _json.dumps([
                {
                    "x1": int(d['x1']), "y1": int(d['y1']),
                    "x2": int(d['x2']), "y2": int(d['y2']),
                    "score": float(d['score']), "scale": float(d['scale']), "angle": float(d['angle'])
                } for d in detections
            ], indent=2)
            st.download_button("📥 Download final detections (JSON)",
                               data=final_json.encode("utf-8"),
                               file_name="final_detections.json",
                               mime="application/json")

        # Debug details
        with st.expander("🧪 Debug Details", expanded=True):
            st.write("**Runtime scales:**", debug_info.get("scales"))
            st.write("**Rotations:**", debug_info.get("rotations"))
            st.write(f"**Scene (full H×W):** {debug_info.get('scene_shape')}")
            st.write(f"**Scene downscaled (H×W):** {debug_info.get('scene_downscaled_shape')}  |  scale_factor={debug_info.get('scale_factor')}")
            st.write(f"**Template (H×W):** {debug_info.get('template_shape')}  |  mask_nonzero={debug_info.get('template_mask_nonzero')}")
            st.write(f"**Candidates before NMS:** {debug_info.get('candidates_before_nms', 0)}")
            st.write(f"**Candidates after NMS:** {debug_info.get('candidates_after_nms', 0)}")
            st.write(f"**Candidates after verification:** {debug_info.get('candidates_after_verification', 0)}")
            vth = debug_info.get("verification_thresholds")
            if vth:
                st.write("**Verification thresholds:**", vth)
            if "fallback" in debug_info:
                st.write("**Fallback best:**", debug_info["fallback"])
            rcands = debug_info.get("raw_candidates_annotated") or debug_info.get("raw_candidates")
            if rcands:
                st.write("**Candidates before NMS (deduplicated with reasons):**")
                # Deduplicate by box and aggregate best score and final status
                agg = {}
                for c in rcands:
                    key = tuple(c["box"])
                    entry = agg.get(key)
                    reason = c.get("reason")
                    score = float(c.get("score", 0.0))
                    if entry is None:
                        agg[key] = {
                            "box": list(key),
                            "best_score": score,
                            "any_reason": reason,
                            "occurrences": 1,
                        }
                    else:
                        entry["best_score"] = max(entry["best_score"], score)
                        entry["occurrences"] += 1
                        if entry.get("any_reason") is None and reason:
                            entry["any_reason"] = reason
                # Sort by best_score desc
                agg_rows = sorted(agg.values(), key=lambda r: r["best_score"], reverse=True)
                st.table(agg_rows[:150])
            # Show top-10 passes by max similarity
            passes = debug_info.get("passes", [])
            if passes:
                top_passes = sorted(passes, key=lambda d: d["max_similarity"], reverse=True)[:10]
                st.write("**Top passes (by max similarity):**")
                st.table([{k: v for k, v in p.items()} for p in top_passes])
            # Verification details for each kept-after-NMS candidate
            vdetails = debug_info.get("verification_details")
            if vdetails:
                st.write("**Verification details (per candidate before verification):**")
                st.table(vdetails)
            nms_list = debug_info.get("nms_candidates")
            if nms_list:
                st.write("**Candidates after NMS (before verification):**")
                st.table(nms_list)
            accepted = debug_info.get("accepted_details")
            if accepted:
                st.write("**Accepted candidates with metrics:**")
                st.table(accepted)
            # Show previews
            if "preview_edges" in debug_info:
                e1, e2 = st.columns(2)
                with e1:
                    st.write("Scene edges (downscaled)")
                    st.image(debug_info["preview_edges"], use_container_width=True, clamp=True)
                with e2:
                    st.write("Scene distance transform (normalized)")
                    st.image(debug_info["preview_dt"], use_container_width=True, clamp=True)

        # Download
        st.divider()
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        buf.seek(0)
        st.download_button("📥 Download Annotated (PNG)", buf, file_name="matches.png", mime="image/png")

# Stop here to avoid executing any legacy code below (if present)
st.stop()

"""
🔩 Screw Finder — Generalized Hough (Raster Detector)
A Streamlit app for detecting screws or small circular features from high-DPI rasterized engineering drawings
using Generalized Hough Transform (Guil) from opencv-contrib-python-headless.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, List, Optional
import math

# Page config
st.set_page_config(
    page_title="Screw Finder - GHT Detector",
    page_icon="🔩",
    layout="wide"
)

# ------------------------------
# Fixed-mode configuration (for fast testing)
# ------------------------------
from pathlib import Path

FIXED_MODE_ENABLED = True  # Set to True to use hardcoded settings and local files
APP_DIR = Path(__file__).parent.resolve()

# Fixed file paths (under app path)
FIXED_SCENE_PATH = APP_DIR / "MasterImage.jpg"
FIXED_TEMPLATE_PATH = APP_DIR / "ScrewSquare.png"

# Fixed preprocessing settings
FIXED_GAUSSIAN_BLUR = 3            # odd, 0 disables
FIXED_CANNY_LOW = 30
FIXED_CANNY_HIGH = 100
FIXED_BINARIZE = False

# Fixed speed control
FIXED_MAX_WIDTH = 2000             # downscale target width for scene

# Fixed detection parameters (suggested defaults for vertical screws)
FIXED_MIN_SCALE = 0.80
FIXED_MAX_SCALE = 1.20
FIXED_SCALE_STEP = 0.02
FIXED_MIN_ANGLE = 85               # degrees
FIXED_MAX_ANGLE = 95               # degrees
FIXED_DP = 1
FIXED_LEVELS = 3                   # >=1
FIXED_MIN_VOTES = 10
FIXED_NMS_DISTANCE = 25            # px at downscaled size

# Fixed overlay
FIXED_SHOW_OVERLAY = True


# Check for opencv-contrib
try:
    from cv2 import createGeneralizedHoughGuil
    HAS_CONTRIB = True
except ImportError:
    HAS_CONTRIB = False
    st.error("⚠️ **Missing opencv-contrib-python-headless**\n\n"
             "Please install it with: `pip install opencv-contrib-python-headless`")
    st.stop()


def normalize_angle_range(min_angle: float, max_angle: float) -> Tuple[float, float]:
    """
    Normalize angle range to [0, 360] and handle wrap-around.
    Returns (min, max) in [0, 360] range.
    """
    min_angle = min_angle % 360
    max_angle = max_angle % 360
    
    if min_angle > max_angle:
        # Wrap-around case (e.g., 350-10)
        # For GHT, we'll use the full range but need to handle it
        # OpenCV GHT expects min <= max, so we'll use 0-360 and filter later
        return 0.0, 360.0
    return min_angle, max_angle


def analyze_template_quality(edges: np.ndarray, img_shape: Tuple[int, int]) -> dict:
    """
    Analyze template quality for GHT and provide recommendations.
    Returns a dict with analysis results and recommendations.
    """
    h, w = edges.shape
    edge_count = np.count_nonzero(edges)
    edge_density = edge_count / (h * w)
    aspect_ratio = h / w if w > 0 else 0
    total_pixels = h * w
    
    # Optimal ranges for GHT
    optimal_aspect_ratio_range = (0.3, 3.0)  # Height/width ratio
    optimal_size_range = (50, 500)  # Pixels in each dimension
    optimal_edge_density = (0.02, 0.30)  # 2-30% edge pixels
    optimal_total_pixels = (2500, 250000)  # Total pixels
    
    issues = []
    recommendations = []
    quality_score = 100
    
    # Check aspect ratio
    if aspect_ratio > optimal_aspect_ratio_range[1]:
        issues.append(f"Template is too tall (aspect ratio {aspect_ratio:.2f}:1)")
        recommendations.append("Rotate template 90° or crop to be wider")
        quality_score -= 30
    elif aspect_ratio < optimal_aspect_ratio_range[0]:
        issues.append(f"Template is too wide (aspect ratio {aspect_ratio:.2f}:1)")
        recommendations.append("Rotate template 90° or crop to be taller")
        quality_score -= 30
    
    # Check dimensions
    if w < optimal_size_range[0] or h < optimal_size_range[0]:
        issues.append(f"Template too small ({w}×{h} px)")
        recommendations.append("Crop a larger region around the screw")
        quality_score -= 20
    elif w > optimal_size_range[1] or h > optimal_size_range[1]:
        issues.append(f"Template very large ({w}×{h} px)")
        recommendations.append("Consider downscaling template slightly")
        quality_score -= 10
    
    # Check edge density
    if edge_density < optimal_edge_density[0]:
        issues.append(f"Too few edges ({edge_density:.1%} density)")
        recommendations.append("Lower Canny thresholds or increase contrast")
        quality_score -= 25
    elif edge_density > optimal_edge_density[1]:
        issues.append(f"Too many edges ({edge_density:.1%} density)")
        recommendations.append("Increase Canny thresholds or add blur")
        quality_score -= 15
    
    # Check total pixels
    if total_pixels < optimal_total_pixels[0]:
        issues.append(f"Template too small overall ({total_pixels} pixels)")
        recommendations.append("Include more context around the screw")
        quality_score -= 15
    elif total_pixels > optimal_total_pixels[1]:
        issues.append(f"Template very large ({total_pixels} pixels)")
        recommendations.append("Consider cropping tighter around the screw")
        quality_score -= 10
    
    # Check edge count
    if edge_count < 50:
        issues.append(f"Very few edge pixels ({edge_count})")
        recommendations.append("Ensure at least 50-100 edge pixels for reliable detection")
        quality_score -= 20
    
    return {
        'width': w,
        'height': h,
        'aspect_ratio': aspect_ratio,
        'edge_count': edge_count,
        'edge_density': edge_density,
        'total_pixels': total_pixels,
        'issues': issues,
        'recommendations': recommendations,
        'quality_score': max(0, quality_score),
        'is_optimal': len(issues) == 0
    }


def compute_template_center(edges: np.ndarray) -> Tuple[int, int]:
    """Compute centroid of edge pixels as template center."""
    y_coords, x_coords = np.where(edges > 0)
    if len(x_coords) == 0:
        return (edges.shape[1] // 2, edges.shape[0] // 2)
    cx = int(np.mean(x_coords))
    cy = int(np.mean(y_coords))
    return (cx, cy)


def get_template_bbox(template_shape: Tuple[int, int], scale: float, angle: float) -> np.ndarray:
    """
    Compute bounding box corners for a rotated and scaled template.
    Returns 4x2 array of corner coordinates.
    """
    h, w = template_shape
    # Scale dimensions
    sw = w * scale
    sh = h * scale
    
    # Corners in template space (centered at origin)
    corners = np.array([
        [-sw/2, -sh/2],
        [sw/2, -sh/2],
        [sw/2, sh/2],
        [-sw/2, sh/2]
    ], dtype=np.float32)
    
    # Rotate
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated_corners = corners @ rotation_matrix.T
    return rotated_corners


def non_max_suppression(
    positions: np.ndarray,
    votes: np.ndarray,
    scales: Optional[np.ndarray],
    angles: Optional[np.ndarray],
    template_shape: Tuple[int, int],
    nms_distance: float
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform Non-Maximum Suppression by center distance.
    Returns filtered positions, votes, scales, angles.
    """
    if len(positions) == 0:
        return positions, votes, scales, angles
    
    # Sort by votes (descending)
    sorted_indices = np.argsort(votes)[::-1]
    positions = positions[sorted_indices]
    votes = votes[sorted_indices]
    if scales is not None:
        scales = scales[sorted_indices]
    if angles is not None:
        angles = angles[sorted_indices]
    
    keep = []
    for i in range(len(positions)):
        if i == 0:
            keep.append(i)
            continue
        
        # Check distance to all kept detections
        current_pos = positions[i]
        too_close = False
        for kept_idx in keep:
            kept_pos = positions[kept_idx]
            dist = np.linalg.norm(current_pos - kept_pos)
            if dist < nms_distance:
                too_close = True
                break
        
        if not too_close:
            keep.append(i)
    
    keep = np.array(keep)
    result_positions = positions[keep]
    result_votes = votes[keep]
    result_scales = scales[keep] if scales is not None else None
    result_angles = angles[keep] if angles is not None else None
    
    return result_positions, result_votes, result_scales, result_angles


def process_image_for_edges(
    img: np.ndarray,
    gaussian_blur: int,
    canny_low: int,
    canny_high: int,
    binarize: bool
) -> np.ndarray:
    """Process image to extract edges."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Optional binarization
    if binarize:
        _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Gaussian blur
    if gaussian_blur > 0 and gaussian_blur % 2 == 1:
        gray = cv2.GaussianBlur(gray, (gaussian_blur, gaussian_blur), 0)
    
    # Canny edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)
    
    return edges


def resize_with_aspect_ratio(img: np.ndarray, target_width: int) -> Tuple[np.ndarray, float]:
    """
    Resize image maintaining aspect ratio.
    Returns (resized_image, scale_factor).
    """
    if img.shape[1] <= target_width:
        return img, 1.0
    
    scale = target_width / img.shape[1]
    new_height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def try_template_matching_fallback(
    reference_img: np.ndarray,
    drawing_img: np.ndarray,
    reference_edges: np.ndarray,
    scene_edges: np.ndarray,
    min_scale: float,
    max_scale: float,
    status_placeholder,
    progress_bar
):
    """Fallback method using template matching when GHT fails."""
    status_placeholder.info("🔄 **Trying Template Matching (Fallback Method)...**")
    progress_bar.progress(0.3)
    
    # Convert to grayscale if needed
    if len(reference_img.shape) == 3:
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_img
    
    if len(drawing_img.shape) == 3:
        scene_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
    else:
        scene_gray = drawing_img
    
    # Downscale scene for speed
    scene_gray_resized, scale_factor = resize_with_aspect_ratio(scene_gray, 2000)
    
    status_placeholder.info(f"🔄 Running multi-scale template matching (scale: {min_scale:.2f}-{max_scale:.2f})...")
    progress_bar.progress(0.5)
    
    matches = []
    scales_to_try = np.arange(min_scale, max_scale + 0.1, 0.1)
    
    for scale in scales_to_try:
        # Resize template
        template_w = int(ref_gray.shape[1] * scale)
        template_h = int(ref_gray.shape[0] * scale)
        if template_w < 10 or template_h < 10 or template_w > scene_gray_resized.shape[1] or template_h > scene_gray_resized.shape[0]:
            continue
        
        template_resized = cv2.resize(ref_gray, (template_w, template_h))
        
        # Template matching
        result = cv2.matchTemplate(scene_gray_resized, template_resized, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        
        # Find matches
        locations = np.where(result >= threshold)
        for pt in zip(*locations[::-1]):
            matches.append({
                'x': int(pt[0] / scale_factor),
                'y': int(pt[1] / scale_factor),
                'confidence': float(result[pt[1], pt[0]]),
                'scale': scale
            })
    
    if len(matches) == 0:
        status_placeholder.warning("🔍 **Template Matching also found no detections.**")
        st.info("💡 **Suggestions:**")
        st.write("1. Check that your reference screw actually appears in the drawing")
        st.write("2. Try rotating the reference image 90°")
        st.write("3. Ensure the reference is a clean crop of a single screw")
        st.write("4. Try adjusting Canny thresholds to get better edge detection")
        return
    
    # Display results
    status_placeholder.success(f"✅ **Template Matching found {len(matches)} potential match(es)!**")
    progress_bar.progress(1.0)
    
    # Draw matches on image
    annotated = drawing_img.copy()
    for i, match in enumerate(matches[:20]):  # Limit to 20 matches
        x, y = match['x'], match['y']
        w = int(ref_gray.shape[1] * match['scale'] / scale_factor)
        h = int(ref_gray.shape[0] * match['scale'] / scale_factor)
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"#{i+1}: {match['confidence']:.2f}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    st.subheader("📊 Template Matching Results")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    st.write(f"**Found {len(matches)} match(es) with confidence ≥ 0.6**")
    st.write("**Note:** Template matching is less robust to rotation than GHT, but works better with extreme aspect ratios.")


def main():
    st.title("🔩 Screw Finder — Generalized Hough (Raster Detector)")
    st.markdown("Detect screws or small circular features from high-DPI rasterized engineering drawings.")

    if FIXED_MODE_ENABLED:
        st.info("⚙️ Running in Fixed Mode (hardcoded settings, no user inputs)")
        return run_fixed_mode()

    # Sidebar
    with st.sidebar:
        st.header("📤 File Upload")
        drawing_file = st.file_uploader(
            "Upload Full Drawing (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            help="Upload the high-resolution rasterized drawing"
        )
        reference_file = st.file_uploader(
            "Upload Reference Screw (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            help="Upload a cropped reference image of one screw"
        )
        
        st.divider()
        
        st.header("⚙️ Preprocessing")
        gaussian_blur = st.slider("Gaussian Blur Kernel Size", 0, 15, 3, step=2, help="0 = no blur, must be odd")
        canny_low = st.slider("Canny Low Threshold", 0, 255, 50)
        canny_high = st.slider("Canny High Threshold", 0, 255, 150)
        binarize = st.checkbox("Binarize (Threshold)", value=False, help="Apply binary threshold before edge detection")
        
        st.divider()
        
        st.header("🚀 Speed Control")
        max_width = st.slider("Resize Scene Max Width (px)", 800, 4000, 1600, step=200, 
                             help="Downscale scene for faster detection, then map back to full resolution")
        
        st.divider()
        
        st.header("🎯 Detection Parameters")
        min_scale = st.slider("Min Scale", 0.05, 2.0, 0.9, step=0.05, 
                             help="Lower values needed when scene is downscaled significantly")
        max_scale = st.slider("Max Scale", 0.1, 3.0, 1.2, step=0.05,
                             help="Higher values may be needed for larger objects")
        scale_step = st.slider("Scale Step", 0.01, 0.1, 0.05, step=0.01)
        
        min_angle = st.slider("Min Angle (degrees)", 0, 360, 0)
        max_angle = st.slider("Max Angle (degrees)", 0, 360, 360)
        st.caption("💡 Tip: For wrap-around (e.g., 350-10), set min > max manually in code or use 0-360")
        
        dp = st.slider("Accumulator dp (resolution)", 1, 5, 1, help="1 = full resolution, higher = coarser")
        levels = st.slider("Pyramid Levels", 1, 5, 1, help="≥1 required by OpenCV")
        min_votes = st.slider("Minimum Votes", 1, 200, 30, help="Minimum accumulator votes for detection")
        nms_distance = st.slider("NMS Distance (px)", 10, 200, 50, help="Non-maximum suppression distance")
        
        st.divider()
        
        st.header("🐛 Debug")
        show_debug_edges = st.checkbox("Show Debug Edges", value=False, help="Display reference and scene edge maps")
        
        st.divider()
        
        st.header("🧪 Quick Test")
        use_test_mode = st.checkbox("Use Permissive Test Parameters", value=False, 
                                    help="Temporarily use very permissive parameters to test if detection works")
        
        st.divider()
        
        st.header("🔍 Detection Troubleshooting")
        st.write("**If detection is wrong or missing:**")
        st.write("1. **Lower Minimum Votes** (try 10-20)")
        st.write("2. **Widen Scale Range** (check warning above)")
        st.write("3. **Lower Canny thresholds** (try 30-100)")
        st.write("4. **Enable 'Show Debug Edges'** to verify edge detection")
        st.write("5. **Check template quality** (see analysis above)")
    
    # Main area
    if drawing_file is None or reference_file is None:
        st.info("👆 Please upload both the full drawing and the reference screw image to begin detection.")
        return
    
    # Progress container
    progress_container = st.container()
    
    # Read images
    with progress_container:
        st.markdown("### 📋 Processing Steps")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
    try:
        status_placeholder.info("📥 **Step 1/8: Reading images from uploads...**")
        progress_bar.progress(1/8)
        
        drawing_bytes = drawing_file.read()
        drawing_np = np.frombuffer(drawing_bytes, np.uint8)
        drawing_img = cv2.imdecode(drawing_np, cv2.IMREAD_COLOR)
        
        reference_bytes = reference_file.read()
        reference_np = np.frombuffer(reference_bytes, np.uint8)
        reference_img = cv2.imdecode(reference_np, cv2.IMREAD_COLOR)
        
        if drawing_img is None or reference_img is None:
            status_placeholder.error("❌ Failed to decode images. Please check file formats.")
            return
        
        status_placeholder.success(f"✅ **Step 1/8 Complete:** Read drawing ({drawing_img.shape[1]}×{drawing_img.shape[0]} px) and reference ({reference_img.shape[1]}×{reference_img.shape[0]} px)")
        
    except Exception as e:
        status_placeholder.error(f"❌ Error reading images: {e}")
        return
    
    # Process reference image
    status_placeholder.info("🖼️ **Step 2/8: Processing reference image (grayscale → blur → Canny edges)...**")
    progress_bar.progress(2/8)
    
    reference_edges = process_image_for_edges(
        reference_img, gaussian_blur, canny_low, canny_high, binarize
    )
    
    edge_count = np.count_nonzero(reference_edges)
    
    # Analyze template quality
    template_analysis = analyze_template_quality(reference_edges, reference_img.shape)
    
    if edge_count < 40:
        status_placeholder.warning(f"⚠️ **Reference has too few edges ({edge_count} pixels).** Crop tighter or increase contrast.")
        if not show_debug_edges:
            return
    
    # Show template quality analysis
    if not template_analysis['is_optimal']:
        status_placeholder.warning(f"⚠️ **Template Quality Score: {template_analysis['quality_score']}/100**")
        with st.expander("📋 Template Quality Analysis & Recommendations", expanded=True):
            st.write("**Current Template:**")
            st.write(f"- Size: {template_analysis['width']}×{template_analysis['height']} px")
            st.write(f"- Aspect ratio: {template_analysis['aspect_ratio']:.2f}:1 (height/width)")
            st.write(f"- Edge pixels: {template_analysis['edge_count']}")
            st.write(f"- Edge density: {template_analysis['edge_density']:.1%}")
            st.write(f"- Total pixels: {template_analysis['total_pixels']:,}")
            st.write("")
            st.write("**⚠️ Issues Found:**")
            for issue in template_analysis['issues']:
                st.write(f"- {issue}")
            st.write("")
            st.write("**💡 Recommendations:**")
            for rec in template_analysis['recommendations']:
                st.write(f"- {rec}")
            st.write("")
            st.write("**✅ Optimal Template Guidelines for GHT:**")
            st.write("- **Aspect ratio:** 0.3:1 to 3:1 (height/width) - roughly square to moderately rectangular")
            st.write("- **Dimensions:** 50-500 pixels per side")
            st.write("- **Edge density:** 2-30% of pixels should be edges")
            st.write("- **Total size:** 2,500-250,000 pixels")
            st.write("- **Edge count:** At least 50-100 edge pixels")
    else:
        status_placeholder.success(f"✅ **Step 2/8 Complete:** Reference edges extracted ({edge_count} edge pixels) - Template quality: Excellent!")
    
    progress_bar.progress(3/8)
    
    # Process scene image
    status_placeholder.info("🖼️ **Step 3/8: Processing scene image (grayscale → blur → Canny edges → resize)...**")
    progress_bar.progress(3/8)
    
    scene_edges_full = process_image_for_edges(
        drawing_img, gaussian_blur, canny_low, canny_high, binarize
    )
    
    # Downscale for detection
    scene_edges, scale_factor = resize_with_aspect_ratio(scene_edges_full, max_width)
    
    scene_edge_count = np.count_nonzero(scene_edges)
    
    # Calculate expected scale range based on downscaling
    # If scene is downscaled, the template needs to be scaled down proportionally
    ref_width = reference_edges.shape[1]
    ref_height = reference_edges.shape[0]
    scene_downscaled_width = scene_edges.shape[1]
    scene_full_width = scene_edges_full.shape[1]
    
    # Expected scale calculation:
    # The scale_factor tells us how much the scene was downscaled
    # If scene was downscaled significantly (e.g., 8272->1600), scale_factor = 1600/8272 = 0.193
    # If template is at original/full resolution, it needs to be scaled by scale_factor to match
    # But we also need to account for the fact that the template might already be at a different resolution
    
    # The key insight: if scene is downscaled by X, and template is at original size,
    # then template needs scale = scale_factor to match the downscaled scene
    # However, if both are already at similar scales, expected_scale should be ~1.0
    
    # For now, assume template is at original resolution and scene is downscaled
    # So expected scale = scale_factor (the downscale factor)
    expected_scale = scale_factor
    
    # Determine expected scale based on downscaling
    # If scene is barely downscaled (>0.95), template and scene are at similar resolutions
    # If scene is significantly downscaled (≤0.95), template needs proportional scaling
    if scale_factor > 0.95:
        # Minimal downscaling (e.g., 1608->1600px) - template and scene at similar scales
        expected_scale = 1.0
        recommended_min_scale = 0.8
        recommended_max_scale = 1.2
        scale_reason = "minimal downscaling - template and scene are at similar resolutions"
    else:
        # Significant downscaling (e.g., 8272->1600px) - template needs to be scaled down
        expected_scale = scale_factor
        # Calculate recommended scale range with ±50% variation
        recommended_min_scale = max(0.05, expected_scale * 0.5)
        recommended_max_scale = min(3.0, expected_scale * 2.0)
        scale_reason = f"significant downscaling ({scale_factor:.3f}x) - template needs proportional scaling"
    
    # Check if current scale range covers the expected scale
    scale_range_ok = (min_scale <= expected_scale <= max_scale)
    
    if not scale_range_ok:
        # Show prominent warning outside expander
        st.error(f"⚠️ **SCALE RANGE MISMATCH!** Expected scale ({expected_scale:.3f}) is outside your range ({min_scale:.2f} - {max_scale:.2f})")
        
        # Show recommended values very prominently
        st.markdown("---")
        st.markdown("### 🔧 **QUICK FIX - Adjust These Values in Sidebar:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4;">
            <h4>📉 Min Scale</h4>
            <p style="font-size: 24px; font-weight: bold; color: #1f77b4;">{recommended_min_scale:.2f}</p>
            <p>Set the <strong>Min Scale</strong> slider in sidebar to this value</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background-color: #fff4e6; padding: 15px; border-radius: 5px; border-left: 4px solid #ff7f0e;">
            <h4>📈 Max Scale</h4>
            <p style="font-size: 24px; font-weight: bold; color: #ff7f0e;">{recommended_max_scale:.2f}</p>
            <p>Set the <strong>Max Scale</strong> slider in sidebar to this value</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("🔧 Scale Range Issue - Detailed Explanation", expanded=True):
            st.write("**The Problem:**")
            st.write(f"- Scene was downscaled by **{scale_factor:.3f}x** (from {scene_full_width}px to {scene_downscaled_width}px)")
            st.write(f"- Reference template is **{ref_width}×{ref_height}px** (at original/full resolution)")
            st.write(f"- **Reason:** {scale_reason}")
            st.write(f"- To match downscaled scene, template needs scale of **~{expected_scale:.3f}x**")
            st.write(f"- Your current scale range: **{min_scale:.2f} - {max_scale:.2f}**")
            st.write("")
            if expected_scale < min_scale:
                st.error(f"❌ **Expected scale ({expected_scale:.3f}) is BELOW your minimum ({min_scale:.2f})!**")
            elif expected_scale > max_scale:
                st.error(f"❌ **Expected scale ({expected_scale:.3f}) is ABOVE your maximum ({max_scale:.2f})!**")
            st.write("")
            st.write("**💡 Why the recommendation changed:**")
            if scale_factor > 0.95:
                st.info("""
                **Previous suggestion (0.1-0.4) was for a different scenario:**
                - When scene is heavily downscaled (e.g., 8272px → 1600px = 0.193x), 
                  template needs scale ~0.19x, so range should be 0.1-0.4
                
                **Current situation:**
                - Scene is barely downscaled (1608px → 1600px = 0.995x)
                - Template and scene are at similar resolutions
                - Template needs scale ~1.0x, so range should be 0.8-1.2
                """)
            else:
                st.info(f"""
                **Current situation:**
                - Scene is significantly downscaled ({scale_factor:.3f}x)
                - Template needs to be scaled down proportionally
                - Recommended range: {recommended_min_scale:.2f} - {recommended_max_scale:.2f}
                """)
            st.write("")
            st.write("**💡 How to Fix:**")
            st.write("1. Go to the **sidebar** (left side)")
            st.write("2. Find the **🎯 Detection Parameters** section")
            st.markdown(f"3. Adjust the **Min Scale** slider to: `{recommended_min_scale:.2f}`")
            st.markdown(f"4. Adjust the **Max Scale** slider to: `{recommended_max_scale:.2f}`")
            st.write("5. Click anywhere or rerun detection to apply changes")
    
    status_placeholder.success(f"✅ **Step 3/8 Complete:** Scene edges extracted ({scene_edge_count} edge pixels, downscaled by {scale_factor:.3f}x)")
    progress_bar.progress(4/8)
    
    # Always show template and scene preview
    st.subheader("📸 Image Preview")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.write("**Reference Screw (Template)**")
        # Convert BGR to RGB for display
        reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        st.image(reference_rgb, use_container_width=True)
        st.caption(f"Size: {reference_img.shape[1]}×{reference_img.shape[0]} px")
        st.caption(f"⚠️ **Aspect ratio:** {reference_img.shape[1]/reference_img.shape[0]:.2f} (width/height)")
        if reference_img.shape[0] > reference_img.shape[1] * 2:
            st.warning("⚠️ Template is very tall/narrow. Consider rotating or using a square crop.")
    with preview_col2:
        st.write("**Full Drawing (Scene)**")
        # Show a downscaled version of the full image for preview
        preview_img, _ = resize_with_aspect_ratio(drawing_img, 800)
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, use_container_width=True)
        st.caption(f"Full size: {drawing_img.shape[1]}×{drawing_img.shape[0]} px")
        st.caption(f"Downscaled for detection: {scene_edges.shape[1]}×{scene_edges.shape[0]} px")
    
    # Show debug edges if requested
    if show_debug_edges:
        st.subheader("🔍 Edge Detection Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Reference Edges**")
            st.image(reference_edges, use_container_width=True, clamp=True)
            st.caption(f"Edge pixels: {np.count_nonzero(reference_edges)}")
        with col2:
            st.write("**Scene Edges (Downscaled)**")
            st.image(scene_edges, use_container_width=True, clamp=True)
            st.caption(f"Edge pixels: {np.count_nonzero(scene_edges)}, Scale: {scale_factor:.3f}")
    
    # Apply test mode if enabled
    if use_test_mode:
        status_placeholder.info("🧪 **Test Mode Enabled:** Using very permissive parameters...")
        test_min_votes = 1
        test_min_scale = 0.3
        test_max_scale = 3.0
        test_scale_step = 0.1
        test_canny_low = 20
        test_canny_high = 100
        # Use test parameters
        min_votes = test_min_votes
        min_scale = test_min_scale
        max_scale = test_max_scale
        scale_step = test_scale_step
        canny_low = test_canny_low
        canny_high = test_canny_high
        st.info(f"🧪 Test Mode: min_votes={test_min_votes}, scale={test_min_scale:.1f}-{test_max_scale:.1f}, Canny={test_canny_low}-{test_canny_high}")
    
    # Setup GHT
    status_placeholder.info("⚙️ **Step 4/8: Setting up Generalized Hough Transform (normalizing angles, creating detector, setting parameters)...**")
    progress_bar.progress(4/8)
    
    try:
        # Normalize angles
        min_angle_norm, max_angle_norm = normalize_angle_range(min_angle, max_angle)
        
        # Create GHT object
        ght = createGeneralizedHoughGuil()
        
        # Set parameters BEFORE setTemplate
        # Make MinDist smaller for better detection of close objects
        ght.setMinDist(5)  # Minimum distance between detections (reduced from 10)
        ght.setDp(dp)
        ght.setLevels(levels)
        ght.setMinScale(min_scale)
        ght.setMaxScale(max_scale)
        ght.setScaleStep(scale_step)
        ght.setCannyLowThresh(canny_low)
        ght.setCannyHighThresh(canny_high)
        ght.setAngleThresh(90)  # Angle threshold for voting
        ght.setPosThresh(50)  # Position threshold (reduced from 100 for more lenient matching)
        
        # Compute template center
        template_center = compute_template_center(reference_edges)
        
        # Set template
        try:
            # Prefer API that accepts center
            ght.setTemplate(reference_edges, template_center)
        except Exception:
            # Fallback to older signature
            ght.setTemplate(reference_edges)

        # Try to set angle range if available in this OpenCV build
        try:
            if hasattr(ght, 'setMinAngle') and hasattr(ght, 'setMaxAngle'):
                min_a, max_a = normalize_angle_range(min_angle, max_angle)
                ght.setMinAngle(float(min_a))
                ght.setMaxAngle(float(max_a))
            if hasattr(ght, 'setAngleStep'):
                # Reasonable default angle step for engineering drawings
                ght.setAngleStep(1.0)
        except Exception:
            pass
        
        status_placeholder.success(f"✅ **Step 4/8 Complete:** GHT configured (scale: {min_scale:.2f}-{max_scale:.2f}, angles: {min_angle}°-{max_angle}°, template center: {template_center})")
        progress_bar.progress(5/8)
        
    except Exception as e:
        status_placeholder.error(f"❌ Error setting up GHT: {e}")
        st.exception(e)
        return
    
    # Run detection
    status_placeholder.info("🔍 **Step 5/8: Running Generalized Hough Transform detection (this may take a few seconds)...**")
    progress_bar.progress(5/8)
    
    # Add diagnostic info before detection
    with st.expander("🔍 Pre-detection Diagnostics", expanded=False):
        st.write(f"**Template (Reference):**")
        st.write(f"- Size: {reference_edges.shape[1]}×{reference_edges.shape[0]} px")
        st.write(f"- Edge pixels: {edge_count}")
        st.write(f"- Edge density: {edge_count / (reference_edges.shape[0] * reference_edges.shape[1]):.2%}")
        st.write(f"**Scene (Drawing):**")
        st.write(f"- Full size: {scene_edges_full.shape[1]}×{scene_edges_full.shape[0]} px")
        st.write(f"- Downscaled size: {scene_edges.shape[1]}×{scene_edges.shape[0]} px")
        st.write(f"- Edge pixels: {scene_edge_count}")
        st.write(f"- Edge density: {scene_edge_count / (scene_edges.shape[0] * scene_edges.shape[1]):.2%}")
        st.write(f"**Scale range:** {min_scale:.2f} - {max_scale:.2f} (step: {scale_step:.2f})")
        st.write(f"**Angle range:** {min_angle}° - {max_angle}°")
        st.write(f"**Number of scale steps:** {int((max_scale - min_scale) / scale_step) + 1}")
    
    try:
        # Detect - handle different return formats
        result = ght.detect(scene_edges)
        
        # Check if result is None or empty
        if result is None:
            status_placeholder.warning("🔍 **No detections found by GHT.**")
            
            # Check for potential issues
            aspect_ratio = reference_edges.shape[0] / reference_edges.shape[1]
            is_extreme_aspect = aspect_ratio > 5 or aspect_ratio < 0.2
            
            with st.expander("🔧 Diagnostic Information & Troubleshooting", expanded=True):
                st.write("**Possible reasons:**")
                
                if is_extreme_aspect:
                    st.error(f"⚠️ **CRITICAL:** Template has extreme aspect ratio ({aspect_ratio:.2f}:1). GHT works best with roughly square or moderately rectangular templates.")
                    st.write("**Solution:** Try rotating your reference image 90° or crop it to be more square.")
                
                st.write("1. **Reference and scene don't match well** - Check if the reference screw looks similar to screws in the drawing")
                st.write("2. **Scale range too narrow** - Try widening min/max scale (e.g., 0.5-2.0)")
                st.write("3. **Edge detection too strict** - Lower Canny thresholds or adjust blur")
                st.write("4. **Template too small** - Reference should have at least 40 edge pixels")
                st.write("5. **Scene downscaled too much** - Increase 'Resize Scene Max Width'")
                st.write("6. **Template orientation** - GHT may struggle with very tall/narrow or wide/flat templates")
                st.write("")
                st.write(f"**Current settings:**")
                st.write(f"- Scale range: {min_scale:.2f} - {max_scale:.2f}")
                st.write(f"- Angle range: {min_angle}° - {max_angle}°")
                st.write(f"- Canny thresholds: {canny_low} - {canny_high}")
                st.write(f"- Scene size: {scene_edges.shape[1]}×{scene_edges.shape[0]} px")
                st.write(f"- Reference edges: {edge_count} pixels")
                st.write(f"- Template aspect ratio: {aspect_ratio:.2f}:1 (height/width)")
                
                # Try template matching as fallback
                st.write("")
                st.write("**💡 Alternative: Try Template Matching**")
                st.write("If GHT fails, you can try OpenCV's template matching which is more robust to extreme aspect ratios.")
                
                if st.button("🔄 Try Template Matching (Fallback)", key="try_template_matching"):
                    try_template_matching_fallback(reference_img, drawing_img, reference_edges, scene_edges, 
                                                   min_scale, max_scale, status_placeholder, progress_bar)
                    return
            return
        
        # Parse results - OpenCV may return different formats
        if len(result) == 2:
            positions, votes = result
            scales = None
            angles = None
        elif len(result) == 4:
            positions, votes, scales, angles = result
        else:
            status_placeholder.error(f"Unexpected return format from detect(): {len(result)} values")
            return
        
        # Check if positions is None, empty, or has invalid shape
        if positions is None:
            status_placeholder.info("🔍 No detections found. Try adjusting parameters (lower min_votes, wider scale/angle ranges).")
            return
        
        # Convert to numpy array if needed and check shape
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        if not isinstance(votes, np.ndarray):
            votes = np.array(votes)
        
        # Ensure positions is 2D array (N, 2) for (x, y) coordinates
        # OpenCV may return positions in different formats
        if positions.ndim == 1:
            # If 1D, try to reshape to (N, 2) if length is even
            if len(positions) % 2 == 0:
                positions = positions.reshape(-1, 2)
            else:
                # If odd length, pad or handle differently
                positions = positions.reshape(-1, 1)
        elif positions.ndim == 2:
            # If 2D, ensure it's (N, 2) - take first 2 columns if more
            if positions.shape[1] > 2:
                positions = positions[:, :2]
        elif positions.ndim > 2:
            # Flatten extra dimensions
            positions = positions.reshape(positions.shape[0], -1)
            if positions.shape[1] > 2:
                positions = positions[:, :2]
        
        # Ensure votes is 1D
        if votes.ndim > 1:
            votes = votes.flatten()
        
        # Ensure they have compatible lengths
        if len(positions) != len(votes):
            min_len = min(len(positions), len(votes))
            if min_len == 0:
                status_placeholder.warning("🔍 **No detections found by GHT.**")
                return
            # Trim to same length
            positions = positions[:min_len]
            votes = votes[:min_len]
            if scales is not None and isinstance(scales, np.ndarray) and len(scales) > min_len:
                scales = scales[:min_len]
            if angles is not None and isinstance(angles, np.ndarray) and len(angles) > min_len:
                angles = angles[:min_len]
        
        if positions.size == 0 or len(positions) == 0:
            status_placeholder.warning("🔍 **No detections found by GHT.**")
            with st.expander("🔧 Diagnostic Information & Troubleshooting", expanded=True):
                st.write("**Possible reasons:**")
                st.write("1. **Reference and scene don't match well** - Check if the reference screw looks similar to screws in the drawing")
                st.write("2. **Scale range too narrow** - Try widening min/max scale (e.g., 0.5-2.0)")
                st.write("3. **Edge detection too strict** - Lower Canny thresholds or adjust blur")
                st.write("4. **Template too small** - Reference should have at least 40 edge pixels")
                st.write("5. **Scene downscaled too much** - Increase 'Resize Scene Max Width'")
                st.write("")
                st.write(f"**Current settings:**")
                st.write(f"- Scale range: {min_scale:.2f} - {max_scale:.2f}")
                st.write(f"- Angle range: {min_angle}° - {max_angle}°")
                st.write(f"- Canny thresholds: {canny_low} - {canny_high}")
                st.write(f"- Scene size: {scene_edges.shape[1]}×{scene_edges.shape[0]} px")
                st.write(f"- Reference edges: {edge_count} pixels")
            return
        
        # Get statistics before filtering
        original_votes = votes.copy()  # Save original for diagnostics
        max_vote = float(np.max(votes)) if len(votes) > 0 else 0
        min_vote = float(np.min(votes)) if len(votes) > 0 else 0
        mean_vote = float(np.mean(votes)) if len(votes) > 0 else 0
        original_count = len(positions)
        
        status_placeholder.success(f"✅ **Step 5/8 Complete:** Found {original_count} raw detection(s) (votes: min={min_vote:.0f}, max={max_vote:.0f}, mean={mean_vote:.1f})")
        progress_bar.progress(6/8)
        
        # Filter by minimum votes
        status_placeholder.info(f"🔍 **Step 6/8: Filtering detections (minimum votes: {min_votes}, found max: {max_vote:.0f})...**")
        progress_bar.progress(6/8)
        
        valid_mask = votes >= min_votes
        
        # Ensure valid_mask is 1D and matches the first dimension of positions
        if valid_mask.ndim > 1:
            valid_mask = valid_mask.flatten()
        
        # Ensure valid_mask length matches positions
        if len(valid_mask) != len(positions):
            min_len = min(len(valid_mask), len(positions))
            valid_mask = valid_mask[:min_len]
            positions = positions[:min_len]
            votes = votes[:min_len]
        
        # Apply boolean indexing - positions should be (N, 2) and valid_mask (N,)
        # Note: valid_mask length now matches positions length after the checks above
        positions = positions[valid_mask]
        votes = votes[valid_mask]
        if scales is not None:
            if isinstance(scales, np.ndarray):
                # Ensure scales has same length as original valid_mask
                if len(scales) >= len(valid_mask):
                    scales = scales[valid_mask]
                else:
                    # If scales is shorter, pad with None or skip
                    scales = None
            else:
                scales = np.array(scales)
                if len(scales) >= len(valid_mask):
                    scales = scales[valid_mask]
                else:
                    scales = None
        if angles is not None:
            if isinstance(angles, np.ndarray):
                # Ensure angles has same length as original valid_mask
                if len(angles) >= len(valid_mask):
                    angles = angles[valid_mask]
                else:
                    # If angles is shorter, pad with None or skip
                    angles = None
            else:
                angles = np.array(angles)
                if len(angles) >= len(valid_mask):
                    angles = angles[valid_mask]
                else:
                    angles = None
        
        if len(positions) == 0:
            status_placeholder.warning(f"🔍 **No detections above {min_votes} votes threshold.**")
            with st.expander("🔧 Diagnostic Information & Troubleshooting", expanded=True):
                st.write(f"**Found {original_count} raw detections, but all were below the threshold.**")
                st.write(f"- Maximum vote found: **{max_vote:.0f}**")
                st.write(f"- Minimum vote found: **{min_vote:.0f}**")
                st.write(f"- Mean vote: **{mean_vote:.1f}**")
                st.write(f"- Your threshold: **{min_votes}**")
                st.write("")
                st.write("**Try these adjustments:**")
                st.write(f"1. **Lower minimum votes** to {max(int(max_vote * 0.5), 1)} or lower (currently {min_votes})")
                st.write("2. **Widen scale range** - Your reference might be a different size")
                st.write("3. **Adjust Canny thresholds** - Lower values to detect more edges")
                st.write("4. **Check debug edges** - Enable 'Show Debug Edges' to see if edges are detected correctly")
                st.write("5. **Increase scene resolution** - Try increasing 'Resize Scene Max Width'")
            return
        
        status_placeholder.success(f"✅ **Step 6/8 Complete:** {len(positions)} detection(s) passed vote threshold")
        progress_bar.progress(7/8)
        
        # Apply NMS
        status_placeholder.info(f"🔍 **Step 7/8: Applying Non-Maximum Suppression (NMS distance: {nms_distance} px)...**")
        progress_bar.progress(7/8)
        
        positions, votes, scales, angles = non_max_suppression(
            positions, votes, scales, angles,
            reference_edges.shape, nms_distance
        )
        
        status_placeholder.success(f"✅ **Step 7/8 Complete:** {len(positions)} detection(s) after NMS")
        progress_bar.progress(8/8)
        
        # Map positions back to full resolution
        status_placeholder.info(f"🔄 **Step 8/8: Mapping detections back to full resolution (scale factor: {scale_factor:.3f})...**")
        
        if scale_factor != 1.0:
            positions = positions / scale_factor
        
        status_placeholder.success(f"✅ **Step 8/8 Complete:** All processing finished! Found {len(positions)} final detection(s)!")
        progress_bar.progress(1.0)
        
    except Exception as e:
        status_placeholder.error(f"❌ Error during detection: {e}")
        st.exception(e)
        return
    
    # Visualize results
    st.subheader("📊 Detection Results")
    
    # Add toggle for overlay visualization
    show_template_overlay = st.checkbox("🔍 Show Template Overlay", value=False, 
                                       help="Overlay the reference template at each detection location with detected scale and rotation")
    
    # Create annotated image
    annotated_img = drawing_img.copy()
    overlay_img = drawing_img.copy() if show_template_overlay else None
    
    # Get template dimensions for bounding box
    template_h, template_w = reference_edges.shape
    
    # Prepare template for overlay (convert to BGR if needed)
    if show_template_overlay:
        if len(reference_img.shape) == 3:
            template_for_overlay = reference_img.copy()
        else:
            template_for_overlay = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    
    for i, (pos, vote) in enumerate(zip(positions, votes)):
        x, y = int(pos[0]), int(pos[1])
        
        # Get scale and angle for this detection
        if scales is not None and angles is not None:
            scale = scales[i] if i < len(scales) else 1.0
            angle = angles[i] if i < len(angles) else 0.0
        else:
            scale = 1.0
            angle = 0.0
        
        # Compute bounding box
        bbox_corners = get_template_bbox((template_h, template_w), scale, angle)
        bbox_corners = bbox_corners + np.array([x, y])
        bbox_corners = bbox_corners.astype(np.int32)
        
        # Draw polygon
        cv2.polylines(annotated_img, [bbox_corners], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw center point
        cv2.circle(annotated_img, (x, y), 5, (0, 255, 0), -1)
        
        # Overlay template if requested
        if show_template_overlay:
            try:
                # Calculate template size at detected scale
                overlay_w = int(template_w * scale)
                overlay_h = int(template_h * scale)
                
                if overlay_w > 0 and overlay_h > 0:
                    # Resize template to detected scale
                    template_scaled = cv2.resize(template_for_overlay, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
                    
                    # Rotate template if angle is not 0
                    if abs(angle) > 0.1:
                        center = (overlay_w // 2, overlay_h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        template_scaled = cv2.warpAffine(template_scaled, rotation_matrix, (overlay_w, overlay_h), 
                                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    
                    # Calculate position to place template (centered at detection point)
                    x1 = max(0, x - overlay_w // 2)
                    y1 = max(0, y - overlay_h // 2)
                    x2 = min(overlay_img.shape[1], x1 + overlay_w)
                    y2 = min(overlay_img.shape[0], y1 + overlay_h)
                    
                    # Adjust template if it goes out of bounds
                    if x1 < 0:
                        template_scaled = template_scaled[:, -x1:]
                        x1 = 0
                    if y1 < 0:
                        template_scaled = template_scaled[-y1:, :]
                        y1 = 0
                    if x2 > overlay_img.shape[1]:
                        template_scaled = template_scaled[:, :overlay_img.shape[1] - x1]
                    if y2 > overlay_img.shape[0]:
                        template_scaled = template_scaled[:overlay_img.shape[0] - y1, :]
                    
                    # Blend template overlay with semi-transparency
                    overlay_alpha = 0.6  # Transparency factor
                    if template_scaled.shape[0] > 0 and template_scaled.shape[1] > 0:
                        roi = overlay_img[y1:y2, x1:x2]
                        if roi.shape == template_scaled.shape:
                            overlay_img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - overlay_alpha, template_scaled, overlay_alpha, 0)
            except Exception as e:
                # If overlay fails for this detection, skip it
                pass
        
        # Label
        label = f"#{i+1}: v={vote:.0f}"
        if angles is not None:
            label += f", θ={angle:.1f}°"
        if scales is not None:
            label += f", s={scale:.2f}"
        
        # Position label above detection
        label_y = max(y - 10, 15)
        cv2.putText(annotated_img, label, (x - 30, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if show_template_overlay:
            cv2.putText(overlay_img, label, (x - 30, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(overlay_img, (x, y), 5, (0, 255, 0), -1)
            cv2.polylines(overlay_img, [bbox_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Display annotated image with detection analysis
    if len(positions) > 0:
        st.write(f"**Found {len(positions)} detection(s).**")
        
        # Show detection locations
        detection_info_col1, detection_info_col2 = st.columns(2)
        with detection_info_col1:
            st.write("**Detection Locations:**")
            for i, (pos, vote) in enumerate(zip(positions, votes)):
                x, y = int(pos[0]), int(pos[1])
                scale_str = f", scale={scales[i]:.2f}" if scales is not None and i < len(scales) else ""
                angle_str = f", angle={angles[i]:.1f}°" if angles is not None and i < len(angles) else ""
                st.write(f"- Detection #{i+1}: Position ({x}, {y}), votes={vote:.0f}{scale_str}{angle_str}")
        
        with detection_info_col2:
            if len(positions) == 1:
                st.warning("⚠️ **Only 1 detection found.**")
                st.write("**Possible issues:**")
                st.write("- Detection might be in wrong location")
                st.write("- Try lowering 'Minimum Votes' threshold")
                st.write("- Widen scale range")
                st.write("- Adjust Canny edge thresholds")
                st.write("- Check if template matches the actual screw")
    
    if show_template_overlay:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Bounding Boxes Only**")
            st.image(annotated_img, use_container_width=True, channels="BGR")
        with col2:
            st.write("**With Template Overlay**")
            st.image(overlay_img, use_container_width=True, channels="BGR")
    else:
        st.image(annotated_img, use_container_width=True, channels="BGR")
    
    # Detection summary
    with st.expander("📋 Detection Summary", expanded=False):
        st.write(f"**Total Detections:** {len(positions)}")
        st.write(f"**Scale Factor Applied:** {scale_factor:.3f}")
        st.write(f"**Template Size:** {template_w}×{template_h} px")
        
        if len(positions) > 0:
            st.write("\n**Detections:**")
            for i, (pos, vote) in enumerate(zip(positions, votes)):
                x, y = int(pos[0]), int(pos[1])
                scale_str = f", scale={scales[i]:.2f}" if scales is not None else ""
                angle_str = f", angle={angles[i]:.1f}°" if angles is not None else ""
                st.write(f"- Detection #{i+1}: ({x}, {y}), votes={vote:.0f}{scale_str}{angle_str}")
    
    # Download button
    st.divider()
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Bounding Boxes (PNG)",
            data=buf,
            file_name="detected_screws_bboxes.png",
            mime="image/png"
        )
    
    if show_template_overlay and overlay_img is not None:
        with download_col2:
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
            buf_overlay = io.BytesIO()
            overlay_pil.save(buf_overlay, format="PNG")
            buf_overlay.seek(0)
            
            st.download_button(
                label="📥 Download With Template Overlay (PNG)",
                data=buf_overlay,
                file_name="detected_screws_overlay.png",
                mime="image/png"
            )
    
    # Optional future extension (commented)
    st.divider()
    with st.expander("🔮 Future Extension: High-Res Refinement", expanded=False):
        st.code("""
# Optional: Refine detections on high-resolution crops
# 1. Extract crop around each detection at full resolution
# 2. Run FFT-based Normalized Cross-Correlation (NCC) or feature matching
# 3. Reject partial matches or low-confidence detections
# 4. This can improve precision but adds computation time
        """, language="python")
# ------------------------------
# Fixed-mode implementation
# ------------------------------
def run_fixed_mode():
    st.title("🔩 Screw Finder — Generalized Hough (Raster Detector)")
    st.caption("Fixed Mode: uses hardcoded settings and local files for faster testing")

    # Left pane: settings details; Right pane: Process button and results
    left, right = st.columns([1, 2])

    with left:
        st.subheader("📋 Fixed Settings (Hardcoded)")
        st.markdown(f"• **Scene file:** `{str(FIXED_SCENE_PATH)}`")
        st.markdown(f"• **Template file:** `{str(FIXED_TEMPLATE_PATH)}`")

        st.markdown("**Preprocessing**")
        st.write(f"- Gaussian Blur: {FIXED_GAUSSIAN_BLUR}")
        st.write(f"- Canny: {FIXED_CANNY_LOW} - {FIXED_CANNY_HIGH}")
        st.write(f"- Binarize: {FIXED_BINARIZE}")

        st.markdown("**Speed Control**")
        st.write(f"- Resize Scene Max Width: {FIXED_MAX_WIDTH}px")

        st.markdown("**Detection Parameters (GHT - Guil)**")
        st.write(f"- Min/Max Scale: {FIXED_MIN_SCALE:.2f} - {FIXED_MAX_SCALE:.2f}")
        st.write(f"- Scale Step: {FIXED_SCALE_STEP:.2f}")
        st.write(f"- Angle Range: {FIXED_MIN_ANGLE}° - {FIXED_MAX_ANGLE}°")
        st.write(f"- Accumulator dp: {FIXED_DP}")
        st.write(f"- Pyramid Levels: {FIXED_LEVELS}")
        st.write(f"- Minimum Votes: {FIXED_MIN_VOTES}")
        st.write(f"- NMS Distance: {FIXED_NMS_DISTANCE}px")
        st.write(f"- Overlay Template: {FIXED_SHOW_OVERLAY}")

        st.markdown("**Notes**")
        st.write("- These settings target vertical screws in engineering drawings.")
        st.write("- Angle is constrained around vertical orientation for speed/precision.")

    with right:
        if st.button("▶️ Process", type="primary"):
            # Validate files
            if not FIXED_SCENE_PATH.exists():
                st.error(f"Scene file not found: {FIXED_SCENE_PATH}")
                return
            if not FIXED_TEMPLATE_PATH.exists():
                st.error(f"Template file not found: {FIXED_TEMPLATE_PATH}")
                return

            # Load images (BGR)
            drawing_img = cv2.imread(str(FIXED_SCENE_PATH), cv2.IMREAD_COLOR)
            reference_img = cv2.imread(str(FIXED_TEMPLATE_PATH), cv2.IMREAD_COLOR)
            if drawing_img is None or reference_img is None:
                st.error("Failed to read scene or template. Ensure files are valid images.")
                return

            # Step 1: reference edges
            st.write("Step 1/5 — Processing reference...")
            reference_edges = process_image_for_edges(
                reference_img,
                FIXED_GAUSSIAN_BLUR,
                FIXED_CANNY_LOW,
                FIXED_CANNY_HIGH,
                FIXED_BINARIZE,
            )
            ref_edge_count = int(np.count_nonzero(reference_edges))
            st.write(f"- Reference edges: {ref_edge_count} px")
            if ref_edge_count < 50:
                st.warning("Reference has very few edges. Consider cropping tighter or increasing contrast.")

            # Step 2: scene edges (with downscale)
            st.write("Step 2/5 — Processing scene...")
            scene_edges_full = process_image_for_edges(
                drawing_img,
                FIXED_GAUSSIAN_BLUR,
                FIXED_CANNY_LOW,
                FIXED_CANNY_HIGH,
                FIXED_BINARIZE,
            )
            scene_edges, scale_factor = resize_with_aspect_ratio(scene_edges_full, FIXED_MAX_WIDTH)
            st.write(f"- Scene size (full): {drawing_img.shape[1]}×{drawing_img.shape[0]}")
            st.write(f"- Scene edges (downscaled): {scene_edges.shape[1]}×{scene_edges.shape[0]} (scale={scale_factor:.3f})")

            # Step 3: setup GHT (Guil)
            st.write("Step 3/5 — Setting up GHT (Guil)...")
            try:
                ght = createGeneralizedHoughGuil()
                ght.setMinDist(5)
                ght.setDp(FIXED_DP)
                ght.setLevels(max(1, int(FIXED_LEVELS)))
                ght.setMinScale(float(FIXED_MIN_SCALE))
                ght.setMaxScale(float(FIXED_MAX_SCALE))
                ght.setScaleStep(float(FIXED_SCALE_STEP))
                ght.setCannyLowThresh(int(FIXED_CANNY_LOW))
                ght.setCannyHighThresh(int(FIXED_CANNY_HIGH))
                ght.setAngleThresh(90)
                ght.setPosThresh(50)

                template_center = compute_template_center(reference_edges)
                try:
                    ght.setTemplate(reference_edges, template_center)
                except Exception:
                    ght.setTemplate(reference_edges)

                # Angle window (normalize to [0,360])
                try:
                    if hasattr(ght, "setMinAngle") and hasattr(ght, "setMaxAngle"):
                        a_min, a_max = normalize_angle_range(FIXED_MIN_ANGLE, FIXED_MAX_ANGLE)
                        ght.setMinAngle(float(a_min))
                        ght.setMaxAngle(float(a_max))
                    if hasattr(ght, "setAngleStep"):
                        ght.setAngleStep(1.0)
                except Exception:
                    pass

                st.write(f"- Template center: {template_center}")
                st.write(f"- Scale: {FIXED_MIN_SCALE:.2f}–{FIXED_MAX_SCALE:.2f}, step={FIXED_SCALE_STEP:.2f}")
                st.write(f"- Angles: {FIXED_MIN_ANGLE}°–{FIXED_MAX_ANGLE}°, dp={FIXED_DP}, levels={FIXED_LEVELS}")
            except Exception as e:
                st.error(f"GHT setup error: {e}")
                st.stop()

            # Step 4: detect
            st.write("Step 4/5 — Detecting...")
            try:
                # Helper to run one GHT pass with specified scale range
                def run_ght_pass(min_s: float, max_s: float, step_s: float):
                    g = createGeneralizedHoughGuil()
                    g.setMinDist(5)
                    g.setDp(FIXED_DP)
                    g.setLevels(max(1, int(FIXED_LEVELS)))
                    g.setMinScale(float(min_s))
                    g.setMaxScale(float(max_s))
                    g.setScaleStep(float(step_s))
                    g.setCannyLowThresh(int(FIXED_CANNY_LOW))
                    g.setCannyHighThresh(int(FIXED_CANNY_HIGH))
                    g.setAngleThresh(90)
                    g.setPosThresh(50)
                    try:
                        if hasattr(g, "setMinAngle") and hasattr(g, "setMaxAngle"):
                            a_min, a_max = normalize_angle_range(FIXED_MIN_ANGLE, FIXED_MAX_ANGLE)
                            g.setMinAngle(float(a_min))
                            g.setMaxAngle(float(a_max))
                        if hasattr(g, "setAngleStep"):
                            g.setAngleStep(1.0)
                    except Exception:
                        pass
                    tc = compute_template_center(reference_edges)
                    try:
                        g.setTemplate(reference_edges, tc)
                    except Exception:
                        g.setTemplate(reference_edges)
                    res = g.detect(scene_edges)
                    if res is None:
                        return None, None, None, None
                    if len(res) == 2:
                        return res[0], res[1], None, None
                    if len(res) == 4:
                        return res
                    return None, None, None, None

                # Try multiple scale bands (coarse-to-fine) to handle template too big/small
                scale_passes = [
                    (FIXED_MIN_SCALE, FIXED_MAX_SCALE, FIXED_SCALE_STEP),  # suggested
                    (0.50, 1.00, 0.03),
                    (0.30, 0.70, 0.03),
                    (0.20, 0.50, 0.05),
                ]
                best = None
                best_vote = -1.0
                for (ms, xs, ss) in scale_passes:
                    p, v, s, a = run_ght_pass(ms, xs, ss)
                    if p is None or v is None:
                        continue
                    # Normalize v to 1D for scoring
                    v_arr = np.asarray(v).reshape(-1)
                    if v_arr.size == 0:
                        continue
                    top = float(np.max(v_arr))
                    if top > best_vote:
                        best_vote = top
                        best = (p, v, s, a, ms, xs, ss)
                if best is None:
                    st.info("No raw detections in any scale pass. Try widening ranges or rotating the template 90°.")
                    return
                positions, votes, scales, angles, used_min_s, used_max_s, used_step_s = best
                st.write(f"- Best scale pass: {used_min_s:.2f}–{used_max_s:.2f} (step {used_step_s:.2f}), top vote={best_vote:.0f}")

                if positions is None or len(positions) == 0:
                    st.info("No raw detections. Try widening scale/angle ranges or lowering min_votes.")
                    return

                # Ensure types/shapes
                if not isinstance(positions, np.ndarray):
                    positions = np.array(positions)
                if not isinstance(votes, np.ndarray):
                    votes = np.array(votes)
                # Normalize positions to shape (N, 2)
                if positions.ndim == 3 and positions.shape[-1] == 2:
                    positions = positions.reshape(-1, 2)
                elif positions.ndim == 2 and positions.shape[1] > 2:
                    positions = positions[:, :2]
                elif positions.ndim == 1:
                    if len(positions) % 2 == 0:
                        positions = positions.reshape(-1, 2)
                    else:
                        positions = positions.reshape(-1, 1)
                # Normalize votes to shape (N,)
                if votes.ndim > 1:
                    votes = votes.flatten()

                # Harmonize lengths before masking
                n_pos = len(positions)
                n_votes = len(votes)
                if n_pos != n_votes:
                    min_len = min(n_pos, n_votes)
                    positions = positions[:min_len]
                    votes = votes[:min_len]
                    if scales is not None:
                        scales = np.array(scales)[:min_len]
                    if angles is not None:
                        angles = np.array(angles)[:min_len]

                # Filter votes
                valid_mask = (votes >= FIXED_MIN_VOTES)
                if valid_mask.ndim > 1:
                    valid_mask = valid_mask.flatten()
                # Ensure mask length matches positions length
                if len(valid_mask) != len(positions):
                    min_len = min(len(valid_mask), len(positions))
                    valid_mask = valid_mask[:min_len]
                    positions = positions[:min_len]
                    votes = votes[:min_len]
                positions = positions[valid_mask]
                votes = votes[valid_mask]
                if scales is not None:
                    scales = np.array(scales)
                    if len(scales) != len(valid_mask):
                        scales = scales[:len(valid_mask)]
                    scales = scales[valid_mask]
                if angles is not None:
                    angles = np.array(angles)
                    if len(angles) != len(valid_mask):
                        angles = angles[:len(valid_mask)]
                    angles = angles[valid_mask]

                if len(positions) == 0:
                    st.info(f"No detections above min_votes ({FIXED_MIN_VOTES}). Lower it or widen scale range.")
                    return

                # NMS
                positions, votes, scales, angles = non_max_suppression(
                    positions, votes, scales, angles, reference_edges.shape, FIXED_NMS_DISTANCE
                )

                # Map back to full res
                if scale_factor != 1.0 and len(positions) > 0:
                    positions = positions / scale_factor

                st.success(f"Found {len(positions)} detection(s) after NMS.")
            except Exception as e:
                st.error(f"Detection error: {e}")
                st.stop()

            # Step 5: visualize
            st.write("Step 5/5 — Visualizing results...")
            annotated_img = drawing_img.copy()
            overlay_img = drawing_img.copy() if FIXED_SHOW_OVERLAY else None

            template_h, template_w = reference_edges.shape
            for i, (pos, vote) in enumerate(zip(positions, votes)):
                # Robustly extract scalar x,y even if pos has nested shape
                pos_flat = np.asarray(pos).astype(float).ravel()
                if pos_flat.size < 2:
                    # Skip malformed entry
                    continue
                x, y = int(round(pos_flat[0])), int(round(pos_flat[1]))
                # Robustly extract scale/angle
                scale = 1.0
                angle = 0.0
                if scales is not None and i < len(scales):
                    s_val = np.asarray(scales[i]).astype(float).ravel()
                    if s_val.size >= 1:
                        scale = float(s_val[0])
                if angles is not None and i < len(angles):
                    a_val = np.asarray(angles[i]).astype(float).ravel()
                    if a_val.size >= 1:
                        angle = float(a_val[0])

                bbox_corners = get_template_bbox((template_h, template_w), scale, angle)
                bbox_corners = bbox_corners + np.array([x, y])
                bbox_corners = bbox_corners.astype(np.int32)

                cv2.polylines(annotated_img, [bbox_corners], True, (0, 255, 0), 2)
                cv2.circle(annotated_img, (x, y), 5, (0, 255, 0), -1)

                # Overlay of template (scaled/rotated)
                if FIXED_SHOW_OVERLAY:
                    try:
                        overlay_w = int(template_w * scale)
                        overlay_h = int(template_h * scale)
                        if overlay_w > 0 and overlay_h > 0:
                            # Use original color template for overlay
                            template_color = reference_img if reference_img.ndim == 3 else cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
                            template_scaled = cv2.resize(template_color, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
                            if abs(angle) > 0.1:
                                center = (overlay_w // 2, overlay_h // 2)
                                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                                template_scaled = cv2.warpAffine(template_scaled, rot_mat, (overlay_w, overlay_h),
                                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                            x1 = max(0, x - overlay_w // 2)
                            y1 = max(0, y - overlay_h // 2)
                            x2 = min(overlay_img.shape[1], x1 + overlay_w)
                            y2 = min(overlay_img.shape[0], y1 + overlay_h)
                            if x2 > x1 and y2 > y1:
                                ts = template_scaled[: y2 - y1, : x2 - x1]
                                roi = overlay_img[y1:y2, x1:x2]
                                if roi.shape == ts.shape:
                                    overlay_img[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.4, ts, 0.6, 0)
                    except Exception:
                        pass

                label = f"#{i+1}: v={int(vote)}"
                if angles is not None:
                    label += f", θ={angle:.1f}°"
                if scales is not None:
                    label += f", s={scale:.2f}"
                cv2.putText(annotated_img, label, (x - 30, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show
            if FIXED_SHOW_OVERLAY and overlay_img is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Bounding Boxes Only")
                    st.image(annotated_img, use_container_width=True, channels="BGR")
                with c2:
                    st.write("With Template Overlay")
                    st.image(overlay_img, use_container_width=True, channels="BGR")
            else:
                st.image(annotated_img, use_container_width=True, channels="BGR")

            # Downloads
            st.divider()
            dl1, dl2 = st.columns(2)
            with dl1:
                buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
                buf.seek(0)
                st.download_button("📥 Download Bounding Boxes (PNG)", buf, "detected_screws_bboxes.png", "image/png")
            if FIXED_SHOW_OVERLAY and overlay_img is not None:
                with dl2:
                    buf2 = io.BytesIO()
                    Image.fromarray(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)).save(buf2, format="PNG")
                    buf2.seek(0)
                    st.download_button("📥 Download With Template Overlay (PNG)", buf2, "detected_screws_overlay.png", "image/png")


if __name__ == "__main__":
    main()

