# ============================================================
# modules/entry_point_detector.py
# Vehicle Entry Point Detector  v1.0
# ALKF Master Land Plan API
#
# Detects vehicle access points (X, Y, Z …) from a lease plan
# image by identifying gaps in the green landscaping / verge
# strip that borders the site boundary.
#
# Pipeline:
#   1. Decode image bytes → OpenCV BGR array (reuses _decode_image)
#   2. Build a binary mask of the green verge strip (HSV segmentation)
#   3. Build a binary mask of the pink site area
#   4. Merge both masks → full site footprint; extract outer contour
#   5. Walk the contour; find contiguous runs of boundary points that
#      have NO green beneath them = access gaps
#   6. Filter gaps by plausible entry-width range
#   7. Within each gap assign sequential labels (X, Y, Z, …)
#      splitting evenly along the gap arc
#   8. Convert pixel coordinates → EPSG:3857 via _pixel_to_geo()
#   9. Return structured dict matching the site-intelligence schema
#
# Label assignment convention:
#   - Labels are assigned in contour-walk order (counter-clockwise
#     by default from OpenCV RETR_EXTERNAL).
#   - First gap → first labels (X, Y, Z if 3 sub-points)
#   - Additional gaps → continue alphabet (A, B, C … or E, F …)
#   - The caller may override label names via `label_names`.
# ============================================================

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from modules.lease_plan_parser import _decode_image, _pixel_to_geo

log = logging.getLogger(__name__)

# ── HSV ranges (same image calibration as lease_plan_parser) ─
_GREEN_LOWER = np.array([25,  50, 140], dtype=np.uint8)
_GREEN_UPPER = np.array([45, 160, 220], dtype=np.uint8)

_PINK_LOWER  = np.array([ 0,  15, 180], dtype=np.uint8)
_PINK_UPPER  = np.array([15, 100, 255], dtype=np.uint8)

# Morphology kernels
_KERNEL_FILL_GREEN = cv2.getStructuringElement(cv2.MORPH_RECT, (8,  8))
_KERNEL_FILL_SITE  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

# Gap size (in boundary contour points) considered a plausible entry
_GAP_MIN_PTS = 3
_GAP_MAX_PTS = 100

# How many pixels inward to scan for green when testing a boundary point
_GREEN_PROBE_RADIUS = 12

# Minimum green pixels in the probe patch to consider "green present"
_GREEN_PRESENT_THRESHOLD = 10

# Default label sequence (X Y Z for first gap, then A B C D … for extras)
_DEFAULT_LABEL_SEQ = list("XYZABCDEFGHIJKLMNOPQRSTUVW")


def _build_site_contour(hsv: np.ndarray) -> Optional[np.ndarray]:
    """
    Combine green verge + pink site area into a single filled mask and
    return the largest outer contour as an (N,2) array of pixel coords.
    Returns None if no plausible site contour is found.
    """
    green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    green_filled = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, _KERNEL_FILL_GREEN)

    pink_mask = cv2.inRange(hsv, _PINK_LOWER, _PINK_UPPER)

    site_mask   = cv2.bitwise_or(pink_mask, green_filled)
    site_filled = cv2.morphologyEx(site_mask, cv2.MORPH_CLOSE, _KERNEL_FILL_SITE)

    contours, _ = cv2.findContours(site_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main) < 1000:
        log.warning("[entry_detector] Site contour too small — image may not contain a valid lease plan")
        return None

    return main.reshape(-1, 2), green_filled


def _find_gaps(boundary_pts: np.ndarray, green_filled: np.ndarray,
               img_shape: tuple) -> list[dict]:
    """
    Walk the site boundary contour.  For each point, probe the surrounding
    pixel patch for green.  Runs of boundary points with no green = entry gaps.

    Returns list of gap dicts:
        { start_idx, end_idx, length, boundary_pts_in_gap }
    """
    h_img, w_img = img_shape[:2]
    gaps: list[dict] = []
    gap_start: Optional[int] = None
    in_gap = False

    for i, (x, y) in enumerate(boundary_pts):
        y1 = max(0, y - _GREEN_PROBE_RADIUS)
        y2 = min(h_img, y + _GREEN_PROBE_RADIUS)
        x1 = max(0, x - _GREEN_PROBE_RADIUS)
        x2 = min(w_img, x + _GREEN_PROBE_RADIUS)
        patch = green_filled[y1:y2, x1:x2]
        has_green = int(np.sum(patch > 0)) >= _GREEN_PRESENT_THRESHOLD

        if not has_green and not in_gap:
            gap_start = i
            in_gap = True
        elif has_green and in_gap:
            gap_end = i
            gap_len = gap_end - gap_start
            if _GAP_MIN_PTS <= gap_len <= _GAP_MAX_PTS:
                gaps.append({
                    "start_idx": gap_start,
                    "end_idx":   gap_end,
                    "length":    gap_len,
                    "pts":       boundary_pts[gap_start:gap_end],
                })
            in_gap = False

    # Handle gap that wraps the end of the contour
    if in_gap and gap_start is not None:
        gap_end = len(boundary_pts)
        gap_len = gap_end - gap_start
        if _GAP_MIN_PTS <= gap_len <= _GAP_MAX_PTS:
            gaps.append({
                "start_idx": gap_start,
                "end_idx":   gap_end,
                "length":    gap_len,
                "pts":       boundary_pts[gap_start:gap_end],
            })

    return gaps


def _assign_labels(gaps: list[dict],
                   points_per_gap: int,
                   label_names: Optional[list[str]]) -> list[dict]:
    """
    For each gap, evenly subdivide into `points_per_gap` sub-points
    and assign sequential label names.

    Returns list of entry-point dicts:
        { label, pixel_x, pixel_y }
    """
    seq = label_names if label_names else _DEFAULT_LABEL_SEQ
    label_idx = 0
    entries: list[dict] = []

    for gap in gaps:
        pts = gap["pts"]
        n   = len(pts)

        # Evenly space `points_per_gap` samples along the gap
        if points_per_gap == 1 or n < points_per_gap:
            sample_indices = [n // 2]
        else:
            step = (n - 1) / (points_per_gap - 1)
            sample_indices = [int(round(i * step)) for i in range(points_per_gap)]

        for si in sample_indices:
            px, py = pts[min(si, n - 1)].tolist()
            lbl = seq[label_idx] if label_idx < len(seq) else f"E{label_idx}"
            entries.append({"label": lbl, "pixel_x": px, "pixel_y": py})
            label_idx += 1

    return entries


def extract_entry_points(
    image_bytes:      bytes,
    site_polygon:     Polygon,
    crs:              str = "EPSG:3857",
    points_per_gap:   int = 3,
    label_names:      Optional[list[str]] = None,
) -> dict:
    """
    Detect vehicle entry/exit points from a lease plan image.

    Parameters
    ----------
    image_bytes     : raw bytes of the lease plan (PNG / JPEG / PDF)
    site_polygon    : Shapely Polygon of the site boundary in `crs`
    crs             : coordinate reference system (default EPSG:3857)
    points_per_gap  : number of labelled sub-points per entry gap.
                      3 → assigns X, Y, Z within a single wide entry gap.
                      1 → assigns a single midpoint per gap.
    label_names     : override the default X/Y/Z/A/B/C … label sequence.

    Returns
    -------
    dict with keys:
        crs          : str
        entry_points : list of {
            label    : str          (X / Y / Z / A …)
            pixel_x  : int          (image pixel column)
            pixel_y  : int          (image pixel row)
            geo_x    : float        (EPSG:3857 easting, metres)
            geo_y    : float        (EPSG:3857 northing, metres)
        }
        gap_count    : int           number of distinct access gaps found
        gaps         : list of {
            gap_index  : int
            length_pts : int
            labels     : list[str]
        }
    """
    log.info("[entry_detector] START")

    # ── 1. Decode image ──────────────────────────────────────
    img_bgr = _decode_image(image_bytes)
    img_h, img_w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    log.info(f"  Image: {img_w}×{img_h}px")

    # ── 2. Site bounding box in EPSG:3857 ────────────────────
    site_bounds = site_polygon.bounds   # (minx, miny, maxx, maxy)
    log.info(
        f"  Site bounds: "
        f"x=[{site_bounds[0]:.0f},{site_bounds[2]:.0f}]  "
        f"y=[{site_bounds[1]:.0f},{site_bounds[3]:.0f}]"
    )

    # ── 3. Build site contour ────────────────────────────────
    result = _build_site_contour(img_hsv)
    if result is None:
        log.warning("[entry_detector] Could not extract site contour")
        return {"crs": crs, "entry_points": [], "gap_count": 0, "gaps": []}

    boundary_pts, green_filled = result
    log.info(f"  Site contour: {len(boundary_pts)} boundary points")

    # ── 4. Find access gaps ──────────────────────────────────
    gaps = _find_gaps(boundary_pts, green_filled, img_bgr.shape)
    log.info(f"  Gaps found: {len(gaps)}")
    for i, g in enumerate(gaps):
        log.info(f"    gap {i}: length={g['length']} boundary pts")

    if not gaps:
        log.warning("[entry_detector] No entry gaps detected")
        return {"crs": crs, "entry_points": [], "gap_count": 0, "gaps": []}

    # ── 5. Assign labels ─────────────────────────────────────
    raw_entries = _assign_labels(gaps, points_per_gap, label_names)

    # ── 6. Convert pixel → geo ───────────────────────────────
    entry_points = []
    for e in raw_entries:
        geo_x, geo_y = _pixel_to_geo(
            float(e["pixel_x"]), float(e["pixel_y"]),
            img_w, img_h, site_bounds
        )
        entry_points.append({
            "label":   e["label"],
            "pixel_x": e["pixel_x"],
            "pixel_y": e["pixel_y"],
            "geo_x":   geo_x,
            "geo_y":   geo_y,
        })
        log.info(
            f"  Entry {e['label']}: pixel=({e['pixel_x']},{e['pixel_y']})  "
            f"geo=({geo_x:.1f},{geo_y:.1f})"
        )

    # ── 7. Gap summary ───────────────────────────────────────
    label_idx = 0
    gap_summary = []
    seq = label_names if label_names else _DEFAULT_LABEL_SEQ
    for i, g in enumerate(gaps):
        n_labels = min(points_per_gap, g["length"])
        labels_in_gap = seq[label_idx: label_idx + n_labels]
        gap_summary.append({
            "gap_index":  i,
            "length_pts": g["length"],
            "labels":     list(labels_in_gap),
        })
        label_idx += n_labels

    log.info(f"[entry_detector] DONE  entry_points={len(entry_points)}")
    return {
        "crs":          crs,
        "entry_points": entry_points,
        "gap_count":    len(gaps),
        "gaps":         gap_summary,
    }
