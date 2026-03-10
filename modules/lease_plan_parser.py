# ============================================================
# modules/lease_plan_parser.py
# Lease Plan Colour Segmentation Engine  v1.0
# ALKF Master Land Plan API
#
# Extracts non-building zone polygons from a lease plan
# (PDF or image) using OpenCV colour segmentation.
#
# Pipeline:
#   1. Decode PDF/image bytes → OpenCV BGR image
#   2. For each non-building colour label:
#      a. Define HSV colour range from label name
#      b. Build binary mask
#      c. Extract contours
#      d. Convert pixel polygon → EPSG:3857 coordinates
#         using the site boundary as spatial reference
#   3. Return structured dict matching the output spec
# ============================================================

from __future__ import annotations

import io
import logging
import re
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

log = logging.getLogger(__name__)

# ── Colour → HSV range lookup ─────────────────────────────────
# HSV in OpenCV:  H ∈ [0,179]  S ∈ [0,255]  V ∈ [0,255]
_COLOUR_HSV: dict[str, dict] = {
    "pink": {
        "lower": np.array([140, 30, 150], dtype=np.uint8),
        "upper": np.array([175, 160, 255], dtype=np.uint8),
    },
    "green": {
        "lower": np.array([35, 40, 40], dtype=np.uint8),
        "upper": np.array([90, 255, 255], dtype=np.uint8),
    },
    "blue": {
        "lower": np.array([90, 50, 50], dtype=np.uint8),
        "upper": np.array([130, 255, 255], dtype=np.uint8),
    },
    "yellow": {
        "lower": np.array([20, 80, 80], dtype=np.uint8),
        "upper": np.array([35, 255, 255], dtype=np.uint8),
    },
    "red": {
        # Red wraps around H=0 in OpenCV
        "lower":  np.array([0,   50, 50], dtype=np.uint8),
        "upper":  np.array([10, 255, 255], dtype=np.uint8),
        "lower2": np.array([165, 50, 50], dtype=np.uint8),
        "upper2": np.array([179, 255, 255], dtype=np.uint8),
    },
    "orange": {
        "lower": np.array([10, 80, 80], dtype=np.uint8),
        "upper": np.array([20, 255, 255], dtype=np.uint8),
    },
    "purple": {
        "lower": np.array([125, 30, 50], dtype=np.uint8),
        "upper": np.array([145, 255, 255], dtype=np.uint8),
    },
    "grey": {
        "lower": np.array([0, 0, 80], dtype=np.uint8),
        "upper": np.array([179, 40, 200], dtype=np.uint8),
    },
    "white": {
        "lower": np.array([0, 0, 200], dtype=np.uint8),
        "upper": np.array([179, 30, 255], dtype=np.uint8),
    },
    "black": {
        "lower": np.array([0, 0, 0], dtype=np.uint8),
        "upper": np.array([179, 255, 50], dtype=np.uint8),
    },
}

# Minimum contour area in pixels to be considered a real zone (noise filter)
_MIN_CONTOUR_AREA_PX = 200


def _normalise_colour_key(raw: str) -> str:
    """'pink cross-hatched black'  →  'pink_cross_hatched_black'"""
    return re.sub(r"[\s\-]+", "_", raw.strip().lower())


def _extract_base_colour(normalised_key: str) -> str:
    """
    Extract the primary colour name from a composite key.
    'pink_cross_hatched_black' → 'pink'
    'yellow_diagonal_stripes'  → 'yellow'
    """
    parts = normalised_key.split("_")
    for part in parts:
        if part in _COLOUR_HSV:
            return part
    # fallback: return the first token
    return parts[0] if parts else "grey"


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes (JPEG, PNG, TIFF, or PDF first-page raster)
    into an OpenCV BGR uint8 image array.

    For PDF input, requires pdf2image + poppler. Falls back to
    treating the bytes as a direct image if PDF decoding fails.
    """
    # Attempt direct image decode (PNG / JPEG / TIFF / BMP)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is not None:
        log.info(f"  LeaseParser: image decoded  shape={img.shape}")
        return img

    # Attempt PDF rasterisation via pdf2image
    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(image_bytes, dpi=150, first_page=1, last_page=1)
        if pages:
            pil_img = pages[0]
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            log.info(f"  LeaseParser: PDF rasterised  shape={img.shape}")
            return img
    except Exception as e:
        log.warning(f"  LeaseParser: PDF rasterisation failed: {e}")

    raise ValueError(
        "Unable to decode lease plan. "
        "Provide a PNG/JPEG image or install pdf2image + poppler for PDF support."
    )


def _build_colour_mask(hsv_img: np.ndarray, colour_key: str) -> np.ndarray:
    """
    Build a binary mask for the given colour key.
    Returns a uint8 binary mask (255 = match, 0 = no match).
    """
    base = _extract_base_colour(colour_key)
    if base not in _COLOUR_HSV:
        log.warning(f"  LeaseParser: no HSV range for colour '{base}' — empty mask")
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    spec = _COLOUR_HSV[base]
    mask = cv2.inRange(hsv_img, spec["lower"], spec["upper"])

    # Red wraps around H=0 → merge two ranges
    if "lower2" in spec:
        mask2 = cv2.inRange(hsv_img, spec["lower2"], spec["upper2"])
        mask  = cv2.bitwise_or(mask, mask2)

    # Handle cross-hatched / pattern colours: AND with the secondary colour
    # e.g. 'pink_cross_hatched_black' — we keep everything that is PINK
    # (the black hatching reduces the effective area but does not exclude the zone).
    # The mask therefore only targets the primary colour declared.

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return mask


def _pixel_to_geo(
    px: float,
    py: float,
    img_w: int,
    img_h: int,
    site_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Convert pixel coordinates (px, py) in an image of size (img_w × img_h)
    to EPSG:3857 coordinates using the site boundary bounding box as
    the spatial reference frame.

    The image is assumed to be a north-up plan where:
      pixel (0, 0)           → (minx, maxy)
      pixel (img_w, img_h)   → (maxx, miny)

    Parameters
    ----------
    px, py          : pixel coordinates (origin = top-left)
    img_w, img_h    : image dimensions in pixels
    site_bounds     : (minx, miny, maxx, maxy) in EPSG:3857

    Returns
    -------
    (geo_x, geo_y)  : EPSG:3857 coordinates
    """
    minx, miny, maxx, maxy = site_bounds
    geo_w = maxx - minx
    geo_h = maxy - miny

    geo_x = minx + (px / img_w) * geo_w
    geo_y = maxy - (py / img_h) * geo_h   # y-axis is inverted

    return round(geo_x, 4), round(geo_y, 4)


def _extract_contour_coordinates(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    site_bounds: tuple[float, float, float, float],
    min_area_px: int = _MIN_CONTOUR_AREA_PX,
) -> tuple[list[float], list[float]]:
    """
    Extract the largest contour from the mask and convert to geo coordinates.
    Returns (xs, ys) in EPSG:3857.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], []

    # Filter by minimum area and select the largest
    valid = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not valid:
        return [], []

    largest = max(valid, key=cv2.contourArea)

    # Simplify contour to reduce point count while preserving shape
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)

    xs: list[float] = []
    ys: list[float] = []
    for pt in approx.reshape(-1, 2):
        gx, gy = _pixel_to_geo(float(pt[0]), float(pt[1]), img_w, img_h, site_bounds)
        xs.append(gx)
        ys.append(gy)

    return xs, ys


def extract_non_building_areas(
    image_bytes:       bytes,
    non_building_json: dict,
    site_polygon:      Polygon,
    crs:               str = "EPSG:3857",
) -> dict:
    """
    Extract non-building zone geometries from a lease plan image.

    Parameters
    ----------
    image_bytes       : raw bytes of the lease plan (PNG/JPEG/PDF)
    non_building_json : dict with keys:
                          color_labels      — colour → description mapping
                          non_building_areas — list of {description, reference_clause}
    site_polygon      : Shapely Polygon of the site boundary (EPSG:3857)
    crs               : coordinate reference system string

    Returns
    -------
    dict — keyed by normalised colour label, each value containing:
           use, reference_clause, location_ref, coordinates {x, y}
    """
    log.info("[lease_plan_parser] START")

    # ── Parse inputs ──────────────────────────────────────────
    color_labels      = non_building_json.get("color_labels", {})
    non_building_list = non_building_json.get("non_building_areas", [])

    # Build lookup: description → (reference_clause, location_ref)
    desc_to_meta: dict[str, dict] = {}
    for item in non_building_list:
        desc = item.get("description", "")
        desc_to_meta[desc] = {
            "reference_clause": item.get("reference_clause", ""),
            "location_ref":     item.get("location_ref", ""),
        }

    # Build lookup: description → normalised colour key
    desc_to_colour: dict[str, str] = {}
    for raw_colour, label_data in color_labels.items():
        norm_key = _normalise_colour_key(raw_colour)
        desc     = label_data.get("description", "")
        desc_to_colour[desc] = norm_key

    # ── Decode image ──────────────────────────────────────────
    img_bgr = _decode_image(image_bytes)
    img_h, img_w = img_bgr.shape[:2]

    # Convert to HSV for colour segmentation
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── Site bounding box in EPSG:3857 ────────────────────────
    site_bounds = site_polygon.bounds   # (minx, miny, maxx, maxy)
    log.info(
        f"  Site bounds: "
        f"x=[{site_bounds[0]:.0f}, {site_bounds[2]:.0f}]  "
        f"y=[{site_bounds[1]:.0f}, {site_bounds[3]:.0f}]"
    )

    # ── Process each non-building area ────────────────────────
    result: dict = {}

    for item in non_building_list:
        desc      = item.get("description", "")
        ref_clause= item.get("reference_clause", "")
        location_ref = item.get("location_ref", "")

        # Find the colour key for this description
        norm_key = desc_to_colour.get(desc)
        if norm_key is None:
            # Try fuzzy match: find colour whose description contains this desc
            for raw_colour, label_data in color_labels.items():
                if desc.lower() in label_data.get("description", "").lower():
                    norm_key = _normalise_colour_key(raw_colour)
                    break

        if norm_key is None:
            log.warning(f"  LeaseParser: no colour mapping for '{desc}' — skipping")
            continue

        log.info(f"  LeaseParser: processing '{desc}' → colour key '{norm_key}'")

        # Build colour mask
        mask = _build_colour_mask(img_hsv, norm_key)

        pixel_count = int(np.sum(mask > 0))
        log.info(f"  LeaseParser: '{norm_key}' mask pixels = {pixel_count}")

        if pixel_count < _MIN_CONTOUR_AREA_PX:
            log.warning(
                f"  LeaseParser: '{norm_key}' mask too sparse ({pixel_count}px) — skipping"
            )
            continue

        # Extract contour coordinates
        xs, ys = _extract_contour_coordinates(
            mask, img_w, img_h, site_bounds
        )

        if len(xs) < 3:
            log.warning(f"  LeaseParser: '{norm_key}' produced < 3 points — skipping")
            continue

        result[norm_key] = {
            "use":              desc,
            "reference_clause": ref_clause,
            "location_ref":     location_ref,
            "coordinates": {
                "x": xs,
                "y": ys,
            },
        }
        log.info(f"  LeaseParser: '{norm_key}' → {len(xs)} polygon vertices")

    log.info(f"[lease_plan_parser] DONE  zones extracted = {len(result)}")
    return result
