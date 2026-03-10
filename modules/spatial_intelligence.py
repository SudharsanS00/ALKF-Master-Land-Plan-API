# ============================================================
# modules/spatial_intelligence.py
# Boundary Intelligence Engine  v1.0
# ALKF Master Land Plan API
#
# Orchestrates the full pipeline:
#   1. Resolve location + retrieve lot boundary
#   2. Densify boundary at 1m intervals
#   3. Classify view at each boundary point  (reuses view.py internals)
#   4. Sample noise at each boundary point   (reuses noise.py internals)
#   5. Evaluate noise threshold
#   6. (Optional) extract non-building areas from lease plan
#   7. Assemble and return structured JSON dict
# ============================================================

from __future__ import annotations

import base64
import gc
import logging
import re
import time
from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from modules.resolver import get_lot_boundary, resolve_location

# Internal engines reused from existing modules
from modules.view  import (
    _classify_sectors,
    _get_site_height,
    _make_sector,
    FETCH_RADIUS  as VIEW_FETCH_RADIUS,
    SECTOR_SIZE,
    COLOR_MAP,
)
from modules.noise import (
    ATCWFSLoader,
    CanyonAssigner,
    EmissionEngine,
    LNRSAssigner,
    LNRSWFSLoader,
    PropagationEngine,
    TrafficAssigner,
    CFG as NOISE_CFG,
)

log = logging.getLogger(__name__)

ox.settings.use_cache  = True
ox.settings.log_console = False

# ── View label mapping ────────────────────────────────────────
# view.py uses  GREEN / WATER / CITY / OPEN  internally.
# The spec requires SEA / HARBOR / RESERVOIR / MOUNTAIN / PARK / GREEN / CITY.
# We map water features by OSM tag to the richer label set.
_WATER_SUBTYPE_TAGS = {
    "sea":        ["bay", "coastline", "strait"],
    "harbor":     ["harbour", "harbor"],
    "reservoir":  ["reservoir"],
}

_VIEW_LABEL_REMAP = {
    "GREEN": "GREEN",
    "WATER": "SEA",     # default water → SEA; overridden by subtype below
    "CITY":  "CITY",
    "OPEN":  "GREEN",   # open areas with no dominant feature → GREEN
}


# ============================================================
# STEP 1 — BOUNDARY DENSIFICATION
# ============================================================

def _densify_boundary(
    polygon: Polygon | MultiPolygon,
    interval_m: float = 1.0,
) -> tuple[list[float], list[float]]:
    """
    Interpolate points along the polygon exterior every `interval_m` metres.
    CRS must be metric (EPSG:3857).

    Returns (xs, ys) — parallel lists of easting / northing coordinates.
    """
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)

    exterior: LineString = polygon.exterior
    length = exterior.length

    if length < interval_m:
        raise ValueError(
            f"Boundary perimeter {length:.1f}m is shorter than "
            f"sampling interval {interval_m}m"
        )

    n = int(np.floor(length / interval_m))
    distances = np.linspace(0.0, length - interval_m, n)

    xs: list[float] = []
    ys: list[float] = []
    for d in distances:
        pt = exterior.interpolate(d)
        xs.append(round(float(pt.x), 4))
        ys.append(round(float(pt.y), 4))

    log.info(
        f"  Boundary densified: {n} points @ {interval_m}m "
        f"over {length:.1f}m perimeter"
    )
    return xs, ys


# ============================================================
# STEP 2 — VIEW CLASSIFICATION PER BOUNDARY POINT
# ============================================================

def _fetch_view_features(
    lon: float,
    lat: float,
    radius_m: int,
) -> dict:
    """
    Fetch OSM features required for view classification.
    Returns dict with keys: buildings, parks, water, water_subtypes.
    All geometries in EPSG:3857.
    """
    features: dict = {}

    # Buildings
    try:
        bld = ox.features_from_point(
            (lat, lon), dist=radius_m, tags={"building": True}
        ).to_crs(3857)
        bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])]
        features["buildings"] = bld
        log.info(f"  View features: {len(bld)} buildings")
    except Exception as e:
        log.warning(f"  Buildings fetch failed: {e}")
        features["buildings"] = gpd.GeoDataFrame(geometry=[], crs=3857)

    # Parks / green
    try:
        parks = ox.features_from_point(
            (lat, lon), dist=radius_m,
            tags={"leisure": ["park", "garden", "nature_reserve"],
                  "landuse": ["grass", "meadow", "forest"],
                  "natural": ["wood", "scrub", "grassland"]},
        ).to_crs(3857)
        parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        features["parks"] = parks
        log.info(f"  View features: {len(parks)} park/green polygons")
    except Exception as e:
        log.warning(f"  Parks fetch failed: {e}")
        features["parks"] = gpd.GeoDataFrame(geometry=[], crs=3857)

    # Water (all types)
    try:
        water = ox.features_from_point(
            (lat, lon), dist=radius_m,
            tags={"natural":  ["water", "bay", "coastline", "strait"],
                  "landuse":  ["reservoir"],
                  "waterway": ["river", "canal"]},
        ).to_crs(3857)
        water = water[water.geometry.type.isin(["Polygon", "MultiPolygon",
                                                "LineString", "MultiLineString"])]
        features["water"] = water
        log.info(f"  View features: {len(water)} water features")
    except Exception as e:
        log.warning(f"  Water fetch failed: {e}")
        features["water"] = gpd.GeoDataFrame(geometry=[], crs=3857)

    return features


def _classify_view_at_point(
    point_x: float,
    point_y: float,
    features: dict,
    building_data: gpd.GeoDataFrame,
    radius_m: int = 200,
) -> str:
    """
    Classify the dominant view type at a single boundary point.
    Reuses view.py's _classify_sectors() and _get_site_height() directly.

    Returns one of: SEA, HARBOR, RESERVOIR, MOUNTAIN, PARK, GREEN, CITY
    """
    center = Point(point_x, point_y)

    # Clip building data to analysis radius
    buf = center.buffer(radius_m)
    nearby_bld = building_data[building_data.geometry.intersects(buf)].copy()

    h_ref = _get_site_height(nearby_bld, center)

    # City candidates — buildings taller than site height, within radius
    city_candidates = nearby_bld[nearby_bld["HEIGHT_M"] > h_ref].copy()

    parks = features["parks"]
    water = features["water"]

    # Clip to radius
    parks_clip = parks[parks.geometry.intersects(buf)] if len(parks) else parks
    water_clip = water[water.geometry.intersects(buf)] if len(water) else water

    sectors = _classify_sectors(center, parks_clip, water_clip, city_candidates, h_ref)

    # Aggregate: find the dominant view across all sectors
    counts: dict[str, int] = {}
    for s in sectors:
        v = s["view"]
        counts[v] = counts.get(v, 0) + (s["end"] - s["start"])

    dominant = max(counts, key=counts.get) if counts else "OPEN"

    # Remap to richer label set
    label = _VIEW_LABEL_REMAP.get(dominant, "GREEN")

    # Refine WATER subtypes using OSM tags
    if label == "SEA" and len(water_clip):
        tags_col = None
        for col in ["natural", "landuse", "waterway"]:
            if col in water_clip.columns:
                tags_col = col
                break
        if tags_col:
            tag_vals = water_clip[tags_col].dropna().str.lower().tolist()
            if any(t in tag_vals for t in ["reservoir"]):
                label = "RESERVOIR"
            elif any(t in tag_vals for t in ["harbour", "harbor"]):
                label = "HARBOR"

    return label


def _batch_classify_views(
    xs: list[float],
    ys: list[float],
    features: dict,
    building_data: gpd.GeoDataFrame,
    radius_m: int = 200,
) -> list[str]:
    """
    Classify view at every boundary point.

    Performance optimisation: rather than running a full sector analysis
    per point (which would be extremely slow for 1000+ points), we
    pre-compute sector features on a 5m spatial grid and do a nearest-
    neighbour lookup. For boundaries < 500 points, we classify directly.

    Returns list of view label strings, same length as xs.
    """
    n = len(xs)
    labels: list[str] = []

    if n <= 500:
        # Direct classification — acceptable for small sites
        log.info(f"  View: direct classification for {n} points")
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                label = _classify_view_at_point(
                    x, y, features, building_data, radius_m
                )
            except Exception:
                label = "CITY"
            labels.append(label)
            if (i + 1) % 100 == 0:
                log.info(f"  View: {i+1}/{n} classified")
    else:
        # Grid sampling — classify on a sparse grid, then assign to
        # nearest boundary point (reduces redundant OSM queries)
        log.info(f"  View: grid-sample strategy for {n} points")

        # Build grid at ~10m spacing
        x_arr = np.array(xs)
        y_arr = np.array(ys)
        x_min, x_max = x_arr.min(), x_arr.max()
        y_min, y_max = y_arr.min(), y_arr.max()

        grid_step = 10.0
        gx = np.arange(x_min, x_max + grid_step, grid_step)
        gy = np.arange(y_min, y_max + grid_step, grid_step)
        grid_pts = [(x, y) for x in gx for y in gy]

        grid_labels: dict[tuple, str] = {}
        for gpt in grid_pts:
            try:
                lbl = _classify_view_at_point(
                    gpt[0], gpt[1], features, building_data, radius_m
                )
            except Exception:
                lbl = "CITY"
            grid_labels[gpt] = lbl

        # Assign each boundary point to nearest grid point
        gx_arr = np.array([p[0] for p in grid_pts])
        gy_arr = np.array([p[1] for p in grid_pts])
        g_labels = [grid_labels[p] for p in grid_pts]

        for x, y in zip(xs, ys):
            dists = (gx_arr - x) ** 2 + (gy_arr - y) ** 2
            nearest_idx = int(np.argmin(dists))
            labels.append(g_labels[nearest_idx])

    log.info(f"  View: classification complete  n={n}")
    return labels


# ============================================================
# STEP 3 — NOISE SAMPLING PER BOUNDARY POINT
# ============================================================

def _build_noise_grid(
    lon: float,
    lat: float,
    site_polygon: Polygon,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the noise propagation grid using the full noise.py pipeline.
    Returns (X, Y, noise) NumPy arrays in EPSG:3857.
    Returns (None, None, None) on failure.
    """
    try:
        roads = ox.features_from_point(
            (lat, lon),
            dist=cfg["study_radius"],
            tags={"highway": True},
        ).to_crs(3857)
        roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
        if roads.empty:
            raise ValueError("no roads found")
        log.info(f"  Noise: {len(roads)} road segments")
    except Exception as e:
        log.warning(f"  Noise road fetch failed: {e}")
        return None, None, None

    try:
        bld = ox.features_from_point(
            (lat, lon),
            dist=cfg["study_radius"],
            tags={"building": True},
        ).to_crs(3857)
        bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        bld = gpd.GeoDataFrame(geometry=[], crs=3857)

    try:
        atc_data = ATCWFSLoader(cfg).load()
        lnrs_gdf = LNRSWFSLoader(cfg).load()

        roads = TrafficAssigner(atc_data, cfg).assign(roads)
        roads = LNRSAssigner(lnrs_gdf, cfg).assign(roads)
        roads = CanyonAssigner(bld, cfg).assign(roads)
        roads = EmissionEngine(cfg).compute(roads)

        X, Y, noise = PropagationEngine(cfg).run(roads, site_polygon)
        return X, Y, noise

    except Exception as e:
        log.warning(f"  Noise pipeline failed: {e}")
        return None, None, None


def _sample_noise_at_points(
    xs: list[float],
    ys: list[float],
    X: np.ndarray,
    Y: np.ndarray,
    noise: np.ndarray,
    noise_floor: float = 45.0,
) -> list[float]:
    """
    Sample the noise grid at each boundary point using nearest-neighbour lookup.
    NaN cells in the noise grid return noise_floor.
    """
    # Grid spacing
    if X.shape[1] < 2:
        return [noise_floor] * len(xs)

    x_vals = X[0, :]    # 1-D x axis
    y_vals = Y[:, 0]    # 1-D y axis

    results: list[float] = []
    for bx, by in zip(xs, ys):
        xi = int(np.argmin(np.abs(x_vals - bx)))
        yi = int(np.argmin(np.abs(y_vals - by)))
        val = noise[yi, xi]
        if not np.isfinite(val):
            val = noise_floor
        results.append(round(float(val), 1))

    return results


def _fallback_noise_from_roads(
    xs: list[float],
    ys: list[float],
    lon: float,
    lat: float,
    noise_floor: float = 45.0,
) -> list[float]:
    """
    Lightweight fallback noise model when the full pipeline fails.
    Uses direct point-source attenuation: L = L_base - 20*log10(d+1)
    """
    _ROAD_BASE = {
        "motorway": 82.0, "motorway_link": 80.0,
        "trunk": 78.0,    "trunk_link": 76.0,
        "primary": 74.0,  "primary_link": 72.0,
        "secondary": 70.0,"secondary_link": 68.0,
        "tertiary": 66.0, "residential": 60.0,
        "service": 57.0,  "unclassified": 58.0,
    }

    try:
        roads = ox.features_from_point(
            (lat, lon), dist=300, tags={"highway": True}
        ).to_crs(3857)
        roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
        if roads.empty:
            raise ValueError("no roads")
    except Exception:
        return [noise_floor] * len(xs)

    hw_col = roads.get("highway", None)
    emit = [
        _ROAD_BASE.get(
            hw if isinstance(hw, str) else (hw[0] if isinstance(hw, list) and hw else ""),
            58.0
        )
        for hw in (hw_col if hw_col is not None else ["unclassified"] * len(roads))
    ]

    # Flatten road segments
    segs: list[tuple] = []
    for geom, L in zip(roads.geometry, emit):
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for p in parts:
            coords = list(p.coords)
            for i in range(len(coords) - 1):
                segs.append((*coords[i], *coords[i + 1], L))

    if not segs:
        return [noise_floor] * len(xs)

    seg_arr = np.array(segs, dtype=np.float64)
    x1v, y1v = seg_arr[:, 0], seg_arr[:, 1]
    x2v, y2v = seg_arr[:, 2], seg_arr[:, 3]
    Lv       = seg_arr[:, 4]

    results: list[float] = []
    CHUNK = 200
    Xb = np.array(xs, dtype=np.float64)
    Yb = np.array(ys, dtype=np.float64)

    for start in range(0, len(xs), CHUNK):
        end = min(start + CHUNK, len(xs))
        Xc = Xb[start:end, np.newaxis]
        Yc = Yb[start:end, np.newaxis]

        dx = x2v - x1v
        dy = y2v - y1v
        q  = np.where(dx * dx + dy * dy < 1e-6, 1e-6, dx * dx + dy * dy)
        t  = np.clip(((Xc - x1v) * dx + (Yc - y1v) * dy) / q, 0.0, 1.0)
        d  = np.sqrt((Xc - x1v - t * dx) ** 2 + (Yc - y1v - t * dy) ** 2)

        Lc     = Lv - 20.0 * np.log10(d + 1.0)
        energy = np.sum(10.0 ** (Lc / 10.0), axis=1)
        db     = 10.0 * np.log10(energy + 1e-12)
        db     = np.where(db < noise_floor, noise_floor, db)
        results.extend(db.tolist())

    return [round(v, 1) for v in results]


# ============================================================
# STEP 4 — NORMALISE COLOUR KEY
# ============================================================

def _normalise_colour_key(raw: str) -> str:
    """
    'pink cross-hatched black'  →  'pink_cross_hatched_black'
    """
    return re.sub(r"[\s\-]+", "_", raw.strip().lower())


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def generate_site_intelligence(
    data_type:         str,
    value:             str,
    building_data:     gpd.GeoDataFrame,
    lon:               Optional[float] = None,
    lat:               Optional[float] = None,
    lot_ids:           Optional[list]  = None,
    extents:           Optional[list]  = None,
    db_threshold:      float           = 65.0,
    non_building_json: Optional[dict]  = None,
    lease_plan_b64:    Optional[str]   = None,
) -> dict:
    """
    Full boundary intelligence pipeline.

    Parameters
    ----------
    data_type         : resolver input type ("LOT", "ADDRESS", etc.)
    value             : resolver input value ("IL 1657", etc.)
    building_data     : preloaded BUILDINGS_FINAL GeoDataFrame (EPSG:3857)
    lon, lat          : pre-resolved WGS84 coordinates (required for ADDRESS)
    lot_ids           : optional multi-lot ID list
    extents           : optional multi-lot extent list
    db_threshold      : noise threshold in dBA (default 65.0)
    non_building_json : optional colour label + non_building_areas dict
    lease_plan_b64    : optional base64-encoded lease plan PDF/image

    Returns
    -------
    dict — structured JSON-serialisable site intelligence dataset
    """
    t0 = time.time()
    log.info(f"[spatial_intelligence] START  {data_type} {value}")

    # ── 1. Resolve location ───────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info(f"  Resolved: lon={lon:.6f}  lat={lat:.6f}")

    # ── 2. Retrieve lot boundary ──────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type, extents)

    if lot_gdf is not None:
        raw_geom = lot_gdf.geometry.iloc[0]
        if raw_geom is not None and raw_geom.geom_type in ("Polygon", "MultiPolygon") and raw_geom.area > 10:
            site_polygon = raw_geom
            log.info(f"  Boundary: lot polygon  area={site_polygon.area:.0f}m²")
        else:
            site_polygon = None
    else:
        site_polygon = None

    if site_polygon is None:
        # Fallback: OSM building footprint
        try:
            cands = ox.features_from_point(
                (lat, lon), dist=100, tags={"building": True}
            ).to_crs(3857)
            cands = cands[cands.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if len(cands):
                site_polygon = cands.assign(
                    area=cands.area
                ).sort_values("area", ascending=False).geometry.iloc[0]
                log.info(f"  Boundary: OSM building  area={site_polygon.area:.0f}m²")
            else:
                raise ValueError("no OSM building")
        except Exception:
            pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
            site_polygon = pt.buffer(40)
            log.info("  Boundary: 40m circular buffer fallback")

    # Ensure valid geometry
    if not site_polygon.is_valid:
        site_polygon = make_valid(site_polygon)

    # ── 3. Densify boundary ───────────────────────────────────
    xs, ys = _densify_boundary(site_polygon, interval_m=1.0)
    n_pts = len(xs)

    # ── 4. Fetch OSM features for view classification ─────────
    log.info(f"  Fetching OSM view features (radius={VIEW_FETCH_RADIUS}m)...")
    features = _fetch_view_features(lon, lat, VIEW_FETCH_RADIUS)

    # ── 5. View classification ────────────────────────────────
    log.info(f"  Classifying view at {n_pts} boundary points...")
    t_view = time.time()
    view_types = _batch_classify_views(xs, ys, features, building_data, radius_m=200)
    log.info(f"  View done in {time.time() - t_view:.1f}s")

    # ── 6. Noise sampling ─────────────────────────────────────
    log.info("  Building noise grid...")
    t_noise = time.time()
    noise_cfg = NOISE_CFG.copy()
    X, Y, noise_grid = _build_noise_grid(lon, lat, site_polygon, noise_cfg)

    if X is not None:
        noise_db = _sample_noise_at_points(xs, ys, X, Y, noise_grid)
        log.info(f"  Noise grid sampled in {time.time() - t_noise:.1f}s")
    else:
        log.warning("  Noise grid failed — using fallback road model")
        noise_db = _fallback_noise_from_roads(xs, ys, lon, lat)
        log.info(f"  Noise fallback done in {time.time() - t_noise:.1f}s")

    gc.collect()

    # ── 7. Threshold evaluation ───────────────────────────────
    is_noisy = [bool(v >= db_threshold) for v in noise_db]

    # ── 8. Site ID ────────────────────────────────────────────
    site_id = re.sub(r"\s+", "_", value.strip().upper())

    # ── 9. Assemble base output ───────────────────────────────
    output: dict = {
        "site_id":            site_id,
        "crs":                "EPSG:3857",
        "sampling_interval_m": 1.0,
        "boundary": {
            "x": xs,
            "y": ys,
        },
        "view_type":   view_types,
        "noise_db":    noise_db,
        "db_threshold": float(db_threshold),
        "is_noisy":    is_noisy,
    }

    # ── 10. Optional lease plan extraction ───────────────────
    if non_building_json and lease_plan_b64:
        log.info("  Lease plan extraction requested...")
        try:
            from modules.lease_plan_parser import extract_non_building_areas
            image_bytes = base64.b64decode(lease_plan_b64)
            non_building = extract_non_building_areas(
                image_bytes      = image_bytes,
                non_building_json= non_building_json,
                site_polygon     = site_polygon,
                crs              = "EPSG:3857",
            )
            output["non_building_areas"] = non_building
            log.info(f"  Lease plan: extracted {len(non_building)} zones")
        except Exception as e:
            log.warning(f"  Lease plan extraction failed: {e}")
            output["non_building_areas"] = {}
    else:
        log.info("  Lease plan inputs not provided — skipping extraction")

    log.info(
        f"[spatial_intelligence] DONE  "
        f"pts={n_pts}  t={time.time()-t0:.1f}s"
    )
    return output
