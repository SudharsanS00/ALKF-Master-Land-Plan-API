# ============================================================
# modules/spatial_intelligence.py
# Boundary Intelligence Engine  v1.1
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
#
# NOTE: imports from view.py and noise.py are done lazily inside
# functions to avoid triggering their matplotlib/contextily imports
# at module load time. This keeps startup clean and fast.
# ============================================================

from __future__ import annotations

import base64
import gc
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from modules.resolver import get_lot_boundary, resolve_location

log = logging.getLogger(__name__)

ox.settings.use_cache   = True
ox.settings.log_console = False

# View fetch radius — 300m covers the 200m analysis radius with margin,
# and keeps OSMnx response sizes small enough to stay under Render's timeout.
_VIEW_FETCH_RADIUS = 300
_VIEW_RADIUS_M     = 200
_SECTOR_SIZE       = 20   # degrees, mirrors view.py SECTOR_SIZE

# ── View label remap ──────────────────────────────────────────
# view.py produces: GREEN / WATER / CITY / OPEN
# Spec requires:    SEA / HARBOR / RESERVOIR / MOUNTAIN / PARK / GREEN / CITY
_VIEW_LABEL_REMAP = {
    "GREEN": "GREEN",
    "WATER": "SEA",
    "CITY":  "CITY",
    "OPEN":  "GREEN",
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

# Per-OSMnx-fetch timeout in seconds.
# Three fetches run concurrently — total wall time = slowest single fetch.
_OSM_FETCH_TIMEOUT = 45


def _osm_fetch_buildings(lat, lon, radius_m):
    try:
        gdf = ox.features_from_point(
            (lat, lon), dist=radius_m, tags={"building": True}
        ).to_crs(3857)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        log.info(f"  View features: {len(gdf)} buildings")
        return "buildings", gdf
    except Exception as e:
        log.warning(f"  Buildings fetch failed: {e}")
        return "buildings", gpd.GeoDataFrame(geometry=[], crs=3857)


def _osm_fetch_parks(lat, lon, radius_m):
    try:
        gdf = ox.features_from_point(
            (lat, lon), dist=radius_m,
            tags={"leisure": ["park", "garden", "nature_reserve"],
                  "landuse": ["grass", "meadow", "forest"],
                  "natural": ["wood", "scrub", "grassland"]},
        ).to_crs(3857)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        log.info(f"  View features: {len(gdf)} park/green polygons")
        return "parks", gdf
    except Exception as e:
        log.warning(f"  Parks fetch failed: {e}")
        return "parks", gpd.GeoDataFrame(geometry=[], crs=3857)


def _osm_fetch_water(lat, lon, radius_m):
    try:
        gdf = ox.features_from_point(
            (lat, lon), dist=radius_m,
            tags={"natural":  ["water", "bay", "coastline", "strait"],
                  "landuse":  ["reservoir"],
                  "waterway": ["river", "canal"]},
        ).to_crs(3857)
        gdf = gdf[gdf.geometry.type.isin(
            ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
        )]
        log.info(f"  View features: {len(gdf)} water features")
        return "water", gdf
    except Exception as e:
        log.warning(f"  Water fetch failed: {e}")
        return "water", gpd.GeoDataFrame(geometry=[], crs=3857)


def _fetch_view_features(lon: float, lat: float, radius_m: int) -> dict:
    """
    Fetch OSM buildings, parks, and water CONCURRENTLY using a thread pool.
    All three requests run in parallel — total time = slowest single fetch
    instead of the sum of all three (~110s -> ~45s on cold Render instance).

    Each fetch has an individual timeout of _OSM_FETCH_TIMEOUT seconds.
    A timed-out layer falls back to an empty GeoDataFrame so the pipeline
    continues with degraded but valid data rather than raising a 500.

    Returns dict with keys: buildings, parks, water (all EPSG:3857).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    _empty = gpd.GeoDataFrame(geometry=[], crs=3857)
    features = {"buildings": _empty, "parks": _empty, "water": _empty}

    fetchers = [
        (_osm_fetch_buildings, (lat, lon, radius_m)),
        (_osm_fetch_parks,     (lat, lon, radius_m)),
        (_osm_fetch_water,     (lat, lon, radius_m)),
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {
            pool.submit(fn, *args): fn.__name__
            for fn, args in fetchers
        }
        try:
            for future in as_completed(future_map, timeout=_OSM_FETCH_TIMEOUT + 5):
                try:
                    key, gdf = future.result(timeout=_OSM_FETCH_TIMEOUT)
                    features[key] = gdf
                except FuturesTimeoutError:
                    fname = future_map[future]
                    log.warning(f"  OSM fetch timed out: {fname} — using empty layer")
                except Exception as e:
                    fname = future_map[future]
                    log.warning(f"  OSM fetch error ({fname}): {e} — using empty layer")
        except FuturesTimeoutError:
            log.warning("  Overall OSM fetch timeout — using whatever completed")

    log.info(f"  All OSM view fetches done in {time.time() - t0:.1f}s")
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
    Imports view.py internals lazily to avoid matplotlib side-effects.
    """
    # Lazy import — avoids triggering matplotlib/contextily at module load
    from modules.view import _classify_sectors, _get_site_height

    center = Point(point_x, point_y)
    buf    = center.buffer(radius_m)

    nearby_bld      = building_data[building_data.geometry.intersects(buf)].copy()
    h_ref           = _get_site_height(nearby_bld, center)
    city_candidates = nearby_bld[nearby_bld["HEIGHT_M"] > h_ref].copy()

    parks = features["parks"]
    water = features["water"]
    parks_clip = parks[parks.geometry.intersects(buf)] if len(parks) else parks
    water_clip = water[water.geometry.intersects(buf)] if len(water) else water

    sectors  = _classify_sectors(center, parks_clip, water_clip, city_candidates, h_ref)
    counts: dict[str, int] = {}
    for s in sectors:
        v = s["view"]
        counts[v] = counts.get(v, 0) + (s["end"] - s["start"])

    dominant = max(counts, key=counts.get) if counts else "OPEN"
    label    = _VIEW_LABEL_REMAP.get(dominant, "GREEN")

    # Refine WATER → SEA / RESERVOIR / HARBOR by OSM tags
    if label == "SEA" and len(water_clip):
        for col in ["natural", "landuse", "waterway"]:
            if col in water_clip.columns:
                tag_vals = water_clip[col].dropna().str.lower().tolist()
                if any(t in tag_vals for t in ["reservoir"]):
                    label = "RESERVOIR"
                elif any(t in tag_vals for t in ["harbour", "harbor"]):
                    label = "HARBOR"
                break

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
    Uses direct classification for small sites (<=500 pts),
    grid-sample + nearest-neighbour for larger sites.
    """
    n      = len(xs)
    labels: list[str] = []

    if n <= 500:
        log.info(f"  View: direct classification for {n} points")
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                label = _classify_view_at_point(x, y, features, building_data, radius_m)
            except Exception:
                label = "CITY"
            labels.append(label)
            if (i + 1) % 100 == 0:
                log.info(f"  View: {i+1}/{n} classified")
    else:
        log.info(f"  View: grid-sample strategy for {n} points")
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

        gx_arr  = np.array([p[0] for p in grid_pts])
        gy_arr  = np.array([p[1] for p in grid_pts])
        g_labels = [grid_labels[p] for p in grid_pts]

        for x, y in zip(xs, ys):
            dists       = (gx_arr - x) ** 2 + (gy_arr - y) ** 2
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
) -> tuple:
    """
    Build noise propagation grid using noise.py pipeline internals.
    Lazy imports to avoid matplotlib side-effects at module load.

    Returns (X, Y, noise, roads_gdf) on success.
    Returns (None, None, None, roads_gdf) on pipeline failure — the road
    GeoDataFrame is always returned so _fallback_noise_from_roads can
    reuse it without a second OSMnx fetch.
    Returns (None, None, None, None) only when the road fetch itself fails.
    """
    from modules.noise import (
        ATCWFSLoader,
        CanyonAssigner,
        EmissionEngine,
        LNRSAssigner,
        LNRSWFSLoader,
        PropagationEngine,
        TrafficAssigner,
    )

    # ── Road fetch ─────────────────────────────────────────────
    roads_raw = None
    try:
        roads_raw = ox.features_from_point(
            (lat, lon),
            dist=cfg["study_radius"],
            tags={"highway": True},
        ).to_crs(3857)
        roads_raw = roads_raw[roads_raw.geometry.type.isin(["LineString", "MultiLineString"])]
        if roads_raw.empty:
            raise ValueError("no roads found")
        log.info(f"  Noise: {len(roads_raw)} road segments")
    except Exception as e:
        log.warning(f"  Noise road fetch failed: {e}")
        return None, None, None, None

    # ── Building fetch (for canyon model) ──────────────────────
    try:
        bld = ox.features_from_point(
            (lat, lon),
            dist=cfg["study_radius"],
            tags={"building": True},
        ).to_crs(3857)
        bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        bld = gpd.GeoDataFrame(geometry=[], crs=3857)

    # ── Full noise pipeline ────────────────────────────────────
    try:
        roads = roads_raw.copy()
        atc_data = ATCWFSLoader(cfg).load()
        lnrs_gdf = LNRSWFSLoader(cfg).load()
        roads    = TrafficAssigner(atc_data, cfg).assign(roads)
        roads    = LNRSAssigner(lnrs_gdf, cfg).assign(roads)
        roads    = CanyonAssigner(bld, cfg).assign(roads)
        roads    = EmissionEngine(cfg).compute(roads)
        X, Y, noise = PropagationEngine(cfg).run(roads, site_polygon)
        return X, Y, noise, roads_raw
    except Exception as e:
        log.warning(f"  Noise pipeline failed: {e}")
        return None, None, None, roads_raw


def _sample_noise_at_points(
    xs: list[float],
    ys: list[float],
    X: np.ndarray,
    Y: np.ndarray,
    noise: np.ndarray,
    noise_floor: float = 45.0,
) -> list[float]:
    """Sample the noise grid at each boundary point via nearest-neighbour."""
    if X.shape[1] < 2:
        return [noise_floor] * len(xs)

    x_vals = X[0, :]
    y_vals = Y[:, 0]

    results: list[float] = []
    for bx, by in zip(xs, ys):
        xi  = int(np.argmin(np.abs(x_vals - bx)))
        yi  = int(np.argmin(np.abs(y_vals - by)))
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
    roads_gdf=None,
) -> list[float]:
    """
    Lightweight fallback noise model when full pipeline fails.
    Direct point-source attenuation: L = L_base - 20*log10(d+1)

    roads_gdf: pre-fetched road GeoDataFrame (EPSG:3857) from _build_noise_grid.
    If provided, no additional OSMnx call is made — eliminates the 119s
    redundant fetch observed when the noise pipeline fails on Render.
    If None, falls back to a fresh OSMnx fetch (original behaviour).
    """
    _ROAD_BASE = {
        "motorway": 82.0, "motorway_link": 80.0,
        "trunk": 78.0,    "trunk_link": 76.0,
        "primary": 74.0,  "primary_link": 72.0,
        "secondary": 70.0,"secondary_link": 68.0,
        "tertiary": 66.0, "residential": 60.0,
        "service": 57.0,  "unclassified": 58.0,
    }

    if roads_gdf is not None and not roads_gdf.empty:
        roads = roads_gdf
        log.info(f"  Fallback noise: reusing {len(roads)} pre-fetched road segments")
    else:
        try:
            roads = ox.features_from_point(
                (lat, lon), dist=300, tags={"highway": True}
            ).to_crs(3857)
            roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
            if roads.empty:
                raise ValueError("no roads")
            log.info(f"  Fallback noise: fetched {len(roads)} road segments")
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
    Xb    = np.array(xs, dtype=np.float64)
    Yb    = np.array(ys, dtype=np.float64)

    for start in range(0, len(xs), CHUNK):
        end = min(start + CHUNK, len(xs))
        Xc  = Xb[start:end, np.newaxis]
        Yc  = Yb[start:end, np.newaxis]
        dx  = x2v - x1v
        dy  = y2v - y1v
        q   = np.where(dx * dx + dy * dy < 1e-6, 1e-6, dx * dx + dy * dy)
        t   = np.clip(((Xc - x1v) * dx + (Yc - y1v) * dy) / q, 0.0, 1.0)
        d   = np.sqrt((Xc - x1v - t * dx) ** 2 + (Yc - y1v - t * dy) ** 2)
        Lc  = Lv - 20.0 * np.log10(d + 1.0)
        energy = np.sum(10.0 ** (Lc / 10.0), axis=1)
        db  = 10.0 * np.log10(energy + 1e-12)
        db  = np.where(db < noise_floor, noise_floor, db)
        results.extend(db.tolist())

    return [round(v, 1) for v in results]


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
    Returns a structured JSON-serialisable site intelligence dataset.
    """
    t0 = time.time()
    log.info(f"[spatial_intelligence] START  {data_type} {value}")

    # ── 1. Resolve location ───────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info(f"  Resolved: lon={lon:.6f}  lat={lat:.6f}")

    # ── 2. Retrieve lot boundary ──────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type, extents)

    site_polygon = None
    if lot_gdf is not None:
        raw_geom = lot_gdf.geometry.iloc[0]
        if (raw_geom is not None
                and raw_geom.geom_type in ("Polygon", "MultiPolygon")
                and raw_geom.area > 10):
            site_polygon = raw_geom
            log.info(f"  Boundary: lot polygon  area={site_polygon.area:.0f}m²")

    if site_polygon is None:
        try:
            cands = ox.features_from_point(
                (lat, lon), dist=100, tags={"building": True}
            ).to_crs(3857)
            cands = cands[cands.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if len(cands):
                site_polygon = (
                    cands.assign(area=cands.area)
                    .sort_values("area", ascending=False)
                    .geometry.iloc[0]
                )
                log.info(f"  Boundary: OSM building  area={site_polygon.area:.0f}m²")
            else:
                raise ValueError("no OSM building")
        except Exception:
            pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
            site_polygon = pt.buffer(40)
            log.info("  Boundary: 40m circular buffer fallback")

    if not site_polygon.is_valid:
        site_polygon = make_valid(site_polygon)

    # ── 3. Densify boundary ───────────────────────────────────
    xs, ys = _densify_boundary(site_polygon, interval_m=1.0)
    n_pts  = len(xs)

    # ── 4. Fetch OSM features for view classification ─────────
    log.info(f"  Fetching OSM view features (radius={_VIEW_FETCH_RADIUS}m)...")
    features = _fetch_view_features(lon, lat, _VIEW_FETCH_RADIUS)

    # ── 5. View classification ────────────────────────────────
    log.info(f"  Classifying view at {n_pts} boundary points...")
    t_view     = time.time()
    view_types = _batch_classify_views(xs, ys, features, building_data, radius_m=_VIEW_RADIUS_M)
    log.info(f"  View done in {time.time() - t_view:.1f}s")

    # ── 6. Noise sampling ─────────────────────────────────────
    log.info("  Building noise grid...")
    t_noise = time.time()

    # Lazy import CFG from noise.py
    from modules.noise import CFG as NOISE_CFG
    noise_cfg   = NOISE_CFG.copy()
    X, Y, noise_grid, roads_cache = _build_noise_grid(lon, lat, site_polygon, noise_cfg)

    if X is not None:
        noise_db = _sample_noise_at_points(xs, ys, X, Y, noise_grid)
        log.info(f"  Noise grid sampled in {time.time() - t_noise:.1f}s")
    else:
        log.warning("  Noise grid failed — using fallback road model")
        noise_db = _fallback_noise_from_roads(xs, ys, lon, lat, roads_gdf=roads_cache)
        log.info(f"  Noise fallback done in {time.time() - t_noise:.1f}s")

    gc.collect()

    # ── 7. Threshold evaluation ───────────────────────────────
    is_noisy = [bool(v >= db_threshold) for v in noise_db]

    # ── 8. Site ID ────────────────────────────────────────────
    site_id = re.sub(r"\s+", "_", value.strip().upper())

    # ── 9. Assemble output ────────────────────────────────────
    output: dict = {
        "site_id":             site_id,
        "crs":                 "EPSG:3857",
        "sampling_interval_m": 1.0,
        "boundary": {
            "x": xs,
            "y": ys,
        },
        "view_type":    view_types,
        "noise_db":     noise_db,
        "db_threshold": float(db_threshold),
        "is_noisy":     is_noisy,
    }

    # ── 10. Optional lease plan extraction ───────────────────
    if non_building_json and lease_plan_b64:
        log.info("  Lease plan extraction requested...")
        try:
            from modules.lease_plan_parser import extract_non_building_areas
            image_bytes  = base64.b64decode(lease_plan_b64)
            non_building = extract_non_building_areas(
                image_bytes       = image_bytes,
                non_building_json = non_building_json,
                site_polygon      = site_polygon,
                crs               = "EPSG:3857",
            )
            output["non_building_areas"] = non_building
            log.info(f"  Lease plan: {len(non_building)} zones extracted")
        except Exception as e:
            log.warning(f"  Lease plan extraction failed: {e}")
            output["non_building_areas"] = {}
    else:
        log.info("  Lease plan inputs not provided — skipping")

    log.info(
        f"[spatial_intelligence] DONE  pts={n_pts}  "
        f"t={time.time() - t0:.1f}s"
    )
    return output


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
