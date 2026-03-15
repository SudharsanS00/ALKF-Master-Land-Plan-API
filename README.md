# ALKF Master Land Plan API

**Boundary Intelligence Engine** — Flask microservice that walks a site boundary at 1-metre intervals and records view classification and noise level at every point, returning a structured JSON dataset or a DXF CAD file ready for import into AutoCAD, Rhino, or QGIS.

Part of the **ALKF+ Automated Spatial Intelligence Platform**.

---

## Architecture Diagram

![ALKF Master Land Plan API Architecture](./architecture.svg)

---

## Table of Contents

1. [Overview](#overview)
2. [What Changed: FastAPI → Flask](#what-changed-fastapi--flask)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [Data Prerequisites](#data-prerequisites)
6. [Modules](#modules)
   - [app.py](#apppy)
   - [spatial_intelligence.py](#spatial_intelligencepy)
   - [dxf_export.py](#dxf_exportpy)
   - [lease_plan_parser.py](#lease_plan_parserpy)
   - [entry_point_detector.py](#entry_point_detectorpy)
   - [resolver.py](#resolverpy)
   - [view.py](#viewpy)
   - [noise.py](#noisepy)
7. [API Reference](#api-reference)
   - [GET /](#get-)
   - [POST /site-intelligence](#post-site-intelligence)
   - [POST /site-intelligence-dxf](#post-site-intelligence-dxf)
8. [Request Model](#request-model)
9. [Response Schema](#response-schema)
   - [Basic Response](#basic-response)
   - [Extended Response — with Lease Plan](#extended-response--with-lease-plan)
   - [Extended Response — with Entry Points](#extended-response--with-entry-points)
10. [DXF Output Specification](#dxf-output-specification)
    - [Layer Definitions](#layer-definitions)
    - [Per-point Colour Mapping](#per-point-colour-mapping)
    - [Title Block](#title-block)
    - [Importing into CAD Software](#importing-into-cad-software)
11. [Algorithms](#algorithms)
    - [Boundary Densification](#boundary-densification)
    - [View Classification](#view-classification)
    - [Noise Sampling](#noise-sampling)
    - [Lease Plan Colour Segmentation](#lease-plan-colour-segmentation)
    - [Vehicle Entry Point Detection](#vehicle-entry-point-detection)
12. [JSON Serialisation — _compact_json()](#json-serialisation--_compact_json)
13. [Caching](#caching)
14. [Deployment](#deployment)
    - [Local Development](#local-development)
    - [Render Cloud](#render-cloud)
    - [Worker Strategy — gthread vs gevent](#worker-strategy--gthread-vs-gevent)
15. [Dependencies](#dependencies)
16. [Environment Notes](#environment-notes)
17. [Testing](#testing)
18. [Known Limitations](#known-limitations)
19. [Bug Fixes in v1.2](#bug-fixes-in-v12)
20. [Relationship to alkf-site-analysis](#relationship-to-alkf-site-analysis)
21. [Changelog](#changelog)

---

## Overview

Given any supported site identifier (lot number, address with coordinates, or CSUID), the Boundary Intelligence Engine:

1. Resolves the identifier to WGS84 coordinates via the HK GeoData API
2. Retrieves the official lot boundary polygon from the LandsD iC1000 API (GML → EPSG:3857)
3. Densifies the boundary exterior at **1-metre intervals** using Shapely interpolation
4. Classifies the dominant **view type** at each boundary point (`SEA`, `HARBOR`, `RESERVOIR`, `GREEN`, `PARK`, `CITY`, `MOUNTAIN`) using the view sector model from `view.py`
5. Samples the **road traffic noise level** (dBA) at each boundary point using the full noise propagation pipeline from `noise.py`, with a vectorised fallback model if the WFS pipeline fails
6. Evaluates each point against a configurable **noise threshold** (default 65 dBA per HK EPD)
7. Optionally extracts **non-building zone polygons** from a lease plan image or PDF using OpenCV colour segmentation, and maps pixel coordinates to EPSG:3857
8. Optionally detects **vehicle entry points** (X, Y, Z labels) from a lease plan image by finding gaps in the green verge strip
9. Returns either a structured **JSON dataset** or a **DXF CAD file** containing all of the above as named layers

---

## What Changed: FastAPI → Flask

The API was migrated from FastAPI + uvicorn to Flask + Gunicorn in v1.2. The external contract (endpoints, request body, response body, HTTP status codes, CORS) is **fully preserved** with one exception.

| Concern | FastAPI (v1.0–v1.1) | Flask (v1.2+) |
|---|---|---|
| Framework | `FastAPI()` + `CORSMiddleware` | `Flask(__name__)` + `flask_cors.CORS` |
| Request validation | Pydantic `BaseModel` auto-parses JSON body | `_parse_body()` checks `Content-Type`; `_normalise_request()` validates fields |
| JSON response | `JSONResponse(content=result)` | `_json_response(result)` using `_compact_json()` |
| DXF response | `StreamingResponse(buf, media_type="application/dxf")` | `Response(buf.read(), mimetype="application/dxf")` |
| Error response | `raise HTTPException(status_code=422, detail=msg)` | `return _err(422, msg)` → `{"error": msg}` |
| **Error body key** | `{"detail": "message"}` | **`{"error": "message"}`** ← only breaking change |
| Auto docs | `/docs` Swagger UI, `/redoc`, `/openapi.json` | Not available |
| Server | `uvicorn app:app` | `gunicorn app:app --worker-class gthread --threads 4` |
| Startup event | `@app.on_event("startup")` | Module-level execution at import time |

> **Frontend note:** If your frontend reads the error message body on failed requests, change `e.detail` to `e.error` in your fetch error handlers. All success paths are unchanged.

---

## Architecture

```
Client Request
      │
      ▼
┌────────────────────────────────────────────────────────────────────┐
│  Flask  (app.py)                                                   │
│  Served by Gunicorn — worker-class gthread, 4 threads, timeout 300 │
│                                                                    │
│  GET  /                          → health check JSON               │
│  POST /site-intelligence         → _json_response(result)          │
│  POST /site-intelligence-dxf     → Response(dxf_bytes)             │
│                                                                    │
│  Request lifecycle:                                                │
│    _parse_body()         — validate Content-Type, decode JSON      │
│    _normalise_request()  — validate fields, coerce types, defaults │
│    cache check           — MD5(data_type+value+threshold)          │
│    generate_site_intelligence()  — pipeline                        │
│    _json_response()      — _compact_json() serialisation           │
│                                                                    │
│  In-memory cache (CACHE_STORE dict)                                │
│    Skipped when: lease_plan_b64 present OR detect_entry_points=True│
└──────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  generate_site_intelligence()  —  modules/spatial_intelligence.py  │
│                                                                    │
│  Step 1   resolve_location()                                       │
│           → HK GeoData API → (lon, lat) WGS84                     │
│           → get_lot_boundary() → iC1000 GML → Polygon EPSG:3857   │
│           fallback 1: largest OSM building polygon within 100m     │
│           fallback 2: 40m circular buffer around centroid          │
│                                                                    │
│  Step 2   _densify_boundary(polygon, interval_m=1.0)               │
│           → Shapely exterior.interpolate(d) every 1m              │
│           → (xs, ys) parallel float lists in EPSG:3857             │
│                                                                    │
│  Step 3   _fetch_view_features(lon, lat, radius_m=300)             │
│           → OSMnx features_from_point — 3 concurrent fetches:     │
│              buildings (tags: building=True)                       │
│              parks     (leisure/landuse/natural)                   │
│              water     (natural/landuse/waterway)                  │
│           → each fetch has _OSM_FETCH_TIMEOUT=45s individual limit │
│                                                                    │
│  Step 4   _batch_classify_views(xs, ys, features, building_data)   │
│           ≤500 pts → direct per-point classification               │
│           >500 pts → 10m grid sample + nearest-neighbour assign    │
│           calls view.py: _classify_sectors(), _get_site_height()   │
│                                                                    │
│  Step 5   _build_noise_grid(lon, lat, site_polygon, cfg)           │
│           Full noise.py pipeline:                                  │
│             ATCWFSLoader  → CSDI WFS ATC traffic census            │
│             LNRSWFSLoader → CSDI WFS low-noise road surface        │
│             TrafficAssigner → snap ATC stations to roads (≤500m)   │
│             LNRSAssigner   → apply −3 dB correction                │
│             CanyonAssigner → add canyon reflection (up to +8 dB)   │
│             EmissionEngine → L_link (dBA) per segment (EPD formula)│
│             PropagationEngine.run() → 5m grid + Gaussian σ=1.5    │
│             → returns (X[grid], Y[grid], noise[i,j])               │
│           On failure → _fallback_noise_from_roads()                │
│             → OSM roads + L(r) = L_class − 20·log₁₀(r+1)         │
│                                                                    │
│  Step 6   _sample_noise_at_points(xs, ys, X, Y, noise)            │
│           → nearest-neighbour index lookup into noise grid         │
│                                                                    │
│  Step 7   is_noisy = [v >= db_threshold for v in noise_db]        │
│                                                                    │
│  Step 8   site_id = re.sub(r"\s+", "_", value.upper())            │
│                                                                    │
│  Step 9   Assemble core output dict                                │
│                                                                    │
│  Step 10  (Optional — if non_building_json AND lease_plan_b64)     │
│           lease_plan_parser.extract_non_building_areas()           │
│           → decode base64 → BGR → HSV                             │
│           → per colour: inRange mask → morphology → findContours   │
│           → approxPolyDP → _pixel_to_geo() → EPSG:3857            │
│                                                                    │
│  Step 11  (Optional — if detect_entry_points AND lease_plan_b64)   │
│           entry_point_detector.extract_entry_points()              │
│           → HSV-segment green verge strip                          │
│           → extract site outer contour                             │
│           → walk contour, find gaps (no green = access opening)    │
│           → subdivide each gap → X/Y/Z labels                     │
│           → _pixel_to_geo() → EPSG:3857                           │
│                                                                    │
│  → return JSON-serialisable dict                                   │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │
                          ┌────────┴─────────┐
                          ▼                  ▼
                  _json_response()      export_dxf()
                  _compact_json()       ezdxf R2010
                  inline arrays         StringIO → encode
                  float decimals        → BytesIO → Response
```

### Static Data (Startup Preload)

At module import time, `app.py` loads one GeoDataFrame into memory:

| Dataset | File | Raw rows | Filtered to | Purpose |
|---|---|---|---|---|
| Building heights | `data/BUILDINGS_FINAL.gpkg` | ~342,000 | `HEIGHT_M > 5m` → 42,073 rows | View classification reference height |

`ZONE_REDUCED.gpkg` is not loaded — zoning analysis is not part of the MLP output.

---

## Repository Structure

```
alkf-master-land-plan/
│
├── app.py                        # Flask application — endpoints, cache, serialisation
├── render.yaml                   # Render cloud deployment (Gunicorn gthread)
├── requirements.txt              # Python dependencies
├── runtime.txt                   # python-3.11.4
├── architecture.svg              # Architecture diagram embedded in this README
│
├── data/
│   └── BUILDINGS_FINAL.gpkg      # Building footprints with HEIGHT_M (EPSG:3857)
│
└── modules/
    ├── __init__.py               # Package init (required — was missing in v1.0)
    │
    │   ── New modules (this repository) ─────────────────────────
    ├── spatial_intelligence.py   # Core 11-step pipeline orchestrator
    ├── dxf_export.py             # DXF R2010 CAD writer (ezdxf)
    ├── lease_plan_parser.py      # OpenCV HSV colour segmentation engine
    ├── entry_point_detector.py   # Vehicle access point (X/Y/Z) detector
    │
    │   ── Copied from alkf-site-analysis/modules/ ────────────────
    ├── resolver.py               # Multi-type location resolver + iC1000 boundary API
    ├── view.py                   # 360° view sector classification engine
    └── noise.py                  # Road traffic noise propagation model (EPD HK)
```

---

## Data Prerequisites

### Step 1 — Copy modules from alkf-site-analysis

```bash
cp ../alkf-site-analysis/modules/resolver.py  ./modules/resolver.py
cp ../alkf-site-analysis/modules/view.py       ./modules/view.py
cp ../alkf-site-analysis/modules/noise.py      ./modules/noise.py
```

These three files are not committed here to avoid duplication. They are used unmodified.

### Step 2 — Copy building heights data

```bash
cp ../alkf-site-analysis/data/BUILDINGS_FINAL.gpkg  ./data/BUILDINGS_FINAL.gpkg
```

`BUILDINGS_FINAL.gpkg` contains 42,073 building footprint polygons with a `HEIGHT_M` column in EPSG:3857, pre-filtered from the full LandsD dataset (~342,000 rows).

### Step 3 — (Optional) Poppler for PDF lease plans

`pdf2image` requires `poppler-utils` to rasterise PDF files. PNG and JPEG inputs work without Poppler.

```bash
# Ubuntu / Render
apt-get install -y poppler-utils

# macOS
brew install poppler
```

---

## Modules

### `app.py`

The Flask application. Responsibilities:

- Load `BUILDINGS_FINAL.gpkg` into `BUILDING_DATA` at startup
- Define `_parse_body()` — checks `request.is_json`, calls `request.get_json(silent=True)`
- Define `_normalise_request()` — validates all fields, coerces numerics, applies defaults
- Define `_compact_json()` — custom JSON serialiser (see [JSON Serialisation](#json-serialisation--_compact_json))
- Define `_json_response()` — wraps `_compact_json()` in a Flask `Response`
- Expose three routes: `GET /`, `POST /site-intelligence`, `POST /site-intelligence-dxf`
- Maintain `CACHE_STORE` in-process dict with MD5 keying

Key internal helpers:

```python
def _parse_body() -> tuple[dict, str | None]:
    """Returns (data, None) on success or (None, error_message) on failure."""

def _normalise_request(data: dict) -> tuple:
    """
    Returns (dt, value, lon, lat, lot_ids, extents, threshold,
             non_building_json, lease_plan_b64, detect_entry_points).
    Raises ValueError on bad input.
    """

def _err(status: int, message: str) -> Response:
    """Returns Flask Response: {"error": message}, status code."""

def _compact_json(obj: dict) -> str:
    """Serialises dict to JSON with inline arrays and float decimals preserved."""

def _json_response(data: dict, status: int = 200) -> Response:
    """Returns Flask Response with compact JSON body."""
```

---

### `spatial_intelligence.py`

Core pipeline orchestrator. Version v1.2.

#### Public interface

```python
def generate_site_intelligence(
    data_type:           str,
    value:               str,
    building_data:       gpd.GeoDataFrame,
    lon:                 Optional[float] = None,
    lat:                 Optional[float] = None,
    lot_ids:             Optional[list]  = None,
    extents:             Optional[list]  = None,
    db_threshold:        float           = 65.0,
    non_building_json:   Optional[dict]  = None,
    lease_plan_b64:      Optional[str]   = None,
    detect_entry_points: bool            = False,
) -> dict
```

Returns a JSON-serialisable dict. See [Response Schema](#response-schema).

#### Internal functions

| Function | Description |
|---|---|
| `_densify_boundary(polygon, interval_m)` | Interpolates points along polygon exterior every `interval_m` metres. Returns `(xs, ys)` lists in EPSG:3857. For MultiPolygon, uses largest sub-polygon. |
| `_osm_fetch_buildings(lat, lon, radius_m)` | Fetches OSM building polygons within radius. Returns `("buildings", GeoDataFrame)`. |
| `_osm_fetch_parks(lat, lon, radius_m)` | Fetches OSM leisure/landuse/natural green polygons. Returns `("parks", GeoDataFrame)`. |
| `_osm_fetch_water(lat, lon, radius_m)` | Fetches OSM water features (polygon + linestring). Returns `("water", GeoDataFrame)`. |
| `_fetch_view_features(lon, lat, radius_m)` | Runs the three OSM fetches concurrently via `ThreadPoolExecutor(max_workers=3)`. Each has a 45s timeout. Returns `{"buildings": gdf, "parks": gdf, "water": gdf}`. |
| `_classify_view_at_point(x, y, features, building_data, radius_m)` | Lazy-imports `view.py`. Builds 200m buffer, clips features, calls `_classify_sectors()` and `_get_site_height()`. Remaps WATER→SEA/HARBOR/RESERVOIR by OSM tags. |
| `_batch_classify_views(xs, ys, features, building_data, radius_m)` | ≤500 points: direct classification. >500 points: classify 10m grid, assign via nearest-neighbour. |
| `_build_noise_grid(lon, lat, site_polygon, cfg)` | Lazy-imports `noise.py`. Full WFS pipeline. Returns `(X, Y, noise_grid, roads_gdf)` or `(None, None, None, roads_gdf)` on failure. |
| `_sample_noise_at_points(xs, ys, X, Y, noise)` | Nearest-neighbour lookup into noise grid per boundary point. |
| `_fallback_noise_from_roads(xs, ys, lon, lat, roads_gdf)` | Point-source attenuation `L = L_class − 20·log₁₀(d+1)`. Vectorised via NumPy broadcasting in 200-point chunks. Reuses pre-fetched `roads_gdf` to avoid duplicate OSMnx call. |

#### Lazy import strategy

`view.py` and `noise.py` both import `matplotlib`, `contextily`, and `scikit-learn` at module level. To avoid pulling those heavy libraries into memory at Flask startup, all calls into those modules use lazy imports inside the consuming function:

```python
def _classify_view_at_point(...):
    from modules.view import _classify_sectors, _get_site_height  # loaded on first call only
    ...

def _build_noise_grid(...):
    from modules.noise import ATCWFSLoader, PropagationEngine, ...  # loaded on first call only
    ...
```

---

### `dxf_export.py`

Converts the site intelligence JSON dict into a DXF R2010 file using `ezdxf`. Version v1.2.

#### Public interface

```python
def export_dxf(intelligence_data: dict) -> BytesIO
```

Returns a `BytesIO` buffer containing a valid DXF R2010 file ready for streaming.

#### v1.2 fix — StringIO replaces tempfile

The original implementation wrote to a temp file on disk and read it back, which created a `NameError` if the read-back failed (`raw` variable referenced outside `try` block). v1.2 uses an in-memory `StringIO`:

```python
import io
txt_buf = io.StringIO()
doc.write(txt_buf)
raw = txt_buf.getvalue().encode("utf-8")
buf = BytesIO(raw)
```

No temp files. No disk I/O. No cleanup needed. No `NameError` risk.

#### DXF document settings

| Setting | Value | Purpose |
|---|---|---|
| DXF version | `R2010` (AC1024) | Broadest compatibility (AutoCAD 2010+, Rhino, QGIS) |
| `$INSUNITS` | `6` | Metres |
| `$LUNITS` | `2` | Decimal |
| `$LUPREC` | `4` | 4 decimal places |

#### Label scaling

Text height and label offset are computed dynamically from the site bounding box:

```python
def _label_scale(xs, ys):
    xmin, ymin, xmax, ymax = _bbox(xs, ys)
    short = max(min(xmax - xmin, ymax - ymin), 5.0)
    return max(0.6, round(short * 0.06, 1)), max(1.0, round(short * 0.10, 1))
```

This ensures labels are readable at any lot size — from a 20m frontage to a 2km campus.

---

### `lease_plan_parser.py`

Extracts non-building zone polygons from a lease plan image or PDF using OpenCV colour segmentation in HSV colour space.

#### Public interface

```python
def extract_non_building_areas(
    image_bytes:        bytes,
    non_building_json:  dict,
    site_polygon:       Polygon,
    crs:                str = "EPSG:3857",
) -> dict
```

Returns a dict keyed by normalised colour label (e.g. `"pink_cross_hatched_black"`), each containing zone metadata and EPSG:3857 coordinates.

#### Pipeline steps

1. **Decode** — `cv2.imdecode` for PNG/JPEG; `pdf2image.convert_from_bytes` for PDF (requires Poppler)
2. **Convert** — BGR → HSV
3. **Per colour label:**
   - Extract base colour: `"pink_cross_hatched_black"` → `"pink"`
   - `cv2.inRange(hsv, lower, upper)` → binary mask (red uses dual-range for H wrap at 180)
   - Morphological cleanup: `MORPH_CLOSE` (fill gaps) then `MORPH_OPEN` (denoise), 5×5 ellipse kernel
   - `cv2.findContours(RETR_EXTERNAL)`
   - Filter: `area >= 200 px`
   - `approxPolyDP(ε = 0.002 × arc_length)` — Douglas-Peucker simplification
   - `_pixel_to_geo()` — linear interpolation assuming north-up image

#### Supported colour labels

| Base colour | HSV lower | HSV upper | Notes |
|---|---|---|---|
| `pink` | H=140, S=30, V=150 | H=175, S=160, V=255 | |
| `green` | H=35, S=40, V=40 | H=90, S=255, V=255 | |
| `blue` | H=90, S=50, V=50 | H=130, S=255, V=255 | |
| `yellow` | H=20, S=80, V=80 | H=35, S=255, V=255 | |
| `red` | H=0, S=50, V=50 | H=10, S=255, V=255 | Dual-range — also H=165..179 |
| `orange` | H=10, S=80, V=80 | H=20, S=255, V=255 | |
| `purple` | H=125, S=30, V=50 | H=145, S=255, V=255 | |
| `grey` | H=0, S=0, V=80 | H=179, S=40, V=200 | |
| `white` | H=0, S=0, V=200 | H=179, S=30, V=255 | |
| `black` | H=0, S=0, V=0 | H=179, S=255, V=50 | |

Composite keys such as `"pink cross-hatched black"` are accepted — the first token matching the colour table is used as the base colour. The cross-hatching pattern does not affect HSV masking; only the base colour is segmented.

#### Pixel-to-geo mapping

Assumes north-up image with linear scale aligned to the site bounding box:

```
geo_x = site_bbox.minx + (pixel_col / image_width)  × (site_bbox.maxx − site_bbox.minx)
geo_y = site_bbox.maxy − (pixel_row / image_height) × (site_bbox.maxy − site_bbox.miny)
```

---

### `entry_point_detector.py`

Detects vehicle access points (X, Y, Z …) from a lease plan image by identifying gaps in the green landscaping / verge strip that borders the site boundary. Added in v1.2.

#### Public interface

```python
def extract_entry_points(
    image_bytes:      bytes,
    site_polygon:     Polygon,
    crs:              str  = "EPSG:3857",
    points_per_gap:   int  = 3,
    label_names:      list = None,
) -> dict
```

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `image_bytes` | `bytes` | Raw bytes of the lease plan (PNG / JPEG / PDF) |
| `site_polygon` | `Polygon` | Shapely Polygon of the site boundary in `crs` |
| `crs` | `str` | Coordinate reference system (default EPSG:3857) |
| `points_per_gap` | `int` | Number of labelled sub-points per entry gap. `3` → assigns X, Y, Z within a single wide entry gap. `1` → single midpoint per gap. |
| `label_names` | `list` | Override the default X/Y/Z/A/B/C… label sequence. |

#### Return value

```json
{
  "crs": "EPSG:3857",
  "gap_count": 1,
  "gaps": [
    { "gap_index": 0, "length_pts": 61, "labels": ["X", "Y", "Z"] }
  ],
  "entry_points": [
    { "label": "X", "pixel_x": 327, "pixel_y": 168, "geo_x": 12700201.5, "geo_y": 2560301.2 },
    { "label": "Y", "pixel_x": 310, "pixel_y": 188, "geo_x": 12700199.3, "geo_y": 2560298.8 },
    { "label": "Z", "pixel_x": 301, "pixel_y": 213, "geo_x": 12700198.1, "geo_y": 2560295.4 }
  ]
}
```

#### Detection pipeline

1. Decode image (reuses `_decode_image` from `lease_plan_parser`)
2. HSV-segment the green verge strip: `H∈[25,45]`, `S∈[50,160]`, `V∈[140,220]`
3. HSV-segment the pink site area: `H∈[0,15]`, `S∈[15,100]`, `V∈[180,255]`
4. Merge masks → fill with `MORPH_CLOSE(20×20)` → extract largest outer contour
5. Walk the contour: for each boundary point, probe a `±12px` patch for green pixels
6. Runs of ≥3 contiguous points with no green = access gap
7. Filter gaps: `3 ≤ length ≤ 100` contour points (plausible vehicle entry width)
8. Subdivide each gap into `points_per_gap` evenly-spaced samples → assign sequential labels (default: X, Y, Z, A, B, C…)
9. Convert pixel → geo via `_pixel_to_geo()` (reused from `lease_plan_parser`)

---

### `resolver.py` *(copied from alkf-site-analysis)*

Multi-type location resolver. Translates a `data_type` + `value` pair into WGS84 coordinates and retrieves the official lot boundary polygon from the LandsD iC1000 API.

**Coordinate transform chain:** EPSG:2326 (HK1980 Grid) → EPSG:4326 (WGS84) → EPSG:3857 (Web Mercator) via PyProj.

| `data_type` | Description | Boundary source |
|---|---|---|
| `LOT` | Inland lot (IL, NKIL, KIL, etc.) | LandsD iC1000 `lot` endpoint |
| `STT` | Short-term tenancy lot | iC1000 `stt` endpoint |
| `GLA` | Government land allocation | iC1000 `gla` endpoint |
| `LPP` | Licence / permit parcel | iC1000 `lpp` endpoint |
| `UN` | Utility notation | iC1000 `un` endpoint |
| `BUILDINGCSUID` | Building CSUID | iC1000 |
| `LOTCSUID` | Lot CSUID | iC1000 |
| `PRN` | Property reference number | iC1000 |
| `ADDRESS` | Pre-resolved | Pass `lon`/`lat` directly; boundary always `None` |

---

### `view.py` *(copied from alkf-site-analysis)*

360° view sector classification engine. Divides the horizon into `SECTOR_SIZE=20°` wedges and scores each sector as `GREEN`, `WATER`, `CITY`, or `OPEN` based on green-space ratio, water-body ratio, and building height/density relative to site reference height.

**Sector scoring model:**

```
Green Score  = green_ratio                              → label: GREEN
Water Score  = water_ratio                              → label: WATER
City Score   = height_norm × density_norm               → label: CITY
Open Score   = (1 − density_norm) × (1 − height_norm)  → label: OPEN
Priority: CITY > WATER > GREEN > OPEN
```

**View label remapping** (applied in `spatial_intelligence.py`):

| `view.py` output | MLP label | Refinement condition |
|---|---|---|
| `GREEN` | `GREEN` | — |
| `WATER` | `SEA` | Default |
| `WATER` | `RESERVOIR` | OSM tag `landuse=reservoir` detected |
| `WATER` | `HARBOR` | OSM tag `natural=harbour` detected |
| `OPEN` | `GREEN` | Remapped to GREEN |
| `CITY` | `CITY` | — |

Used in this repository via lazy imports of `_classify_sectors()` and `_get_site_height()`.

---

### `noise.py` *(copied from alkf-site-analysis)*

Road traffic noise propagation model implementing EPD Hong Kong empirical formulae.

#### Full pipeline

```
ATCWFSLoader     — fetches ATC traffic census data from CSDI WFS (up to 10,000 stations)
LNRSWFSLoader    — fetches Low Noise Road Surface zones from CSDI WFS
TrafficAssigner  — snaps ATC stations to road segments (threshold: 500m)
LNRSAssigner     — applies −3 dB correction to LNRS-designated segments
CanyonAssigner   — adds canyon reflection bonus (0–8 dB) based on building enclosure
EmissionEngine   — computes L_link (dBA) per segment using EPD formula
PropagationEngine.run()
    → accumulates noise on a 5m resolution grid over 150m radius
    → applies Gaussian smoothing (σ=1.5)
    → returns (X[grid], Y[grid], noise[i,j])
```

#### Base emission model (EPD HK)

```
L₀ = A + B·log₁₀(Q) + correction_terms
```

Where `Q` = flow (vehicles/hour) and correction terms cover: heavy vehicle fraction, speed adjustment, ground absorption (`α=0.6`), and canyon reflection bonus.

#### Fallback model

When the WFS pipeline fails (CSDI API unavailable, no roads in radius, or propagation error):

```
L(r) = L_road_class − 20·log₁₀(r + 1)

Road-class base levels (dBA):
  motorway:     82    trunk:         78
  primary:      74    secondary:     70
  tertiary:     66    residential:   60
  service:      57    unclassified:  58
```

Vectorised across all boundary points and road segments simultaneously using NumPy broadcasting in 200-point chunks. The pre-fetched `roads_gdf` from `_build_noise_grid` is passed directly to avoid a duplicate OSMnx call.

---

## API Reference

### Base URL

```
https://alkf-master-land-plan-api.onrender.com
```

### `GET /`

Health check.

**Response — 200 OK:**

```json
{
  "service": "ALKF Master Land Plan API",
  "version": "1.2",
  "status":  "operational"
}
```

---

### `POST /site-intelligence`

Runs the full boundary intelligence pipeline and returns a structured JSON dataset.

**Content-Type:** `application/json`
**Response:** `200 OK` — `application/json`

**Error responses:**

| Code | Body | Reason |
|---|---|---|
| `400` | `{"error": "Request Content-Type must be application/json"}` | Wrong or missing Content-Type |
| `400` | `{"error": "Invalid or empty JSON body"}` | Malformed JSON |
| `422` | `{"error": "ADDRESS type requires pre-resolved lon and lat"}` | ADDRESS without coordinates |
| `422` | `{"error": "'data_type' is required"}` | Missing required field |
| `500` | `{"error": "<detail>"}` | Resolver failure; OSM timeout; geometry error |

---

### `POST /site-intelligence-dxf`

Identical computation to `/site-intelligence`. Returns a DXF R2010 file as a binary download.

**Content-Type:** `application/json`
**Response:** `200 OK` — `application/dxf`
**Content-Disposition:** `attachment; filename="{site_id}_boundary_intelligence.dxf"`

**Additional error:**

| Code | Body | Reason |
|---|---|---|
| `500` | `{"error": "DXF export error: <detail>"}` | ezdxf serialisation failure |

---

## Request Model

```json
{
  "data_type":           "LOT",
  "value":               "IL 1657",
  "lon":                 null,
  "lat":                 null,
  "lot_ids":             null,
  "extents":             null,
  "db_threshold":        65.0,
  "non_building_json":   null,
  "lease_plan_b64":      null,
  "detect_entry_points": false
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `data_type` | string | ✅ | — | `LOT`, `STT`, `GLA`, `LPP`, `ADDRESS`, etc. Case-insensitive. |
| `value` | string | ✅ | — | Identifier value, e.g. `"IL 1657"`, `"CPTL 16"` |
| `lon` | float | ADDRESS only | `null` | Pre-resolved WGS84 longitude |
| `lat` | float | ADDRESS only | `null` | Pre-resolved WGS84 latitude |
| `lot_ids` | string[] | ○ | `[]` | Multi-lot identifiers |
| `extents` | object[] | ○ | `[]` | Multi-lot EPSG:2326 bounding boxes `{xmin, ymin, xmax, ymax}` |
| `db_threshold` | float | ○ | `65.0` | Noise threshold in dBA (HK EPD daytime limit = 65.0) |
| `non_building_json` | object | ○ | `null` | Colour label definitions. Must be paired with `lease_plan_b64`. |
| `lease_plan_b64` | string | ○ | `null` | Base64-encoded lease plan (PDF, PNG, or JPEG) |
| `detect_entry_points` | bool | ○ | `false` | If `true` and `lease_plan_b64` provided, detects vehicle access points X/Y/Z |

### `non_building_json` schema

```json
{
  "color_labels": {
    "pink": {
      "height": null,
      "description": "Site area",
      "reference_clause": "PARTICULARS OF THE LOT"
    },
    "pink cross-hatched black": {
      "height": "5.1 metres",
      "description": "Drainage Reserve Area",
      "reference_clause": "Drainage Reserve Area"
    },
    "green": {
      "height": null,
      "description": "Future public roads (the Green Area)",
      "reference_clause": "Formation of the Green Area"
    }
  },
  "non_building_areas": [
    {
      "description": "Drainage Reserve Area",
      "location_ref": "shown coloured pink cross-hatched black and marked \"D.R.\" on the plan",
      "reference_clause": "Drainage Reserve Area"
    },
    {
      "description": "Future public roads",
      "location_ref": "shown coloured green on the plan annexed hereto",
      "reference_clause": "Formation of the Green Area"
    }
  ]
}
```

Only entries listed in `non_building_areas` are extracted from the lease plan image. `color_labels` provides the metadata (description, height, clause) that gets attached to each extracted zone in the response.

> **Important:** The pink area that represents the main buildable site should appear in `color_labels` (so the engine knows what pink means) but should NOT appear in `non_building_areas` (so it is not extracted as a restricted zone).

---

## Response Schema

### Basic Response

Returned when neither `lease_plan_b64`/`non_building_json` nor `detect_entry_points` is provided.

```json
{
  "site_id": "IL_1657",
  "crs": "EPSG:3857",
  "sampling_interval_m": 1.0,
  "boundary": {
    "x": [12700123.4, 12700124.3, 12700125.1],
    "y": [2560234.1, 2560235.0, 2560235.8]
  },
  "view_type": ["SEA", "CITY", "GREEN"],
  "noise_db": [62.3, 71.8, 58.4],
  "db_threshold": 65.0,
  "is_noisy": [false, true, false]
}
```

| Field | Type | Description |
|---|---|---|
| `site_id` | string | Normalised identifier: `value.upper().replace(" ", "_")` |
| `crs` | string | Always `"EPSG:3857"` (Web Mercator, metres) |
| `sampling_interval_m` | float | Always `1.0` |
| `boundary.x` | float[] | Easting coordinates of boundary points (metres, EPSG:3857) |
| `boundary.y` | float[] | Northing coordinates of boundary points (metres, EPSG:3857) |
| `view_type` | string[] | Per-point view label: `SEA`, `HARBOR`, `RESERVOIR`, `GREEN`, `PARK`, `CITY`, `MOUNTAIN` |
| `noise_db` | float[] | Per-point noise level in dBA, rounded to 1 decimal place |
| `db_threshold` | float | The threshold used to compute `is_noisy` |
| `is_noisy` | bool[] | `true` where `noise_db[i] >= db_threshold` |

All five arrays (`boundary.x`, `boundary.y`, `view_type`, `noise_db`, `is_noisy`) are guaranteed to have identical length equal to the number of 1-metre boundary samples.

---

### Extended Response — with Lease Plan

When both `non_building_json` and `lease_plan_b64` are provided, the response includes a `non_building_areas` key:

```json
{
  "site_id": "IL_1657",
  "crs": "EPSG:3857",
  "sampling_interval_m": 1.0,
  "boundary": { "x": [...], "y": [...] },
  "view_type": [...],
  "noise_db": [...],
  "db_threshold": 65.0,
  "is_noisy": [...],
  "non_building_areas": {
    "pink_cross_hatched_black": {
      "use": "Drainage Reserve Area",
      "reference_clause": "Drainage Reserve Area",
      "location_ref": "shown coloured pink cross-hatched black and marked \"D.R.\" on the plan",
      "height": "5.1 metres",
      "coordinates": {
        "x": [12700200.1, 12700210.4, 12700215.2, 12700205.8],
        "y": [2560300.0, 2560298.5, 2560295.1, 2560297.3]
      }
    },
    "green": {
      "use": "Future public roads (the Green Area)",
      "reference_clause": "Formation of the Green Area",
      "location_ref": "shown coloured green on the plan annexed hereto",
      "height": null,
      "coordinates": {
        "x": [12700220.0, 12700232.5, 12700228.1],
        "y": [2560310.0, 2560308.2, 2560305.6]
      }
    }
  }
}
```

`non_building_areas` keys are normalised colour labels: spaces and hyphens replaced with underscores, lowercased. Each zone has at least 3 coordinate points forming a closed polygon in EPSG:3857.

---

### Extended Response — with Entry Points

When `detect_entry_points: true` and `lease_plan_b64` is provided, the response includes an `entry_points` key:

```json
{
  "site_id": "GLA_DN_77",
  "crs": "EPSG:3857",
  "sampling_interval_m": 1.0,
  "boundary": { "x": [...], "y": [...] },
  "view_type": [...],
  "noise_db": [...],
  "db_threshold": 65.0,
  "is_noisy": [...],
  "entry_points": {
    "crs": "EPSG:3857",
    "gap_count": 2,
    "gaps": [
      { "gap_index": 0, "length_pts": 61, "labels": ["X", "Y", "Z"] },
      { "gap_index": 1, "length_pts": 4,  "labels": ["A"] }
    ],
    "entry_points": [
      { "label": "X", "pixel_x": 327, "pixel_y": 168, "geo_x": 12700201.5, "geo_y": 2560301.2 },
      { "label": "Y", "pixel_x": 310, "pixel_y": 188, "geo_x": 12700199.3, "geo_y": 2560298.8 },
      { "label": "Z", "pixel_x": 301, "pixel_y": 213, "geo_x": 12700198.1, "geo_y": 2560295.4 },
      { "label": "A", "pixel_x": 463, "pixel_y": 220, "geo_x": 12700215.0, "geo_y": 2560294.5 }
    ]
  }
}
```

---

## DXF Output Specification

The DXF file produced by `/site-intelligence-dxf` conforms to AutoCAD R2010 (AC1024) format.

### Document units

```
$INSUNITS = 6    (metres)
$LUNITS   = 2    (decimal)
$LUPREC   = 4    (4 decimal places)
```

All coordinates are in EPSG:3857 (metric Web Mercator). When importing into QGIS, set the layer CRS to EPSG:3857.

### Layer Definitions

| Layer | ACI Colour | Linetype | Stride | Content |
|---|---|---|---|---|
| `SITE_BOUNDARY` | 7 (white/black) | CONTINUOUS | — | 1 closed LWPOLYLINE of all boundary points |
| `VIEW_POINTS` | 3 (green) | CONTINUOUS | Every 5th pt | POINT entities + TEXT labels |
| `NOISE_POINTS` | 1 (red) | CONTINUOUS | Every 5th pt | POINT entities + TEXT labels |
| `NON_BUILDING` | 5 (blue) | DASHED | — | 1 closed LWPOLYLINE per extracted zone + TEXT centroid label |
| `ENTRY_POINTS` | 6 (magenta) | CONTINUOUS | — | POINT + CIRCLE per access point + TEXT label |
| `LABELS` | 2 (yellow) | CONTINUOUS | — | All TEXT entities + title block |

### Per-point Colour Mapping

**VIEW_POINTS** entities use ACI colour override:

| View Type | ACI Colour | Appearance |
|---|---|---|
| `SEA` / `HARBOR` / `RESERVOIR` | 4 | Cyan |
| `GREEN` / `PARK` | 3 | Green |
| `CITY` | 1 | Red |
| `OPEN` | 2 | Yellow |
| `MOUNTAIN` | 8 | Grey |

**NOISE_POINTS** entities:

| Condition | ACI Colour | Appearance |
|---|---|---|
| `is_noisy = true` | 1 | Red |
| `is_noisy = false` | 4 | Cyan |

### Title Block

A minimal title block is written below the boundary drawing on the `LABELS` layer. Contents: site ID, CRS, sampling interval, noise threshold, and boundary point count. Text height and offset scale proportionally from the site bounding box short-side dimension.

### Importing into CAD Software

**AutoCAD / BricsCAD:**
```
File → Open → select .dxf → confirm units: metres
```

**Rhino:**
```
File → Import → DXF/DWG → set import units to metres
```

**QGIS:**
```
Layer → Add Layer → Add Vector Layer → select .dxf
When prompted for CRS, enter: EPSG:3857
```

---

## Algorithms

### Boundary Densification

Given a lot polygon in EPSG:3857, the boundary is densified by walking the exterior ring at constant 1-metre arc-length intervals:

```python
exterior  = polygon.exterior          # Shapely LinearRing
length    = exterior.length           # perimeter in metres (metric CRS)
n         = int(floor(length / 1.0))  # number of sample points
distances = linspace(0.0, length - 1.0, n)

for d in distances:
    pt = exterior.interpolate(d)      # point at arc distance d
    xs.append(round(float(pt.x), 4))
    ys.append(round(float(pt.y), 4))
```

The last point is offset by 1m from the polygon closure to avoid exact duplication of the first point. For `MultiPolygon` inputs, the largest sub-polygon by area is used.

### View Classification

**Stage 1 — Feature fetch** (once per request):

Three OSMnx fetches run concurrently via `ThreadPoolExecutor(max_workers=3)` within a 300m radius of the site centroid. Each fetch has an individual 45-second timeout; a failed fetch falls back to an empty GeoDataFrame so the pipeline continues with degraded but valid data.

**Stage 2 — Per-point classification:**

For each boundary point, a 200m radius analysis buffer is constructed. `view.py` divides the 360° horizon into 20° wedges (18 sectors) and scores each:

```
For each 20° wedge:
  green_share = parks.intersection(wedge).area / wedge.area
  water_share = water.intersection(wedge).area / wedge.area
  city_score  = height_norm × density_norm
    where height_norm = max(building_heights_in_wedge) / h_ref
          density_norm = count(buildings_in_wedge) / max_buildings

  Dominant label = argmax(city_score, water_share, green_share, open_score)
  Priority: CITY > WATER (if > 2%) > GREEN (if > 2%) > OPEN
```

**Performance scaling:**
- ≤500 boundary points → direct classification (one call per point)
- >500 boundary points → classify a 10m grid covering the site extent, then assign each boundary point via nearest-neighbour lookup into the grid

### Noise Sampling

**Full pipeline** (when CSDI WFS is available):

The `noise.py` pipeline produces a 5m-resolution noise grid over the study area (150m radius). Each boundary point is sampled from this grid:

```python
xi = argmin(abs(X[0, :] - point_x))   # nearest grid column
yi = argmin(abs(Y[:, 0] - point_y))   # nearest grid row
db = noise_grid[yi, xi]
```

**Fallback** (when WFS pipeline fails):

```
For each road segment s within 300m:
  d_s = perpendicular distance from boundary point to segment s
  L_s = L_road_class(s) − 20·log₁₀(d_s + 1)

Total at boundary point:
  L_total = 10·log₁₀(Σ_s  10^(L_s / 10))
```

Vectorised across all boundary points × road segments using NumPy broadcasting. Chunk size: 200 boundary points.

### Lease Plan Colour Segmentation

The pixel-to-geographic coordinate mapping assumes a **north-up** image orientation with linear scale aligned to the site bounding box:

```
geo_x = site_bbox.minx + (col / img_width)  × (site_bbox.maxx − site_bbox.minx)
geo_y = site_bbox.maxy − (row / img_height) × (site_bbox.maxy − site_bbox.miny)
```

Contours are simplified using Douglas-Peucker with ε = 0.002 × contour arc length. Contours with area < 200 pixels are discarded as noise. The largest contour per colour is selected as the representative zone polygon.

### Vehicle Entry Point Detection

The detector works on the principle that vehicle access points are where the green verge strip is absent from the site boundary:

1. **Build site mask** — HSV-segment green verge (`H∈[25,45]`, `S∈[50,160]`, `V∈[140,220]`) and pink site area separately; merge; fill with `MORPH_CLOSE(20×20)` to bridge small gaps
2. **Extract outer contour** — `findContours(RETR_EXTERNAL)` on the merged filled mask; take the largest contour by area
3. **Probe for green** — for each of the N contour boundary points, sample a `±12px` neighbourhood patch in the green-only mask; `sum(patch > 0) < 10` → no green present
4. **Find gaps** — contiguous runs of "no green" points; filter `3 ≤ length ≤ 100` contour points
5. **Label sub-points** — evenly space `points_per_gap` samples within each gap; assign sequential labels (default: X, Y, Z, A, B, C…)
6. **Convert to world** — `_pixel_to_geo()` maps each pixel coordinate to EPSG:3857

---

## JSON Serialisation — `_compact_json()`

Flask's default `jsonify()` has two issues for this API:

1. **Whole-number floats lose their decimal point** — `65.0` becomes `65`, `1.0` becomes `1`. The spec requires `65.0` and `1.0`.
2. **Arrays are formatted vertically** — each element on its own indented line. The spec shows arrays inline.

`_compact_json()` solves both with a two-phase approach:

**Phase 1 — Protect float precision:**

```python
class _RawFloat:
    def __init__(self, s): self.s = s   # holds repr(float), e.g. "65.0", "834568.1"

def _fix(o):
    if isinstance(o, bool):  return o           # bool before float check (bool IS a float)
    if isinstance(o, float):
        s = repr(o)                             # full precision, e.g. "834568.1" not "834568"
        return _RawFloat(s if '.' in s else s + '.0')  # ensure decimal point
    if isinstance(o, dict):  return {k: _fix(v) for k, v in o.items()}
    if isinstance(o, list):  return [_fix(v)    for v in o]
    return o
```

Float values are replaced with `_RawFloat` sentinels, serialised as quoted strings by `json.dumps`, then the quotes are stripped in a post-processing step.

**Phase 2 — Collapse arrays:**

```python
pattern = re.compile(r'\[(?:[^\[\]{}])*\]', re.DOTALL)
while prev != raw:
    prev = raw
    raw = pattern.sub(lambda m: re.sub(r'\s+', ' ', m.group(0)).strip(), raw)
```

The regex iteratively collapses innermost arrays (no nested `[` or `{`) onto a single line. Multiple passes handle nested structures like `boundary.x` inside `boundary` inside the root dict.

**Output example:**

```json
{
  "site_id": "IL_1657",
  "crs": "EPSG:3857",
  "sampling_interval_m": 1.0,
  "boundary": {
    "x": [12706931.8001, 12706932.4437, 12706933.0873],
    "y": [2545615.8584, 2545616.6359, 2545617.4134]
  },
  "view_type": ["CITY", "CITY", "CITY"],
  "noise_db": [52.5, 52.5, 52.2],
  "db_threshold": 65.0,
  "is_noisy": [false, false, false]
}
```

---

## Caching

Responses are cached in-process in a Python dict (`CACHE_STORE`):

```
Cache key = MD5( data_type.upper() + "_" + value + "_" + str(db_threshold) )
```

| Condition | Cached | Reason |
|---|---|---|
| Standard request (no lease plan, no entry points) | ✅ | Deterministic for given inputs |
| Request with `lease_plan_b64` | ❌ | File content can differ between calls |
| Request with `detect_entry_points: true` | ❌ | Depends on lease plan content |
| Different `db_threshold` for same site | ❌ | Different cache key |

The cache is in-memory only. It is cleared on server restart. On Render free tier, the server sleeps after 15 minutes of inactivity and the cache is lost on wake.

For multi-worker deployments, the in-memory cache is not shared across processes. Replace `CACHE_STORE` with a Redis-backed cache using `flask-caching` if running multiple workers.

---

## Deployment

### Local Development

```bash
# 1. Clone and set up environment
git clone https://github.com/your-org/alkf-master-land-plan.git
cd alkf-master-land-plan
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Copy required files from alkf-site-analysis
cp ../alkf-site-analysis/modules/resolver.py  ./modules/
cp ../alkf-site-analysis/modules/view.py       ./modules/
cp ../alkf-site-analysis/modules/noise.py      ./modules/
cp ../alkf-site-analysis/data/BUILDINGS_FINAL.gpkg  ./data/

# 4a. Run with Flask dev server (auto-reload)
python app.py

# 4b. Run with Gunicorn matching production config
gunicorn app:app \
  --bind 0.0.0.0:10000 \
  --worker-class gthread \
  --workers 1 \
  --threads 4 \
  --timeout 300 \
  --keep-alive 5
```

Flask dev server: `http://localhost:10000`

### Render Cloud

`render.yaml` is pre-configured:

```yaml
services:
  - type: web
    name: alkf-master-land-plan
    env: python
    region: singapore
    pythonVersion: "3.11.4"
    buildCommand: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 4 --timeout 300 --keep-alive 5
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.4"
    autoDeploy: true
```

**Deploy steps:**

1. Push repository to GitHub (include `data/BUILDINGS_FINAL.gpkg` and all three copied module files)
2. Connect the repository to Render dashboard
3. Render auto-detects `render.yaml` and deploys

> **Note:** `BUILDINGS_FINAL.gpkg` is 64MB. If it exceeds GitHub's 100MB file size limit, use Git LFS or upload via Render's persistent disk feature.

**Timing expectations (Render free tier):**

| Scenario | Time |
|---|---|
| Cold start (server wakes after sleep) | 30–90 seconds |
| First analysis request (no cache) | 25–65 seconds |
| Cached request (same site + threshold) | < 2 seconds |
| Server sleep timeout | 15 minutes of inactivity |

### Worker Strategy — gthread vs gevent

The original v1.1 deployment used Gunicorn's default `sync` worker, which blocked the entire process for 25–65 seconds while OSMnx and WFS HTTP calls completed. Render's proxy has a 30-second idle timeout, causing "Failed to fetch" errors in the frontend.

**Why not gevent?** gevent patches Python's asyncio event loop. This works on Python 3.11, but on Python 3.12+ (including 3.14 which Render deployed despite `runtime.txt`) the patch causes `RuntimeError: loop is not the running loop`. gevent does not yet have stable support for Python 3.12+.

**gthread** uses real OS threads. `--threads 4` means 4 concurrent requests can be handled simultaneously. Each thread blocks on I/O independently — while one thread waits on OSMnx, the other 3 can serve health checks or cached responses. No monkey-patching required; no asyncio conflicts.

```
Gunicorn gthread model:

  gunicorn master
      └── worker (1 process)
              ├── thread 1 → long-running analysis (OSMnx + WFS = 40s)
              ├── thread 2 → cached response (< 2s)
              ├── thread 3 → health check GET / (< 5ms)
              └── thread 4 → available
```

---

## Dependencies

```
# Build system — provides pkg_resources (removed from stdlib in Python 3.12+)
setuptools

# API framework
flask
flask-cors
gunicorn

# Geospatial core
geopandas
shapely
pyproj
fiona
osmnx

# Numerical
numpy
scipy
pandas
networkx

# HTTP (resolver API calls to HK GeoData + iC1000)
requests

# Visualisation — required by view.py and noise.py internals
# These are lazy-loaded — they do not affect startup time
matplotlib
contextily
Pillow
scikit-learn

# DXF export
ezdxf

# Lease plan parsing
opencv-python-headless
pdf2image
```

All versions are unpinned — pip resolves compatible wheels for the active Python runtime (3.11.4). The geospatial stack (`shapely`, `fiona`, `geopandas`, `pyproj`) requires prebuilt C-extension wheels that exist on PyPI for Python 3.8–3.12. Python 3.13 and 3.14 do not yet have complete wheel coverage for this stack.

---

## Environment Notes

### Python version

Python **3.11.4** is required and pinned in three places:

1. `runtime.txt`: `python-3.11.4`
2. `render.yaml` env var: `PYTHON_VERSION: "3.11.4"`
3. `render.yaml` service field: `pythonVersion: "3.11.4"`

The triple-pin is necessary because Render ignores `runtime.txt` and `PYTHON_VERSION` independently, and was deploying Python 3.14.3 in testing. Only `pythonVersion` in the service block is reliably respected.

### pkg_resources

`pkg_resources` was removed from the Python standard library in Python 3.12. It is provided by `setuptools`, which is listed as an explicit top-level dependency to ensure it is always installed.

### Poppler (PDF lease plans only)

`pdf2image` requires `poppler-utils` to rasterise PDF files to images for OpenCV processing. PNG and JPEG lease plan inputs do not require Poppler and work on all platforms without additional system dependencies.

### OSMnx caching

`ox.settings.use_cache = True` is set in `spatial_intelligence.py`. OSMnx caches HTTP responses in `~/.cache/osmnx/`. On Render, this cache is lost on server restart/sleep but significantly speeds up repeated requests to the same geographic area during a single session.

---

## Testing

A complete Google Colab test notebook is provided: `ALKF_MLP_API_Test.ipynb`.

Set `BASE_URL` in cell 2 before running:

```python
BASE_URL = "https://alkf-master-land-plan-api.onrender.com"
```

| Test | What it validates |
|---|---|
| Health check | `GET /` returns 200, `status: "operational"` |
| LOT → JSON | Full basic output, all array fields present |
| LOT → DXF | File downloads, valid `AC1024` DXF header |
| ADDRESS type | Pre-resolved coordinate path works |
| Custom threshold | `db_threshold=55.0` applied, reflected in `is_noisy` |
| Float formatting | `sampling_interval_m` is `1.0` not `1`; `db_threshold` is `65.0` not `65` |
| Array format | Arrays are inline `[x, y, z]` not vertical |
| Array length consistency | `boundary.x`, `view_type`, `noise_db`, `is_noisy` all identical length |
| Coordinate sanity | All EPSG:3857 points within HK bounding box |
| View type labels | Only valid labels returned (`SEA`, `HARBOR`, `RESERVOIR`, `GREEN`, `PARK`, `CITY`, `MOUNTAIN`) |
| Noise value range | All values in [20, 120] dB; `is_noisy` consistent with threshold |
| Cache verification | Second call to same site+threshold faster than first |
| Error body key | Failed requests return `{"error": "…"}` not `{"detail": "…"}` |
| Error — ADDRESS without coords | Returns `422` with `error` key |
| Error — invalid LOT | Returns `422` or `500` with `error` key |
| Non-building JSON without lease plan | `non_building_areas` absent from response |
| Full lease plan extraction | `non_building_areas` present, each zone ≥3 coordinate points |
| Entry point detection | `entry_points` present with X/Y/Z labels, `gap_count` ≥ 1 |
| DXF layer verification | Open in QGIS, confirm SITE_BOUNDARY, VIEW_POINTS, NOISE_POINTS layers exist |

---

## Known Limitations

**Noise model accuracy**
The noise propagation model is a near-field screening model based on EPD empirical formulae. It is not ISO 9613-2 compliant and has not been calibrated against field measurements. It should not be used for formal Environmental Impact Assessment submissions.

**View classification radius**
Each boundary point uses a fixed 200m analysis radius. For very large sites (perimeter > 2000m), view classifications at inner boundary segments may be influenced by features that are functionally further away in the urban fabric.

**Lease plan coordinate alignment**
`_pixel_to_geo()` assumes the lease plan image is a north-up, scale-correct orthographic projection aligned exactly with the site bounding box. Scanned lease plans that are rotated, skewed, or at an oblique angle will produce incorrect zone coordinates. The current version does not perform image registration or rotation correction.

**Entry point detection accuracy**
The green verge gap detection method works well for standard HK Government Lands Department lease plan formats where the verge is a consistent HSV colour. Photocopied, faded, or non-standard lease plans may produce missed or false detections.

**In-memory cache**
The cache is not shared across worker processes. The single-worker Gunicorn configuration (`--workers 1`) ensures all requests hit the same cache. For multi-worker deployments, replace `CACHE_STORE` with a shared Redis cache.

**CSDI WFS availability**
Both the ATC traffic census (`atc_wfs_url`) and LNRS road surface data (`lnrs_wfs_url`) are fetched from CSDI HK government WFS endpoints. These services have intermittent downtime windows. When unavailable, the noise module automatically falls back to the road-class attenuation model with no manual intervention required.

**Render free tier cold starts**
The server sleeps after 15 minutes of inactivity. The first request after a cold start triggers `BUILDINGS_FINAL.gpkg` reload (~2s) plus OSMnx fetch (~30s) plus noise pipeline (~20s) — expect 45–90 second response times on cold start. Subsequent requests to cached sites return in < 2 seconds.

---

## Bug Fixes in v1.2

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `app.py` | Duplicate `import geopandas as gpd` (lines 14 and 20) | Removed second import |
| 2 | `app.py` + `data/` | `BUILDINGS_FINAL .gpkg` had trailing space in filename | Renamed file; updated reference |
| 3 | `modules/` | `__init__.py` missing — package imports unreliable | Created `modules/__init__.py` |
| 4 | `spatial_intelligence.py` | `concurrent.futures` imported 3× (module level + 2 lines inside function) | Collapsed to 1 lazy import inside `_fetch_view_features` |
| 5 | `dxf_export.py` | `raw` variable assigned inside `try` but referenced in `log.info()` after `finally` → `NameError` | Replaced tempfile round-trip with `StringIO().encode()` |
| 6 | `dxf_export.py` | `_write_title_block` function body present but `def` statement accidentally dropped | Restored `def _write_title_block(msp, site_id, intelligence, xs, ys, n):` |
| 7 | `app.py` | FastAPI → Flask: error body key was `detail`, frontend expected `detail` | Changed to `error`; updated frontend fetch error handlers |
| 8 | `app.py` | `jsonify()` drops `.0` from whole-number floats (`65.0` → `65`) | Added `_compact_json()` with `_RawFloat` sentinel + regex array collapse |
| 9 | `render.yaml` | `runtime.txt` and `PYTHON_VERSION` env var ignored by Render → deployed Python 3.14.3 | Added `pythonVersion: "3.11.4"` service field (the one Render actually reads) |
| 10 | `render.yaml` | gevent worker crashes on Python 3.12+ (`RuntimeError: loop is not the running loop`) | Switched to `--worker-class gthread --threads 4` |

---

## Relationship to alkf-site-analysis

| | `alkf-site-analysis` | `alkf-master-land-plan` |
|---|---|---|
| **Purpose** | Site feasibility maps + PDF report | Boundary-sampled JSON / DXF dataset |
| **Output** | PNG maps, PDF report | JSON array data, DXF CAD file |
| **Spatial unit** | Whole-site analysis, single output per module | Per-point data at 1m intervals along boundary |
| **View module** | Renders polar diagram image | Uses internal functions only — no image produced |
| **Noise module** | Renders heatmap overlay image | Samples grid values per boundary point |
| **Zoning data** | `ZONE_REDUCED.gpkg` loaded and used | Not loaded — zoning not part of MLP output |
| **New modules** | — | `spatial_intelligence.py`, `dxf_export.py`, `lease_plan_parser.py`, `entry_point_detector.py` |
| **Shared modules** | `resolver.py`, `view.py`, `noise.py` | Same files (copied, not modified) |
| **Framework** | FastAPI + uvicorn | Flask + Gunicorn gthread |

The two APIs are designed to be used together: `alkf-site-analysis` provides the analyst with visual feasibility maps; `alkf-master-land-plan` provides the architect or planner with structured data and CAD-ready geometry for the next stage of design.

---

## Changelog

### v1.2 (current)

- **Framework migration:** FastAPI + uvicorn → Flask + Gunicorn `gthread` worker
- **New module:** `entry_point_detector.py` — vehicle access point (X/Y/Z) detection from lease plan green verge gaps
- **New DXF layer:** `ENTRY_POINTS` (magenta, ACI 6) — POINT + CIRCLE per access point
- **New request field:** `detect_entry_points: bool` — activates Step 11 in the pipeline
- **New serialiser:** `_compact_json()` — inline arrays + float decimal preservation
- **Error body key changed:** `{"detail": "…"}` → `{"error": "…"}` (only breaking change)
- **Python version enforcement:** Added `pythonVersion: "3.11.4"` to `render.yaml` (Render was ignoring `runtime.txt` and `PYTHON_VERSION`, deploying Python 3.14)
- **Gunicorn worker:** `sync` → `gthread` (fixes 30s Render proxy timeout; avoids gevent Python 3.14 crash)
- **Fixed:** duplicate `import geopandas` in `app.py`
- **Fixed:** trailing space in `BUILDINGS_FINAL .gpkg` filename
- **Fixed:** missing `modules/__init__.py`
- **Fixed:** `concurrent.futures` triple-import in `spatial_intelligence.py`
- **Fixed:** `_write_title_block` missing `def` statement in `dxf_export.py`
- **Fixed:** `dxf_export.py` tempfile `NameError` — replaced with `StringIO`
- **Fixed:** DXF title block floating over drawing in `DXFViewer.jsx` — moved to fixed HTML panel
- **Fixed:** lease plan preview blank in Firefox/Chrome — replaced `data:` URL with `Blob URL` for iframe

### v1.1

- Replaced top-level imports from `view.py` and `noise.py` with lazy imports inside consuming functions — eliminates `ModuleNotFoundError: No module named 'contextily'` on startup
- Added `matplotlib`, `contextily`, `Pillow`, `scikit-learn` to `requirements.txt`
- Updated `render.yaml` build command to upgrade `pip`, `setuptools`, and `wheel` before install
- Pinned to Python 3.11.4 via `runtime.txt` and `PYTHON_VERSION` env var

### v1.0

Initial release — boundary densification, view classification, noise sampling, DXF export, lease plan colour segmentation.
