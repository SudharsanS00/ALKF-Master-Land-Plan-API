# ALKF Master Land Plan API

**Boundary Intelligence Engine** — Flask microservice that walks a site boundary at 1-metre intervals and records view classification and noise level at every point, returning a structured JSON dataset or a DXF CAD file ready for import into AutoCAD, Rhino, or QGIS.

Part of the **ALKF+ Automated Spatial Intelligence Platform**.

---

## Architecture

![ALKF Master Land Plan API Architecture](./architecture.svg)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Data Prerequisites](#data-prerequisites)
5. [Modules](#modules)
6. [API Reference](#api-reference)
7. [Request Model](#request-model)
8. [Response Schema](#response-schema)
9. [DXF Output Specification](#dxf-output-specification)
10. [Algorithms](#algorithms)
11. [Caching](#caching)
12. [Deployment](#deployment)
13. [Dependencies](#dependencies)
14. [Environment Notes](#environment-notes)
15. [Testing](#testing)
16. [Known Limitations](#known-limitations)
17. [Changelog](#changelog)

---

## Overview

Given any supported site identifier (lot number, address with coordinates, or CSUID), the Boundary Intelligence Engine:

1. Resolves the identifier to WGS84 coordinates via the HK GeoData API
2. Retrieves the official lot boundary polygon from the LandsD iC1000 API (GML → EPSG:3857)
3. Densifies the boundary exterior at **1-metre intervals** using Shapely interpolation
4. Classifies the dominant **view type** at each boundary point (`SEA`, `HARBOR`, `RESERVOIR`, `GREEN`, `PARK`, `CITY`) using the view sector model from `view.py`
5. Samples the **road traffic noise level** (dBA) at each boundary point using the full noise propagation pipeline from `noise.py`, with a vectorised fallback model if the WFS pipeline fails
6. Evaluates each point against a configurable **noise threshold** (default 65 dBA per HK EPD)
7. Optionally extracts **non-building zone polygons** from a lease plan image or PDF using OpenCV colour segmentation, and maps pixel coordinates to EPSG:3857
8. Returns either a structured **JSON dataset** or a **DXF CAD file** containing all of the above as named layers

---

## Architecture

```
Client Request
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Flask  (app.py)  —  served by Gunicorn gthread worker      │
│                                                             │
│  POST /site-intelligence        → jsonify(result)           │
│  POST /site-intelligence-dxf   → Response(.dxf binary)      │
│  GET  /                         → health check              │
│                                                             │
│  _parse_body()       — validates Content-Type + JSON        │
│  _normalise_request() — validates fields, sets defaults     │
│  _compact_json()     — inline-array float-preserving serial │
│                                                             │
│  In-memory cache (CACHE_STORE)                              │
│    key: MD5(data_type + value + db_threshold)               │
│    lease_plan / entry_point requests: never cached          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  generate_site_intelligence()  (spatial_intelligence.py)    │
│                                                             │
│  Step 1  resolve_location()     → (lon, lat) WGS84          │
│          get_lot_boundary()     → Polygon EPSG:3857         │
│          fallback 1: OSM building polygon nearest site      │
│          fallback 2: 40m circular buffer                    │
│                                                             │
│  Step 2  _densify_boundary()                                │
│          Shapely exterior.interpolate(d) every 1m           │
│          → (xs, ys) parallel lists, EPSG:3857               │
│                                                             │
│  Step 3  _fetch_view_features()                             │
│          OSMnx features_from_point(radius=300m)             │
│          buildings / parks / water  — concurrent threads    │
│                                                             │
│  Step 4  _batch_classify_views()                            │
│          ≤500 pts → direct per-point classification         │
│          >500 pts → 10m grid sample + NN assignment         │
│          calls view.py: _classify_sectors() _get_site_height│
│                                                             │
│  Step 5  _build_noise_grid()                                │
│          Full noise.py pipeline:                            │
│          ATCWFSLoader → TrafficAssigner → LNRSAssigner      │
│          → CanyonAssigner → EmissionEngine                  │
│          → PropagationEngine.run() → (X, Y, noise[i,j])     │
│          _sample_noise_at_points() — NN grid lookup         │
│          fallback: _fallback_noise_from_roads() vectorised  │
│                                                             │
│  Step 6  is_noisy = [v >= db_threshold for v in noise_db]   │
│                                                             │
│  Step 7  Assemble output dict                               │
│                                                             │
│  Step 10 (Optional) lease_plan_parser                       │
│          .extract_non_building_areas()                      │
│          → HSV segmentation → contours → EPSG:3857 coords   │
│                                                             │
│  Step 11 (Optional) entry_point_detector                    │
│          .extract_entry_points()                            │
│          → gap detection → X/Y/Z labels → EPSG:3857 coords  │
│                                                             │
│  → return dict (JSON-serialisable)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
        jsonify(result)       export_dxf()
        _compact_json()       ezdxf R2010
        inline arrays         → BytesIO → Response
```

### Static Data (Startup Preload)

At startup, `app.py` loads one dataset into memory:

| Dataset | File | Rows | Filtered to | Used by |
|---|---|---|---|---|
| Building heights | `data/BUILDINGS_FINAL.gpkg` | 42,073 | `HEIGHT_M > 5m` | `spatial_intelligence._batch_classify_views()` |

---

## Repository Structure

```
alkf-master-land-plan/
│
├── app.py                        # Flask application — endpoints, cache, startup
├── render.yaml                   # Render cloud deployment (Gunicorn gthread)
├── requirements.txt              # Python dependencies
├── runtime.txt                   # python-3.11.4
├── architecture.svg              # Architecture diagram (this file)
│
├── data/
│   └── BUILDINGS_FINAL.gpkg
│
└── modules/
    ├── __init__.py
    ├── spatial_intelligence.py   # Core pipeline orchestrator
    ├── dxf_export.py             # DXF CAD writer (ezdxf R2010)
    ├── lease_plan_parser.py      # OpenCV colour segmentation engine
    ├── entry_point_detector.py   # Vehicle entry point (X/Y/Z) detector
    ├── resolver.py               # Multi-type location resolver + iC1000 boundary API
    ├── view.py                   # 360° view sector classification engine
    └── noise.py                  # Road traffic noise propagation model
```

---

## Data Prerequisites

```bash
# Copy modules from alkf-site-analysis
cp ../alkf-site-analysis/modules/resolver.py  ./modules/
cp ../alkf-site-analysis/modules/view.py       ./modules/
cp ../alkf-site-analysis/modules/noise.py      ./modules/

# Copy data
cp ../alkf-site-analysis/data/BUILDINGS_FINAL.gpkg  ./data/

# (Optional) Poppler for PDF lease plans
apt-get install -y poppler-utils   # Ubuntu / Render
brew install poppler               # macOS
```

---

## Modules

### `app.py` — Flask application

Key differences from the original FastAPI version:

| Concern | FastAPI (old) | Flask (current) |
|---|---|---|
| Framework | `FastAPI()` + `CORSMiddleware` | `Flask(__name__)` + `flask_cors.CORS` |
| Request parsing | Pydantic `BaseModel` auto-validation | `_parse_body()` + `_normalise_request()` |
| JSON response | `JSONResponse(content=result)` | `_json_response(result)` via `_compact_json()` |
| DXF response | `StreamingResponse(buf, media_type=…)` | `Response(buf.read(), mimetype=…)` |
| Error response | `raise HTTPException(status_code=422)` | `return _err(422, message)` |
| Error body key | `{"detail": "…"}` | `{"error": "…"}` |
| Server | `uvicorn` | `gunicorn --worker-class gthread --threads 4` |

#### `_compact_json()`

Custom JSON serialiser that:
- Preserves float decimal points (`65.0` not `65`, `1.0` not `1`)
- Collapses all arrays onto a single line — output matches the spec exactly:

```json
{
  "site_id": "IL_1657",
  "sampling_interval_m": 1.0,
  "boundary": {
    "x": [12706931.8001, 12706932.4437, ...],
    "y": [2545615.8584, 2545616.6359, ...]
  },
  "view_type": ["CITY", "CITY", ...],
  "noise_db": [54.1, 54.1, ...],
  "db_threshold": 65.0,
  "is_noisy": [false, false, ...]
}
```

### `spatial_intelligence.py`

Core pipeline orchestrator. Public interface:

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

### `dxf_export.py`

Converts the site intelligence JSON dict into a DXF R2010 file using `ezdxf`.

- Uses `StringIO` + `.encode("utf-8")` — no temp files, no disk I/O
- Layers: `SITE_BOUNDARY`, `VIEW_POINTS`, `NOISE_POINTS`, `NON_BUILDING`, `ENTRY_POINTS`, `LABELS`
- Title block positioned at bottom-right, bbox-proportional text scaling

### `lease_plan_parser.py`

Extracts non-building zone polygons from a lease plan using OpenCV HSV colour segmentation.

### `entry_point_detector.py`

Detects vehicle access points (X, Y, Z …) from a lease plan image by finding gaps in the green verge strip along the site boundary contour.

```python
def extract_entry_points(
    image_bytes:    bytes,
    site_polygon:   Polygon,
    crs:            str  = "EPSG:3857",
    points_per_gap: int  = 3,
    label_names:    list = None,
) -> dict
```

Returns `{ crs, entry_points: [{label, pixel_x, pixel_y, geo_x, geo_y}], gap_count, gaps }`.

---

## API Reference

### Base URL

```
https://alkf-master-land-plan-api.onrender.com
```

### `GET /`

```json
{
  "service": "ALKF Master Land Plan API",
  "version": "1.2",
  "status":  "operational"
}
```

### `POST /site-intelligence`

Returns structured JSON. Content-Type must be `application/json`.

**Errors:**

| Code | Body key | Reason |
|---|---|---|
| `400` | `error` | Missing or non-JSON body |
| `422` | `error` | ADDRESS without lon/lat; missing required field |
| `500` | `error` | Pipeline failure |

### `POST /site-intelligence-dxf`

Identical computation, returns DXF binary.  
`Content-Disposition: attachment; filename="{site_id}_boundary_intelligence.dxf"`

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

| Field | Type | Required | Description |
|---|---|---|---|
| `data_type` | string | ✅ | `LOT`, `STT`, `GLA`, `LPP`, `ADDRESS`, etc. Case-insensitive. |
| `value` | string | ✅ | Identifier, e.g. `"IL 1657"` |
| `lon` | float | ADDRESS only | Pre-resolved WGS84 longitude |
| `lat` | float | ADDRESS only | Pre-resolved WGS84 latitude |
| `lot_ids` | string[] | ○ | Multi-lot identifiers |
| `extents` | object[] | ○ | Multi-lot EPSG:2326 bounding boxes |
| `db_threshold` | float | ○ | Noise threshold dBA. Default: `65.0` |
| `non_building_json` | object | ○ | Colour label definitions (with `lease_plan_b64`) |
| `lease_plan_b64` | string | ○ | Base64-encoded lease plan PDF/PNG/JPEG |
| `detect_entry_points` | bool | ○ | Detect X/Y/Z vehicle access points from lease plan |

---

## Response Schema

### Basic response

```json
{
  "site_id": "IL_1657",
  "crs": "EPSG:3857",
  "sampling_interval_m": 1.0,
  "boundary": {
    "x": [12700123.4, 12700124.3],
    "y": [2560234.1, 2560235.0]
  },
  "view_type": ["SEA", "CITY"],
  "noise_db": [62.3, 71.8],
  "db_threshold": 65.0,
  "is_noisy": [false, true]
}
```

### Extended response (with lease plan)

Adds `non_building_areas` and/or `entry_points`:

```json
{
  "...": "...",
  "non_building_areas": {
    "pink_cross_hatched_black": {
      "use": "Drainage Reserve Area",
      "reference_clause": "Drainage Reserve Area",
      "location_ref": "shown coloured pink cross-hatched black on the plan",
      "coordinates": { "x": [12700200.1, 12700210.4], "y": [2560300.0, 2560298.5] }
    }
  },
  "entry_points": {
    "crs": "EPSG:3857",
    "gap_count": 1,
    "gaps": [{ "gap_index": 0, "length_pts": 61, "labels": ["X","Y","Z"] }],
    "entry_points": [
      { "label": "X", "pixel_x": 327, "pixel_y": 168, "geo_x": 12700201.5, "geo_y": 2560301.2 },
      { "label": "Y", "pixel_x": 310, "pixel_y": 188, "geo_x": 12700199.3, "geo_y": 2560298.8 },
      { "label": "Z", "pixel_x": 301, "pixel_y": 213, "geo_x": 12700198.1, "geo_y": 2560295.4 }
    ]
  }
}
```

---

## DXF Output Specification

Format: AutoCAD R2010 (AC1024). Units: metres (`$INSUNITS = 6`). CRS: EPSG:3857.

| Layer | ACI | Content |
|---|---|---|
| `SITE_BOUNDARY` | 7 | Closed LWPOLYLINE of all boundary points |
| `VIEW_POINTS` | 3 | POINT entities every 5th point, coloured by view type |
| `NOISE_POINTS` | 1 | POINT entities every 5th point, red=noisy / cyan=quiet |
| `NON_BUILDING` | 5 | Closed LWPOLYLINE per extracted zone (dashed) |
| `ENTRY_POINTS` | 6 | POINT + CIRCLE per X/Y/Z access point (magenta) |
| `LABELS` | 2 | TEXT entities — zone names + title block |

---

## Algorithms

### Boundary Densification

```python
exterior  = polygon.exterior
length    = exterior.length           # perimeter in metres
n         = int(floor(length / 1.0))
distances = linspace(0.0, length - 1.0, n)
for d in distances:
    pt = exterior.interpolate(d)
    xs.append(pt.x); ys.append(pt.y)
```

### View Classification

OSMnx fetches buildings/parks/water within 300m. Each boundary point uses a 200m radius analysis circle. `view.py` divides the horizon into 20° wedges and scores GREEN / WATER / CITY / OPEN. Output is remapped: WATER→SEA/HARBOR/RESERVOIR by OSM tags.

### Noise Sampling

Full EPD HK empirical formula pipeline via `noise.py`. Falls back to road-class attenuation `L(r) = L_class − 20·log₁₀(r+1)` when CSDI WFS is unavailable.

### Vehicle Entry Point Detection (`entry_point_detector.py`)

1. HSV-segment the green verge strip from the lease plan image
2. Extract the full site outer contour (green + pink merged)
3. Walk the contour — runs of points with no green = access gaps
4. Subdivide each gap into `points_per_gap` sub-points, assign X/Y/Z labels
5. Convert pixel → EPSG:3857 via `_pixel_to_geo()`

---

## Caching

```
Cache key = MD5( data_type + "_" + value + "_" + db_threshold )
```

| Condition | Cached |
|---|---|
| Standard request (no lease plan, no entry points) | ✅ |
| Request with `lease_plan_b64` | ❌ |
| Request with `detect_entry_points: true` | ❌ |
| Different `db_threshold` for same site | ❌ (different key) |

---

## Deployment

### Local Development

```bash
git clone https://github.com/your-org/alkf-master-land-plan.git
cd alkf-master-land-plan
python -m venv .venv && source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Copy required files from alkf-site-analysis
cp ../alkf-site-analysis/modules/resolver.py ./modules/
cp ../alkf-site-analysis/modules/view.py     ./modules/
cp ../alkf-site-analysis/modules/noise.py    ./modules/
cp ../alkf-site-analysis/data/BUILDINGS_FINAL.gpkg ./data/

# Start with Flask dev server
python app.py

# Or with Gunicorn (matches production)
gunicorn app:app --bind 0.0.0.0:10000 --worker-class gthread --workers 1 --threads 4 --timeout 300
```

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

**Why `gthread` not `gevent`:** gevent is incompatible with Python 3.12+. The `gthread` worker uses OS threads — each of the 4 threads handles one concurrent request, so long-running OSMnx/WFS fetches don't block health checks or cached responses. `--timeout 300` allows up to 5 minutes per analysis request.

---

## Dependencies

```
# Build system
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

# HTTP
requests

# Visualisation (required by view.py and noise.py internals — lazy loaded)
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

---

## Environment Notes

**Python 3.11 required.** The geospatial stack (`shapely`, `fiona`, `geopandas`, `pyproj`) does not yet have complete wheel coverage for Python 3.13/3.14. `runtime.txt` and `pythonVersion` in `render.yaml` both pin `3.11.4`.

**`setuptools` explicit** — provides `pkg_resources` which was removed from stdlib in Python 3.12+.

**Poppler** — only required for PDF lease plans. PNG/JPEG inputs work without it.

---

## Testing

Test notebook: `ALKF_MLP_API_Test.ipynb`

```python
BASE_URL = "https://alkf-master-land-plan-api.onrender.com"
```

| Test | Validates |
|---|---|
| Health check | Status 200, correct keys |
| LOT → JSON | All array fields present, same length |
| LOT → DXF | Downloads, valid DXF header |
| ADDRESS type | Pre-resolved coordinate path |
| Custom threshold | `db_threshold=55.0` reflected in `is_noisy` |
| Float formatting | `sampling_interval_m` is `1.0` not `1` |
| Array format | Arrays inline, not vertical |
| Cache verification | Second call faster |
| Error body key | `{"error": "…"}` not `{"detail": "…"}` |
| ADDRESS without lon/lat | Returns `422` |
| Lease plan extraction | `non_building_areas` present, ≥3 pts per zone |
| Entry point detection | `entry_points` present with X/Y/Z labels |

---

## Known Limitations

**Noise model** — near-field screening model only. Not ISO 9613-2 compliant; not for EIA submissions.

**View radius** — fixed 200m per point. Very large sites may have cross-influenced boundary segments.

**Lease plan alignment** — `_pixel_to_geo()` assumes north-up, scale-correct image. Rotated or distorted scans produce incorrect coordinates.

**In-memory cache** — not shared across processes. Single-process Gunicorn deployment mitigates this. For multi-worker, use Redis.

**CSDI WFS** — government endpoints have intermittent downtime. Noise module falls back to road-class attenuation automatically.

**Render free tier** — server sleeps after 15 min inactivity; cold start takes 30–90 seconds.

---

## Changelog

### v1.2
- Migrated from FastAPI + uvicorn to **Flask + Gunicorn gthread**
- Added `_parse_body()` and `_normalise_request()` for explicit JSON validation
- Added `_compact_json()` — inline-array serialiser preserving float decimal points
- Fixed `modules/__init__.py` missing (package import reliability)
- Fixed duplicate `import geopandas` in `app.py`
- Fixed `BUILDINGS_FINAL .gpkg` trailing space in filename
- Fixed `concurrent.futures` triple-import in `spatial_intelligence.py`
- Fixed `dxf_export.py` `_write_title_block` missing `def` statement
- Fixed `dxf_export.py` tempfile NameError — replaced with `StringIO` + encode
- Added `entry_point_detector.py` — vehicle access point (X/Y/Z) detection
- Added `ENTRY_POINTS` DXF layer (magenta, ACI 6)
- Error response key changed: `detail` → `error`
- Switched to `pythonVersion: "3.11.4"` in `render.yaml` (Render ignored `runtime.txt`)
- Removed gevent (incompatible with Python 3.14); switched to `gthread` worker

### v1.1
- Replaced top-level imports from `view.py` and `noise.py` with lazy imports
- Added `matplotlib`, `contextily`, `Pillow`, `scikit-learn` to `requirements.txt`
- Upgraded build command in `render.yaml`
- Pinned Python 3.11.4

### v1.0
- Initial release — boundary densification, view classification, noise sampling, DXF export, lease plan colour segmentation
