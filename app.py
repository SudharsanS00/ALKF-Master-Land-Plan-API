# ============================================================
# app.py
# ALKF Master Land Plan API  v1.2  (Flask)
# ============================================================

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from io import BytesIO
import geopandas as gpd
import os
import time
import logging
import hashlib
import json

from modules.spatial_intelligence import generate_site_intelligence
from modules.dxf_export            import export_dxf

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)


# ── JSON serialiser — preserves float formatting (65 → 65.0) ─
class _FloatEncoder(json.JSONEncoder):
    """
    Standard json.JSONEncoder drops the decimal on whole-number floats
    (65.0 → 65, 1.0 → 1).  This subclass forces at least one decimal
    place so the output always matches the spec (65.0, 1.0).
    """
    def iterencode(self, o, _one_shot=False):
        # Replace every Python float that is a whole number with a
        # string representation that includes ".0", then let the
        # standard encoder handle everything else.
        return super().iterencode(self._fix(o), _one_shot)

    def _fix(self, obj):
        if isinstance(obj, float):
            # Format as "65.0", "1.0", "62.3" etc.
            return _FloatMarker(obj)
        if isinstance(obj, dict):
            return {k: self._fix(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._fix(v) for v in obj]
        return obj


class _FloatMarker(float):
    """Thin float subclass so we can override __repr__ for the encoder."""
    def __repr__(self):
        # Always emit at least one decimal place
        s = format(float(self), 'g')
        return s if '.' in s or 'e' in s else s + '.0'


def _json_response(data, status=200):
    """Drop-in replacement for jsonify() that preserves float decimals."""
    body = json.dumps(data, cls=_FloatEncoder, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")

# ── Cache ─────────────────────────────────────────────────────
CACHE_STORE: dict = {}

def make_cache_key(data_type: str, value: str, db_threshold: float) -> str:
    raw = f"{data_type.upper()}_{value}_{db_threshold}"
    return hashlib.md5(raw.encode()).hexdigest()

# ── Static data (loaded once at import time) ──────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

log.info("Loading building height dataset...")
BUILDING_DATA = gpd.read_file(
    os.path.join(DATA_DIR, "BUILDINGS_FINAL .gpkg")
).to_crs(3857)
if "HEIGHT_M" not in BUILDING_DATA.columns:
    raise RuntimeError(
        f"HEIGHT_M column missing. Available: {list(BUILDING_DATA.columns)}"
    )
BUILDING_DATA = BUILDING_DATA[BUILDING_DATA["HEIGHT_M"] > 5].copy()
log.info(f"Building data loaded: {len(BUILDING_DATA):,} rows")
log.info("Startup complete.")


# ── Helpers ───────────────────────────────────────────────────

def _parse_body():
    """
    Parse the incoming JSON body.
    Returns (data, None) on success or (None, error_message) on failure.
    """
    if not request.is_json:
        return None, "Request Content-Type must be application/json"
    data = request.get_json(silent=True)
    if data is None:
        return None, "Invalid or empty JSON body"
    return data, None


def _normalise_request(data: dict):
    """
    Validate and normalise the parsed JSON body.
    Returns (dt, value, lon, lat, lot_ids, extents, threshold,
             non_building_json, lease_plan_b64, detect_entry_points).
    Raises ValueError on bad input.
    """
    data_type = data.get("data_type")
    value     = data.get("value")

    if not data_type:
        raise ValueError("'data_type' is required")
    if not value:
        raise ValueError("'value' is required")

    dt  = data_type.upper()
    lon = data.get("lon")
    lat = data.get("lat")

    if dt == "ADDRESS" and (lon is None or lat is None):
        raise ValueError("ADDRESS type requires pre-resolved lon and lat")

    lon = float(lon) if lon is not None else None
    lat = float(lat) if lat is not None else None

    raw_threshold = data.get("db_threshold")
    threshold = float(raw_threshold) if raw_threshold is not None else 65.0

    lot_ids = data.get("lot_ids") or []
    extents = data.get("extents") or []

    if not isinstance(lot_ids, list):
        raise ValueError("'lot_ids' must be a list")
    if not isinstance(extents, list):
        raise ValueError("'extents' must be a list")

    non_building_json   = data.get("non_building_json") or None
    lease_plan_b64      = data.get("lease_plan_b64")    or None
    detect_entry_points = bool(data.get("detect_entry_points", False))

    return (dt, value, lon, lat, lot_ids, extents, threshold,
            non_building_json, lease_plan_b64, detect_entry_points)


def _err(status: int, message: str):
    return jsonify({"error": message}), status


# ── GET / — health check ──────────────────────────────────────

@app.get("/")
def health():
    return _json_response({
        "service": "ALKF Master Land Plan API",
        "version": "1.2",
        "status":  "operational",
    })


# ── POST /site-intelligence ───────────────────────────────────

@app.post("/site-intelligence")
def site_intelligence():
    """
    Returns a structured JSON dataset describing the site boundary
    sampled at 1-metre intervals with:
      - view_type           : per-point view classification label
      - noise_db            : per-point noise level (dBA)
      - is_noisy            : boolean array (noise_db >= db_threshold)
      - non_building_areas  : (optional) colour-segmented zones from lease plan
      - entry_points        : (optional) vehicle access points X/Y/Z from lease plan
    """
    data, err = _parse_body()
    if err:
        return _err(400, err)

    try:
        (dt, v, lon, lat, lot_ids, extents, threshold,
         non_building_json, lease_plan_b64, detect_entry_points) = _normalise_request(data)
    except ValueError as e:
        return _err(422, str(e))

    log.info(
        f"[site-intelligence] {dt} {v}  threshold={threshold} dB"
        f"  entry_points={detect_entry_points}"
    )
    start = time.time()

    # Cache: skip if lease plan or entry-point detection is requested
    cache_key = None
    if lease_plan_b64 is None and not detect_entry_points:
        cache_key = make_cache_key(dt, v, threshold)
        if cache_key in CACHE_STORE:
            log.info(f"  Cache hit: {cache_key}")
            return _json_response(CACHE_STORE[cache_key])

    try:
        result = generate_site_intelligence(
            data_type           = dt,
            value               = v,
            building_data       = BUILDING_DATA,
            lon                 = lon,
            lat                 = lat,
            lot_ids             = lot_ids,
            extents             = extents,
            db_threshold        = threshold,
            non_building_json   = non_building_json,
            lease_plan_b64      = lease_plan_b64,
            detect_entry_points = detect_entry_points,
        )
    except Exception as e:
        log.exception("site-intelligence failed")
        return _err(500, str(e))

    if cache_key:
        CACHE_STORE[cache_key] = result

    log.info(f"  Completed in {time.time() - start:.2f}s")
    return _json_response(result)


# ── POST /site-intelligence-dxf ───────────────────────────────

@app.post("/site-intelligence-dxf")
def site_intelligence_dxf():
    """
    Same computation as /site-intelligence but returns a DXF CAD file.

    Layers in the DXF:
      SITE_BOUNDARY  — densified boundary polyline
      VIEW_POINTS    — point entities coloured by view type
      NOISE_POINTS   — point entities coloured red (noisy) / cyan (quiet)
      NON_BUILDING   — closed polygons for non-buildable areas (optional)
      ENTRY_POINTS   — point entities for vehicle access points X/Y/Z (optional)
      LABELS         — text annotations and title block
    """
    data, err = _parse_body()
    if err:
        return _err(400, err)

    try:
        (dt, v, lon, lat, lot_ids, extents, threshold,
         non_building_json, lease_plan_b64, detect_entry_points) = _normalise_request(data)
    except ValueError as e:
        return _err(422, str(e))

    log.info(
        f"[site-intelligence-dxf] {dt} {v}  threshold={threshold} dB"
        f"  entry_points={detect_entry_points}"
    )
    start = time.time()

    try:
        result = generate_site_intelligence(
            data_type           = dt,
            value               = v,
            building_data       = BUILDING_DATA,
            lon                 = lon,
            lat                 = lat,
            lot_ids             = lot_ids,
            extents             = extents,
            db_threshold        = threshold,
            non_building_json   = non_building_json,
            lease_plan_b64      = lease_plan_b64,
            detect_entry_points = detect_entry_points,
        )
    except Exception as e:
        log.exception("site-intelligence-dxf failed at analysis stage")
        return _err(500, str(e))

    try:
        dxf_buf: BytesIO = export_dxf(result)
    except Exception as e:
        log.exception("site-intelligence-dxf failed at DXF export stage")
        return _err(500, f"DXF export error: {e}")

    site_id  = result.get("site_id", "site")
    filename = f"{site_id}_boundary_intelligence.dxf"

    log.info(f"  DXF completed in {time.time() - start:.2f}s  file={filename}")

    dxf_buf.seek(0)
    return Response(
        dxf_buf.read(),
        mimetype="application/dxf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
