# ============================================================
# app.py
# ALKF Master Land Plan API  v1.0
# ============================================================

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO
import geopandas as gpd
import os
import time
import logging
import hashlib
import json

import geopandas as gpd

from modules.spatial_intelligence import generate_site_intelligence
from modules.dxf_export            import export_dxf

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="ALKF Master Land Plan API",
    version="1.0",
    description="Boundary Intelligence Engine — site boundary sampled at 1m intervals with view classification, noise sampling, and optional lease plan non-building area extraction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache ─────────────────────────────────────────────────────
CACHE_STORE: dict = {}

def make_cache_key(data_type: str, value: str, db_threshold: float) -> str:
    raw = f"{data_type.upper()}_{value}_{db_threshold}"
    return hashlib.md5(raw.encode()).hexdigest()

# ── Static data ───────────────────────────────────────────────
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


# ── Request / Response models ─────────────────────────────────

class SiteIntelligenceRequest(BaseModel):
    data_type:        str
    value:            str
    lon:              Optional[float]      = None
    lat:              Optional[float]      = None
    lot_ids:          Optional[List[str]]  = None
    extents:          Optional[List[dict]] = None
    db_threshold:     Optional[float]      = None   # defaults to 65.0
    non_building_json: Optional[dict]      = None   # colour label + non_building_areas
    lease_plan_b64:   Optional[str]        = None   # base64-encoded PDF/image


# ── Helpers ───────────────────────────────────────────────────

def normalise_request(req: SiteIntelligenceRequest):
    dt  = req.data_type.upper()
    if dt == "ADDRESS" and (req.lon is None or req.lat is None):
        raise ValueError(
            "ADDRESS type requires pre-resolved lon/lat."
        )
    threshold = req.db_threshold if req.db_threshold is not None else 65.0
    lot_ids   = req.lot_ids  or []
    extents   = req.extents  or []
    return dt, req.value, req.lon, req.lat, lot_ids, extents, threshold


# ── Health check ──────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "service": "ALKF Master Land Plan API",
        "version": "1.0",
        "status":  "operational",
    }


# ── POST /site-intelligence ───────────────────────────────────

@app.post("/site-intelligence")
def site_intelligence(req: SiteIntelligenceRequest):
    """
    Returns a structured JSON dataset describing the site boundary
    sampled at 1-metre intervals with:
      - view_type  : per-point view classification label
      - noise_db   : per-point noise level (dBA)
      - is_noisy   : boolean array (noise_db >= db_threshold)
      - non_building_areas (optional) : extracted from lease plan
    """
    try:
        dt, v, lon, lat, lot_ids, extents, threshold = normalise_request(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    log.info(f"[site-intelligence] {dt} {v}  threshold={threshold} dB")
    start = time.time()

    # Cache check — lease plan requests are not cached (file content varies)
    cache_key = None
    if req.lease_plan_b64 is None:
        cache_key = make_cache_key(dt, v, threshold)
        if cache_key in CACHE_STORE:
            log.info(f"  Cache hit: {cache_key}")
            return JSONResponse(content=CACHE_STORE[cache_key])

    try:
        result = generate_site_intelligence(
            data_type        = dt,
            value            = v,
            building_data    = BUILDING_DATA,
            lon              = lon,
            lat              = lat,
            lot_ids          = lot_ids,
            extents          = extents,
            db_threshold     = threshold,
            non_building_json= req.non_building_json,
            lease_plan_b64   = req.lease_plan_b64,
        )
    except Exception as e:
        log.exception("site-intelligence failed")
        raise HTTPException(status_code=500, detail=str(e))

    if cache_key:
        CACHE_STORE[cache_key] = result

    log.info(f"  Completed in {time.time() - start:.2f}s")
    return JSONResponse(content=result)


# ── POST /site-intelligence-dxf ───────────────────────────────

@app.post("/site-intelligence-dxf")
def site_intelligence_dxf(req: SiteIntelligenceRequest):
    """
    Same computation as /site-intelligence but returns a DXF CAD file.

    Layers in the DXF:
      SITE_BOUNDARY  — densified boundary polyline
      VIEW_POINTS    — point entities with view label
      NOISE_POINTS   — point entities with dBA value
      NON_BUILDING   — closed polygons for non-buildable areas (optional)
    """
    try:
        dt, v, lon, lat, lot_ids, extents, threshold = normalise_request(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    log.info(f"[site-intelligence-dxf] {dt} {v}  threshold={threshold} dB")
    start = time.time()

    try:
        result = generate_site_intelligence(
            data_type        = dt,
            value            = v,
            building_data    = BUILDING_DATA,
            lon              = lon,
            lat              = lat,
            lot_ids          = lot_ids,
            extents          = extents,
            db_threshold     = threshold,
            non_building_json= req.non_building_json,
            lease_plan_b64   = req.lease_plan_b64,
        )
    except Exception as e:
        log.exception("site-intelligence-dxf failed at analysis stage")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        dxf_buf: BytesIO = export_dxf(result)
    except Exception as e:
        log.exception("site-intelligence-dxf failed at DXF export stage")
        raise HTTPException(status_code=500, detail=f"DXF export error: {e}")

    site_id = result.get("site_id", "site")
    filename = f"{site_id}_boundary_intelligence.dxf"

    log.info(f"  DXF completed in {time.time() - start:.2f}s  file={filename}")

    dxf_buf.seek(0)
    return StreamingResponse(
        dxf_buf,
        media_type="application/dxf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
