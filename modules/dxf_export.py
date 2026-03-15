# ============================================================
# modules/dxf_export.py
# DXF CAD Export Engine  v1.2
# ALKF Master Land Plan API
#
# v1.2 fixes:
#   - Replaced tempfile round-trip with StringIO.write() — eliminates
#     NameError on 'raw' if the read-back failed, removes all disk I/O
#   - Removed unused os and tempfile imports
# v1.1 fixes:
#   - ASCII-safe strings (_ascii() sanitiser — no Unicode corruption)
#   - Label offsets scaled from site bounding box (not hardcoded 0.6m)
#   - Text height scaled from bbox short side
#   - Title block uses hyphen not em dash
#   - Title block positioned with bbox-proportional margin
#   - _label_scale() helper: readable on any lot size
# ============================================================

from __future__ import annotations

import logging
from io import BytesIO

import ezdxf

log = logging.getLogger(__name__)

_LAYERS = {
    "SITE_BOUNDARY": {"color": 7,  "linetype": "CONTINUOUS"},
    "VIEW_POINTS":   {"color": 3,  "linetype": "CONTINUOUS"},
    "NOISE_POINTS":  {"color": 1,  "linetype": "CONTINUOUS"},
    "NON_BUILDING":  {"color": 5,  "linetype": "DASHED"},
    "LABELS":        {"color": 2,  "linetype": "CONTINUOUS"},
}

_VIEW_COLOR = {
    "SEA": 4, "HARBOR": 4, "RESERVOIR": 4,
    "MOUNTAIN": 8,
    "PARK": 3, "GREEN": 3,
    "CITY": 1,
    "OPEN": 2,
}


def _ascii(s: str) -> str:
    return (
        s.replace("\u2014", "-").replace("\u2013", "-")
         .replace("\u00b0", "deg").replace("\u2265", ">=").replace("\u2264", "<=")
         .encode("ascii", errors="replace").decode("ascii")
    )


def _bbox(xs, ys):
    return min(xs), min(ys), max(xs), max(ys)


def _label_scale(xs, ys):
    xmin, ymin, xmax, ymax = _bbox(xs, ys)
    short = max(min(xmax - xmin, ymax - ymin), 5.0)
    return max(0.6, round(short * 0.06, 1)), max(1.0, round(short * 0.10, 1))


def _setup_layers(doc):
    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED", pattern=[0.5, 0.25, -0.25])
    for name, props in _LAYERS.items():
        if name not in doc.layers:
            layer = doc.layers.add(name)
            layer.color = props["color"]
            layer.linetype = props["linetype"]


def _write_site_boundary(msp, xs, ys):
    if not xs:
        return
    pline = msp.add_lwpolyline(list(zip(xs, ys)), dxfattribs={"layer": "SITE_BOUNDARY"})
    pline.close(True)
    log.info(f"  DXF: SITE_BOUNDARY  {len(xs)} vertices")


def _write_view_points(msp, xs, ys, view_types, text_h, offset, stride=5):
    count = 0
    for i, (x, y, vt) in enumerate(zip(xs, ys, view_types)):
        if i % stride != 0:
            continue
        color = _VIEW_COLOR.get(vt, 3)
        msp.add_point((x, y, 0), dxfattribs={"layer": "VIEW_POINTS", "color": color})
        msp.add_text(_ascii(vt), dxfattribs={
            "layer": "LABELS", "height": text_h, "color": color,
            "insert": (x + offset, y + offset),
        })
        count += 1
    log.info(f"  DXF: VIEW_POINTS  {count} points (stride={stride})")


def _write_noise_points(msp, xs, ys, noise_db, is_noisy, db_threshold, text_h, offset, stride=5):
    count = 0
    for i, (x, y, db, noisy) in enumerate(zip(xs, ys, noise_db, is_noisy)):
        if i % stride != 0:
            continue
        color = 1 if noisy else 4
        msp.add_point((x, y, 0), dxfattribs={"layer": "NOISE_POINTS", "color": color})
        msp.add_text(f"{db:.1f}dB", dxfattribs={
            "layer": "LABELS", "height": text_h, "color": color,
            "insert": (x + offset, y - offset),
        })
        count += 1
    log.info(f"  DXF: NOISE_POINTS  {count} points (stride={stride})")


def _write_non_building_areas(msp, non_building_areas, text_h):
    if not non_building_areas:
        return
    written = 0
    for key, zone in non_building_areas.items():
        coords = zone.get("coordinates", {})
        zxs, zys = coords.get("x", []), coords.get("y", [])
        if len(zxs) < 3:
            log.warning(f"  DXF: NON_BUILDING {key} < 3 pts skipped")
            continue
        pline = msp.add_lwpolyline(list(zip(zxs, zys)), dxfattribs={"layer": "NON_BUILDING", "color": 5})
        pline.close(True)
        cx, cy = sum(zxs) / len(zxs), sum(zys) / len(zys)
        msp.add_text(_ascii(zone.get("use", key))[:40], dxfattribs={
            "layer": "LABELS", "height": text_h * 1.5, "color": 5,
            "insert": (cx, cy),
        })
        written += 1
    log.info(f"  DXF: NON_BUILDING  {written} zones")


def _write_title_block(msp, site_id, intelligence, xs, ys, n):
    xmin, ymin, xmax, ymax = _bbox(xs, ys)
    width   = xmax - xmin
    height  = ymax - ymin
    margin  = max(3.0, height * 0.08)
    title_h = max(1.5, width * 0.08)
    sub_h   = max(0.8, width * 0.04)
    title_y = ymin - margin - title_h
    sub_y   = title_y - title_h * 1.5

    msp.add_text(
        _ascii(f"ALKF MASTER LAND PLAN - {site_id}"),
        dxfattribs={"layer": "LABELS", "height": title_h, "color": 7, "insert": (xmin, title_y)},
    )
    msp.add_text(
        _ascii(
            f"CRS: {intelligence.get('crs', 'EPSG:3857')}  |  "
            f"Sampling: {intelligence.get('sampling_interval_m', 1.0)}m  |  "
            f"Noise threshold: {float(intelligence.get('db_threshold', 65.0))}dB  |  "
            f"Boundary pts: {n}"
        ),
        dxfattribs={"layer": "LABELS", "height": sub_h, "color": 8, "insert": (xmin, sub_y)},
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def export_dxf(intelligence_data: dict) -> BytesIO:
    site_id      = intelligence_data.get("site_id", "SITE")
    log.info(f"[dxf_export] START  site_id={site_id}")

    boundary     = intelligence_data.get("boundary", {})
    xs           = boundary.get("x", [])
    ys           = boundary.get("y", [])
    view_types   = intelligence_data.get("view_type",  [])
    noise_db     = intelligence_data.get("noise_db",   [])
    is_noisy     = intelligence_data.get("is_noisy",   [])
    db_threshold = float(intelligence_data.get("db_threshold", 65.0))
    n            = len(xs)

    if n == 0:
        raise ValueError("boundary.x is empty")

    view_types = (view_types + ["CITY"] * n)[:n]
    noise_db   = (noise_db   + [45.0]  * n)[:n]
    is_noisy   = (is_noisy   + [False] * n)[:n]

    text_h, offset = _label_scale(xs, ys)
    log.info(f"  DXF: scale text_h={text_h}m  offset={offset}m")

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 6
    doc.header["$LUNITS"]   = 2
    doc.header["$LUPREC"]   = 4

    _setup_layers(doc)
    msp = doc.modelspace()

    _write_site_boundary(msp, xs, ys)
    _write_view_points(msp, xs, ys, view_types, text_h, offset, stride=5)
    _write_noise_points(msp, xs, ys, noise_db, is_noisy, db_threshold, text_h, offset, stride=5)

    non_building = intelligence_data.get("non_building_areas", {})
    if non_building:
        _write_non_building_areas(msp, non_building, text_h)

    _write_title_block(msp, site_id, intelligence_data, xs, ys, n)

    # Write to an in-memory text stream then encode to bytes — no temp files needed.
    import io as _io
    txt_buf = _io.StringIO()
    doc.write(txt_buf)
    raw = txt_buf.getvalue().encode("utf-8")
    buf = BytesIO(raw)
    buf.seek(0)

    log.info(f"[dxf_export] DONE  pts={n}  size={len(raw):,} bytes  text_h={text_h}  offset={offset}")
    return buf
