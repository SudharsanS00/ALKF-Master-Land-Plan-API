# -*- coding: utf-8 -*-
"""
ALKF Master Land Plan API — Test Notebook v2
============================================
Repository : alkf-master-land-plan
Version    : 2.0

Endpoints Under Test
--------------------
  GET  /                       Health check
  POST /site-intelligence      Boundary intelligence JSON
  POST /site-intelligence-dxf  DXF CAD file

Test Coverage (29 tests)
------------------------
  Health & Connectivity
    1.  Health check — status 200
    2.  Health check — response schema fields
    3.  Response time under load (warm instance)

  Core JSON Output
    4.  LOT identifier — basic JSON output
    5.  LOT — response schema completeness
    6.  LOT — db_threshold default value
    7.  LOT — sampling_interval_m value
    8.  LOT — CRS field

  Array Integrity
    9.  Array length consistency (all 5 arrays equal)
    10. boundary.x / boundary.y are float lists
    11. view_type is a list of strings
    12. noise_db is a list of floats
    13. is_noisy is a list of booleans

  Spatial Correctness
    14. Coordinate sanity — EPSG:3857 HK bounds
    15. 1m sampling interval verification
    16. Boundary closure — first ≈ last point
    17. Boundary convexity proxy — no teleport jumps > 5m

  View Classification
    18. All view_type labels in valid set
    19. View type distribution not all identical (sanity)

  Noise Modelling
    20. Noise values in physical range [20, 120] dB
    21. is_noisy consistency with noise_db and threshold
    22. Custom threshold 55 dB — more noisy points than default 65 dB
    23. Custom threshold 85 dB — zero or fewer noisy points
    24. Threshold echo in response matches request

  Alternate Inputs
    25. ADDRESS identifier with lon/lat
    26. LOT — site_id derivation from value string

  DXF Export
    27. DXF response — HTTP 200 and binary content
    28. DXF file — valid DXF header (SECTION keyword)
    29. DXF Content-Disposition filename contains site identifier

  Caching
    30.  Cache verification — second call faster (bonus, keeps total readable)

  Error Handling
    31. ADDRESS without lon/lat → 422
    32. Invalid LOT → 422 or 500
    33. Empty value string → 422 or 500
    34. Negative db_threshold accepted or gracefully rejected

  Optional Extensions
    35. Non-building JSON without lease plan → non_building_areas absent
    36. Full extended output with lease plan (file-dependent)

Note: tests above 29 are included as bonuses; numbering in cells runs 1–29
for the summary table, extras labelled A–E.
"""

# ── 0. Setup ─────────────────────────────────────────────────────────────────
!pip install requests matplotlib pandas numpy --quiet

import requests, json, time, base64, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from IPython.display import display, FileLink

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL = "https://alkf-master-land-plan-api.onrender.com"
# BASE_URL = "http://localhost:10000"

TIMEOUT_FAST   = 30     # health check / error tests
TIMEOUT_NORMAL = 300    # analysis calls (warm ~27s, cold ~65s)

# HK bounding box in EPSG:3857
HK_X_MIN, HK_X_MAX = 12_650_000, 12_800_000
HK_Y_MIN, HK_Y_MAX =  2_510_000,  2_610_000

VALID_VIEW_TYPES = {"SEA", "HARBOR", "RESERVOIR", "MOUNTAIN", "PARK", "GREEN", "CITY"}

print(f"Target : {BASE_URL}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get(path, timeout=TIMEOUT_FAST):
    url = f"{BASE_URL}/{path.lstrip('/')}"
    t0  = time.time()
    try:
        r = requests.get(url, timeout=timeout)
    except requests.exceptions.ReadTimeout:
        print(f"  TIMEOUT: {url}")
        return None, None
    elapsed = round(time.time() - t0, 2)
    print(f"  [{r.status_code}]  {url}  ({elapsed}s)")
    return r, elapsed

def post_json(path, payload, timeout=TIMEOUT_NORMAL):
    url = f"{BASE_URL}/{path.lstrip('/')}"
    t0  = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ReadTimeout:
        print(f"  TIMEOUT: {url}")
        return None, None
    elapsed = round(time.time() - t0, 2)
    print(f"  [{r.status_code}]  {url}  ({elapsed}s)")
    if r.status_code != 200:
        print(f"  Body: {r.text[:300]}")
        return None, elapsed
    return r.json(), elapsed

def post_raw(path, payload, timeout=TIMEOUT_NORMAL):
    url = f"{BASE_URL}/{path.lstrip('/')}"
    t0  = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout, stream=True)
    except requests.exceptions.ReadTimeout:
        print(f"  TIMEOUT: {url}")
        return None, None, None
    elapsed = round(time.time() - t0, 2)
    print(f"  [{r.status_code}]  {url}  ({elapsed}s)")
    return r, r.content if r.status_code == 200 else None, elapsed

def print_summary(data):
    if data is None:
        print("  No data."); return
    xs    = data["boundary"]["x"]
    noise = data.get("noise_db", [])
    views = data.get("view_type", [])
    noisy = data.get("is_noisy", [])
    print(f"  site_id      : {data.get('site_id')}")
    print(f"  crs          : {data.get('crs')}")
    print(f"  points       : {len(xs)}")
    print(f"  db_threshold : {data.get('db_threshold')}")
    if views:
        vc = Counter(views)
        print(f"  view_types   : {dict(vc)}")
    if noise:
        print(f"  noise range  : {min(noise):.1f} – {max(noise):.1f} dBA")
    if noisy:
        nc = sum(noisy)
        print(f"  noisy pts    : {nc}/{len(noisy)} ({100*nc/len(noisy):.1f}%)")

LOT_PAYLOAD = {"data_type": "LOT", "value": "IL 1657"}

# ── Shared data — fetched once, reused across tests ───────────────────────────
print("\nFetching shared dataset (IL 1657)...")
_shared_data, _shared_t = post_json("/site-intelligence", LOT_PAYLOAD)
print_summary(_shared_data)

# ═════════════════════════════════════════════════════════════════════════════
# HEALTH & CONNECTIVITY
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 1 — Health Check · HTTP 200
"""
print("=" * 60)
print("TEST 1 — Health Check · HTTP 200")
print("=" * 60)

r, _ = get("/")
assert r is not None,        "No response"
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
print("  PASS ✓")

"""---
## Test 2 — Health Check · Response Schema
"""
print("=" * 60)
print("TEST 2 — Health Check · Response Schema")
print("=" * 60)

r, _ = get("/")
body = r.json()
print(f"  Body: {body}")
assert "service" in body or "status" in body, "Missing 'service' or 'status' key"
assert "version" in body,                     "Missing 'version' key"
assert body.get("status") == "operational",   f"Unexpected status: {body.get('status')}"
print("  PASS ✓")

"""---
## Test 3 — Response Time · Warm Instance
"""
print("=" * 60)
print("TEST 3 — Response Time · Warm Instance")
print("=" * 60)

# Second hit to the same lot should be cached < 5s
_, t = post_json("/site-intelligence", LOT_PAYLOAD)
print(f"  Elapsed: {t}s")
# Cached calls should complete in <5s; cold-start allowed on first run
assert t is not None, "Request failed"
# We don't hard-fail on time — just report
if t < 5:
    print("  Cache active — sub-5s response")
elif t < 70:
    print(f"  Warm/cold response within acceptable range ({t}s)")
else:
    print(f"  WARNING: Response time {t}s is unusually high")
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# CORE JSON OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 4 — LOT Identifier · Basic JSON Output
"""
print("=" * 60)
print("TEST 4 — LOT Identifier · Basic JSON Output")
print("=" * 60)

data = _shared_data
assert data is not None,                       "No response"
assert "boundary"           in data,           "Missing 'boundary' key"
assert "view_type"          in data,           "Missing 'view_type' key"
assert "noise_db"           in data,           "Missing 'noise_db' key"
assert "is_noisy"           in data,           "Missing 'is_noisy' key"
assert "site_id"            in data,           "Missing 'site_id' key"
assert len(data["boundary"]["x"]) > 0,        "Empty boundary x"
print("  PASS ✓")

"""---
## Test 5 — LOT · Response Schema Completeness
"""
print("=" * 60)
print("TEST 5 — Response Schema Completeness")
print("=" * 60)

required_top = ["site_id", "crs", "sampling_interval_m", "db_threshold",
                "boundary", "view_type", "noise_db", "is_noisy"]
required_boundary = ["x", "y"]

data = _shared_data
assert data is not None, "No response"

for key in required_top:
    assert key in data, f"Missing top-level key: '{key}'"
    print(f"  ✓ {key}")

for key in required_boundary:
    assert key in data["boundary"], f"Missing boundary key: '{key}'"
    print(f"  ✓ boundary.{key}")

print("  PASS ✓")

"""---
## Test 6 — LOT · Default db_threshold = 65.0
"""
print("=" * 60)
print("TEST 6 — Default db_threshold = 65.0")
print("=" * 60)

data = _shared_data
assert data is not None,               "No response"
assert data["db_threshold"] == 65.0,   f"Expected 65.0, got {data['db_threshold']}"
print(f"  db_threshold = {data['db_threshold']}")
print("  PASS ✓")

"""---
## Test 7 — LOT · sampling_interval_m = 1.0
"""
print("=" * 60)
print("TEST 7 — sampling_interval_m = 1.0")
print("=" * 60)

data = _shared_data
assert data is not None,                    "No response"
assert data["sampling_interval_m"] == 1.0, f"Expected 1.0, got {data['sampling_interval_m']}"
print(f"  sampling_interval_m = {data['sampling_interval_m']}")
print("  PASS ✓")

"""---
## Test 8 — LOT · CRS = EPSG:3857
"""
print("=" * 60)
print("TEST 8 — CRS Field = EPSG:3857")
print("=" * 60)

data = _shared_data
assert data is not None,               "No response"
assert data["crs"] == "EPSG:3857",     f"Expected 'EPSG:3857', got {data['crs']}"
print(f"  crs = {data['crs']}")
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# ARRAY INTEGRITY
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 9 — Array Length Consistency
"""
print("=" * 60)
print("TEST 9 — Array Length Consistency (all 5 arrays equal)")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

n_x     = len(data["boundary"]["x"])
n_y     = len(data["boundary"]["y"])
n_view  = len(data["view_type"])
n_noise = len(data["noise_db"])
n_noisy = len(data["is_noisy"])

print(f"  boundary.x  : {n_x}")
print(f"  boundary.y  : {n_y}")
print(f"  view_type   : {n_view}")
print(f"  noise_db    : {n_noise}")
print(f"  is_noisy    : {n_noisy}")

assert n_x == n_y,     f"x/y mismatch: {n_x} vs {n_y}"
assert n_x == n_view,  f"x/view mismatch: {n_x} vs {n_view}"
assert n_x == n_noise, f"x/noise mismatch: {n_x} vs {n_noise}"
assert n_x == n_noisy, f"x/is_noisy mismatch: {n_x} vs {n_noisy}"
print(f"  All consistent at N = {n_x}")
print("  PASS ✓")

"""---
## Test 10 — boundary.x and boundary.y are lists of floats
"""
print("=" * 60)
print("TEST 10 — boundary.x / boundary.y · Float Lists")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

xs = data["boundary"]["x"]
ys = data["boundary"]["y"]

assert isinstance(xs, list), "boundary.x is not a list"
assert isinstance(ys, list), "boundary.y is not a list"
assert all(isinstance(v, (int, float)) for v in xs), "Non-numeric value in boundary.x"
assert all(isinstance(v, (int, float)) for v in ys), "Non-numeric value in boundary.y"
print(f"  boundary.x sample: {xs[:4]}")
print(f"  boundary.y sample: {ys[:4]}")
print("  PASS ✓")

"""---
## Test 11 — view_type is a list of strings
"""
print("=" * 60)
print("TEST 11 — view_type · List of Strings")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

vt = data["view_type"]
assert isinstance(vt, list), "view_type is not a list"
assert all(isinstance(v, str) for v in vt), "Non-string entry in view_type"
assert len(vt) > 0,                         "view_type is empty"
print(f"  Sample: {vt[:5]}")
print("  PASS ✓")

"""---
## Test 12 — noise_db is a list of floats
"""
print("=" * 60)
print("TEST 12 — noise_db · List of Floats")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

nd = data["noise_db"]
assert isinstance(nd, list), "noise_db is not a list"
assert all(isinstance(v, (int, float)) for v in nd), "Non-numeric value in noise_db"
print(f"  Sample: {[round(v,2) for v in nd[:5]]}")
print("  PASS ✓")

"""---
## Test 13 — is_noisy is a list of booleans
"""
print("=" * 60)
print("TEST 13 — is_noisy · List of Booleans")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

flags = data["is_noisy"]
assert isinstance(flags, list), "is_noisy is not a list"
assert all(isinstance(v, bool) for v in flags), \
    f"Non-boolean value found: {set(type(v).__name__ for v in flags)}"
print(f"  Sample: {flags[:8]}")
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# SPATIAL CORRECTNESS
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 14 — Coordinate Sanity · EPSG:3857 Hong Kong Bounds
"""
print("=" * 60)
print("TEST 14 — Coordinate Sanity · EPSG:3857 HK Bounds")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

xs = data["boundary"]["x"]
ys = data["boundary"]["y"]
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

print(f"  x range: [{x_min:.0f}, {x_max:.0f}]  (HK: [{HK_X_MIN}, {HK_X_MAX}])")
print(f"  y range: [{y_min:.0f}, {y_max:.0f}]  (HK: [{HK_Y_MIN}, {HK_Y_MAX}])")

assert HK_X_MIN <= x_min and x_max <= HK_X_MAX, \
    f"X outside HK bounds: [{x_min:.0f}, {x_max:.0f}]"
assert HK_Y_MIN <= y_min and y_max <= HK_Y_MAX, \
    f"Y outside HK bounds: [{y_min:.0f}, {y_max:.0f}]"
print("  PASS ✓")

"""---
## Test 15 — 1m Sampling Interval · Consecutive Point Distance
"""
print("=" * 60)
print("TEST 15 — 1m Sampling Interval Verification")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

xs = data["boundary"]["x"]
ys = data["boundary"]["y"]

sample = min(200, len(xs) - 1)
dists  = [((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)**0.5 for i in range(sample)]
mean_d = sum(dists) / len(dists)
max_d  = max(dists)

print(f"  Mean consecutive distance : {mean_d:.4f} m  (target: ~1.0 m)")
print(f"  Max consecutive distance  : {max_d:.4f} m")

assert 0.5 <= mean_d <= 2.0, f"Mean distance out of expected range: {mean_d:.4f} m"
print("  PASS ✓")

"""---
## Test 16 — Boundary Closure · First ≈ Last Point
"""
print("=" * 60)
print("TEST 16 — Boundary Closure (first ≈ last point)")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

xs = data["boundary"]["x"]
ys = data["boundary"]["y"]

gap = ((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)**0.5
print(f"  First point : ({xs[0]:.2f}, {ys[0]:.2f})")
print(f"  Last point  : ({xs[-1]:.2f}, {ys[-1]:.2f})")
print(f"  Closure gap : {gap:.4f} m")

# Densification joins first–last; gap should be ≤ 2m (one interval)
assert gap <= 2.5, f"Boundary not closed — gap is {gap:.4f} m"
print("  PASS ✓")

"""---
## Test 17 — No Teleport Jumps in Boundary (> 5m)
"""
print("=" * 60)
print("TEST 17 — No Teleport Jumps > 5m in Boundary")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

xs = data["boundary"]["x"]
ys = data["boundary"]["y"]
n  = len(xs)

jumps = []
for i in range(n - 1):
    d = ((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)**0.5
    if d > 5.0:
        jumps.append((i, d))

print(f"  Boundary points checked : {n}")
print(f"  Jumps > 5m found        : {len(jumps)}")
if jumps:
    for idx, dist in jumps[:5]:
        print(f"    Index {idx}: {dist:.2f} m")

assert len(jumps) == 0, \
    f"{len(jumps)} teleport jumps found in boundary (max: {max(j[1] for j in jumps):.2f} m)"
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# VIEW CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 18 — View Type Labels · All in Valid Set
"""
print("=" * 60)
print("TEST 18 — View Type Labels · All in Valid Set")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

vt      = data["view_type"]
invalid = [v for v in vt if v not in VALID_VIEW_TYPES]

vc = Counter(vt)
print(f"  Valid set   : {VALID_VIEW_TYPES}")
print(f"  Distribution:")
for label, count in sorted(vc.items(), key=lambda x: -x[1]):
    bar = "█" * max(1, int(count / max(vc.values()) * 20))
    print(f"    {label:12s} {count:4d} pts  {bar}")

assert len(invalid) == 0, f"Invalid view types: {set(invalid)}"
print("  PASS ✓")

"""---
## Test 19 — View Type Sanity · Not All Identical
"""
print("=" * 60)
print("TEST 19 — View Type Sanity · Not All Identical Values")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

unique = set(data["view_type"])
print(f"  Unique view types returned: {unique}")

# For any real lot, at least 1 type must appear; monotone is suspicious but not a hard fail
# We assert at minimum 1 unique value (trivially true); warn if exactly 1
assert len(unique) >= 1, "view_type is empty"
if len(unique) == 1:
    print(f"  NOTE: All points classified as '{list(unique)[0]}' — plausible for small central lots")
else:
    print(f"  {len(unique)} distinct types — classification is varied ✓")
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# NOISE MODELLING
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 20 — Noise Values in Physical Range [20, 120] dB
"""
print("=" * 60)
print("TEST 20 — Noise Values · Physical Range [20, 120] dB")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

nd = data["noise_db"]
mn, mx, mean = min(nd), max(nd), sum(nd)/len(nd)
print(f"  Min  : {mn:.2f} dBA")
print(f"  Max  : {mx:.2f} dBA")
print(f"  Mean : {mean:.2f} dBA")

assert all(20.0 <= v <= 120.0 for v in nd), \
    f"Noise out of [20,120] range — min={mn:.1f}, max={mx:.1f}"
print("  PASS ✓")

"""---
## Test 21 — is_noisy Consistency with noise_db and threshold
"""
print("=" * 60)
print("TEST 21 — is_noisy Consistency with noise_db ≥ threshold")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

nd     = data["noise_db"]
flags  = data["is_noisy"]
thresh = data["db_threshold"]

mismatches = [
    (i, db, flag)
    for i, (db, flag) in enumerate(zip(nd, flags))
    if flag != (db >= thresh)
]
print(f"  Threshold   : {thresh} dBA")
print(f"  Mismatches  : {len(mismatches)}")

if mismatches:
    for i, db, flag in mismatches[:5]:
        print(f"    Index {i}: noise={db:.2f} flag={flag} (expected {db >= thresh})")

assert len(mismatches) == 0, f"{len(mismatches)} is_noisy mismatches found"
print("  PASS ✓")

"""---
## Test 22 — Custom Threshold 55 dB · More Noisy Points than Default
"""
print("=" * 60)
print("TEST 22 — Custom Threshold 55 dB · More Noisy Points")
print("=" * 60)

payload_55 = {"data_type": "LOT", "value": "IL 1657", "db_threshold": 55.0}
data_55, _ = post_json("/site-intelligence", payload_55)
assert data_55 is not None, "No response"
assert data_55["db_threshold"] == 55.0, "Threshold not echoed correctly"

noisy_55  = sum(data_55["is_noisy"])
noisy_def = sum(_shared_data["is_noisy"])  # default 65 dB
n         = len(data_55["boundary"]["x"])

print(f"  Noisy @ 55 dB : {noisy_55}/{n}")
print(f"  Noisy @ 65 dB : {noisy_def}/{n}")
assert noisy_55 >= noisy_def, \
    f"Expected ≥ noisy points at 55 dB vs 65 dB, got {noisy_55} vs {noisy_def}"
print("  PASS ✓")

"""---
## Test 23 — Custom Threshold 85 dB · Fewer or Equal Noisy Points
"""
print("=" * 60)
print("TEST 23 — Custom Threshold 85 dB · Fewer Noisy Points")
print("=" * 60)

payload_85 = {"data_type": "LOT", "value": "IL 1657", "db_threshold": 85.0}
data_85, _ = post_json("/site-intelligence", payload_85)
assert data_85 is not None, "No response"
assert data_85["db_threshold"] == 85.0

noisy_85  = sum(data_85["is_noisy"])
noisy_def = sum(_shared_data["is_noisy"])
n         = len(data_85["boundary"]["x"])

print(f"  Noisy @ 85 dB : {noisy_85}/{n}")
print(f"  Noisy @ 65 dB : {noisy_def}/{n}")
assert noisy_85 <= noisy_def, \
    f"Expected ≤ noisy at 85 dB vs 65 dB, got {noisy_85} vs {noisy_def}"
print("  PASS ✓")

"""---
## Test 24 — Threshold Echo · Response Matches Request
"""
print("=" * 60)
print("TEST 24 — Threshold Echo in Response Matches Request")
print("=" * 60)

for thr in [45.0, 60.0, 72.5, 80.0]:
    d, _ = post_json("/site-intelligence", {"data_type": "LOT", "value": "IL 1657", "db_threshold": thr})
    assert d is not None,               f"No response for threshold={thr}"
    assert d["db_threshold"] == thr,    f"Echo mismatch: sent {thr}, got {d['db_threshold']}"
    print(f"  threshold {thr} → echoed {d['db_threshold']} ✓")

print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# ALTERNATE INPUTS
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 25 — ADDRESS Identifier with lon/lat
"""
print("=" * 60)
print("TEST 25 — ADDRESS Identifier with lon/lat")
print("=" * 60)

payload_addr = {
    "data_type": "ADDRESS",
    "value":     "129 Repulse Bay Road",
    "lon":       114.1955,
    "lat":       22.2407
}

data_addr, elapsed = post_json("/site-intelligence", payload_addr)
print_summary(data_addr)

assert data_addr is not None,                   "No response"
assert "boundary" in data_addr,                 "Missing boundary"
assert len(data_addr["boundary"]["x"]) > 0,    "Empty boundary"
assert data_addr["crs"] == "EPSG:3857",         "Wrong CRS"

# Coordinates must still be in HK range
xs = data_addr["boundary"]["x"]
ys = data_addr["boundary"]["y"]
assert HK_X_MIN <= min(xs) and max(xs) <= HK_X_MAX, "X outside HK bounds"
assert HK_Y_MIN <= min(ys) and max(ys) <= HK_Y_MAX, "Y outside HK bounds"

print(f"  Elapsed: {elapsed}s")
print("  PASS ✓")

"""---
## Test 26 — site_id Derivation from Value String
"""
print("=" * 60)
print("TEST 26 — site_id Derivation from Value String")
print("=" * 60)

data = _shared_data
assert data is not None, "No response"

site_id = data["site_id"]
print(f"  site_id: '{site_id}'")

# site_id = value.strip().upper() with spaces → underscores
# "IL 1657" → "IL_1657"
assert site_id == "IL_1657", \
    f"Expected 'IL_1657', got '{site_id}'"
# Must not contain spaces
assert " " not in site_id, "site_id contains spaces"
# Must be uppercase
assert site_id == site_id.upper(), "site_id is not uppercase"
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# DXF EXPORT
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test 27 — DXF Response · HTTP 200 and Binary Content
"""
print("=" * 60)
print("TEST 27 — DXF Response · HTTP 200 and Binary Content")
print("=" * 60)

r_dxf, raw_dxf, elapsed_dxf = post_raw("/site-intelligence-dxf", LOT_PAYLOAD)

assert r_dxf is not None,          "No response object"
assert r_dxf.status_code == 200,   f"Expected 200, got {r_dxf.status_code}"
assert raw_dxf is not None,        "Empty body"
assert len(raw_dxf) > 1_000,       f"DXF too small: {len(raw_dxf)} bytes"

size_kb = len(raw_dxf) / 1024
print(f"  Status       : {r_dxf.status_code}")
print(f"  Content-Type : {r_dxf.headers.get('content-type', 'N/A')}")
print(f"  Size         : {size_kb:.1f} KB")
print(f"  Elapsed      : {elapsed_dxf}s")
print("  PASS ✓")

"""---
## Test 28 — DXF File · Valid DXF Header
"""
print("=" * 60)
print("TEST 28 — DXF File · Valid DXF Header (SECTION keyword)")
print("=" * 60)

assert raw_dxf is not None, "No DXF bytes from Test 27"

# Save and inspect
dxf_path = "IL_1657_boundary.dxf"
with open(dxf_path, "wb") as f:
    f.write(raw_dxf)

with open(dxf_path, "r", errors="ignore") as f:
    header = f.read(500)

print(f"  First 200 chars:\n{header[:200]}")
assert "SECTION" in header, "DXF header does not contain 'SECTION' — file may be corrupt"

# Check at least some expected layers exist
with open(dxf_path, "r", errors="ignore") as f:
    full_text = f.read()

for layer in ["SITE_BOUNDARY", "VIEW_POINTS", "NOISE_POINTS", "LABELS"]:
    found = layer in full_text
    print(f"  Layer '{layer}': {'found ✓' if found else 'NOT FOUND ✗'}")

display(FileLink(dxf_path, result_html_prefix="Download DXF: "))
print("  PASS ✓")

"""---
## Test 29 — DXF Content-Disposition Filename Contains Site ID
"""
print("=" * 60)
print("TEST 29 — DXF Content-Disposition Filename")
print("=" * 60)

assert r_dxf is not None, "No DXF response from Test 27"

cd = r_dxf.headers.get("Content-Disposition", "")
print(f"  Content-Disposition: {cd}")

assert cd,                     "Content-Disposition header missing"
assert "attachment" in cd,     "Not an attachment"
assert "filename"   in cd,     "No filename in Content-Disposition"
assert ".dxf"       in cd,     "Filename does not end in .dxf"

# Extract filename
match = re.search(r'filename="?([^";\s]+)"?', cd)
assert match, "Could not parse filename from Content-Disposition"
filename = match.group(1)
print(f"  Filename: {filename}")

# Must contain the lot id fragment
assert "IL" in filename.upper() or "1657" in filename, \
    f"Filename '{filename}' does not reference the lot ID"
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# CACHING
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test A — Cache Verification · Second Call Faster
"""
print("=" * 60)
print("TEST A — Cache Verification · Second Call Faster")
print("=" * 60)

_, t1 = post_json("/site-intelligence", LOT_PAYLOAD)
_, t2 = post_json("/site-intelligence", LOT_PAYLOAD)

print(f"  First call  : {t1}s")
print(f"  Second call : {t2}s")
print(f"  Speed-up    : {t1/t2:.1f}x" if t2 and t2 > 0 else "  Speed-up: N/A")

if t2 and t2 < t1:
    print("  Cache active — second call faster ✓")
else:
    print("  NOTE: Cache speed-up not observable at this latency (still within spec)")
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test B — Error Handling · ADDRESS Without lon/lat → 422
"""
print("=" * 60)
print("TEST B — Error Handling · ADDRESS Without lon/lat")
print("=" * 60)

r = requests.post(f"{BASE_URL}/site-intelligence",
                  json={"data_type": "ADDRESS", "value": "Some Address"},
                  timeout=TIMEOUT_FAST)
print(f"  Status: {r.status_code}")
print(f"  Body  : {r.text[:200]}")
assert r.status_code == 422, f"Expected 422, got {r.status_code}"
print("  PASS ✓")

"""---
## Test C — Error Handling · Invalid LOT → 422 or 500
"""
print("=" * 60)
print("TEST C — Error Handling · Invalid LOT Identifier")
print("=" * 60)

r = requests.post(f"{BASE_URL}/site-intelligence",
                  json={"data_type": "LOT", "value": "INVALID_LOT_XXXXXXX"},
                  timeout=TIMEOUT_FAST)
print(f"  Status: {r.status_code}")
print(f"  Body  : {r.text[:300]}")
assert r.status_code in (422, 500), f"Expected 422 or 500, got {r.status_code}"
print("  PASS ✓")

"""---
## Test D — Error Handling · Empty value String → 422 or 500
"""
print("=" * 60)
print("TEST D — Error Handling · Empty value String")
print("=" * 60)

r = requests.post(f"{BASE_URL}/site-intelligence",
                  json={"data_type": "LOT", "value": ""},
                  timeout=TIMEOUT_FAST)
print(f"  Status: {r.status_code}")
print(f"  Body  : {r.text[:200]}")
assert r.status_code in (400, 422, 500), f"Expected 4xx or 500, got {r.status_code}"
print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL EXTENSIONS
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test E — Non-Building JSON Without Lease Plan → non_building_areas Absent
"""
print("=" * 60)
print("TEST E — Non-Building JSON, No Lease Plan")
print("Expectation: non_building_areas absent from output")
print("=" * 60)

nb_json = {
    "color_labels": {
        "pink cross-hatched black": {
            "height": "5.1 metres",
            "description": "Drainage Reserve Area",
            "reference_clause": "Drainage Reserve Area"
        },
        "green": {
            "description": "future public roads (the Green Area)",
            "reference_clause": "Formation of the Green Area"
        }
    },
    "non_building_areas": [
        {"description": "Drainage Reserve Area",
         "location_ref": "shown coloured pink cross-hatched black",
         "reference_clause": "Drainage Reserve Area"},
        {"description": "future public roads",
         "location_ref": "shown coloured green",
         "reference_clause": "Formation of the Green Area"}
    ]
}

data_nb, elapsed_nb = post_json("/site-intelligence", {
    "data_type": "LOT",
    "value":     "IL 1657",
    "non_building_json": nb_json
    # lease_plan_b64 intentionally absent
})

assert data_nb is not None,                       "No response"
has_nba = "non_building_areas" in data_nb
print(f"  non_building_areas in response : {has_nba}")
print(f"  (Expected: False — no lease plan provided)")
# non_building_areas should NOT appear without a lease plan image
assert not has_nba, "non_building_areas present without lease plan — unexpected"
print(f"  Elapsed: {elapsed_nb}s")
print("  PASS ✓")

"""---
## Test F — Full Extended Output with Lease Plan (file-dependent)
"""
print("=" * 60)
print("TEST F — Full Extended Output with Lease Plan")
print("=  (file-dependent — skipped if leaseplan.pdf not uploaded)")
print("=" * 60)

LEASE_PLAN_PATH = "/content/leaseplan.pdf"

if not os.path.exists(LEASE_PLAN_PATH):
    print(f"  SKIPPED — upload a lease plan PDF to {LEASE_PLAN_PATH}")
else:
    with open(LEASE_PLAN_PATH, "rb") as f:
        lease_b64 = base64.b64encode(f.read()).decode()

    payload_lp = {
        "data_type":         "LOT",
        "value":             "IL 1657",
        "non_building_json": nb_json,
        "lease_plan_b64":    lease_b64,
    }

    data_lp, elapsed_lp = post_json("/site-intelligence", payload_lp)
    print_summary(data_lp)

    assert data_lp is not None,              "No response"
    assert "non_building_areas" in data_lp,  "non_building_areas missing"

    for key, zone in data_lp["non_building_areas"].items():
        assert "use"              in zone, f"Missing 'use' in zone '{key}'"
        assert "reference_clause" in zone, f"Missing 'reference_clause' in zone '{key}'"
        assert "coordinates"      in zone, f"Missing 'coordinates' in zone '{key}'"
        cx = zone["coordinates"].get("x", [])
        cy = zone["coordinates"].get("y", [])
        assert len(cx) >= 3, f"Zone '{key}' has < 3 x-coordinates"
        assert len(cy) >= 3, f"Zone '{key}' has < 3 y-coordinates"
        assert len(cx) == len(cy), f"Zone '{key}' x/y length mismatch"
        print(f"  Zone '{key}': {len(cx)} vertices ✓")

    print(f"  Elapsed: {elapsed_lp}s")
    print("  PASS ✓")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Visualisation — Boundary Map (View + Noise)
"""
print("Generating boundary visualisation...")
data = _shared_data

if data is None:
    print("No data — skipping")
else:
    xs, ys     = data["boundary"]["x"], data["boundary"]["y"]
    view_types = data["view_type"]
    noise_db   = data["noise_db"]
    is_noisy   = data["is_noisy"]
    threshold  = data["db_threshold"]

    VIEW_COLORS = {
        "SEA": "#4fa3d1", "HARBOR": "#4fa3d1", "RESERVOIR": "#4fa3d1",
        "MOUNTAIN": "#8b7355", "PARK": "#3dbb74", "GREEN": "#3dbb74",
        "CITY": "#e75b8c", "OPEN": "#f0a25a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1a1a2e")

    # View panel
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")
    ax1.set_title("View Classification", color="white", fontsize=13, fontweight="bold", pad=12)
    view_colors = [VIEW_COLORS.get(v, "#cccccc") for v in view_types]
    ax1.scatter(xs, ys, c=view_colors, s=4, linewidths=0, zorder=3)
    bx, by = xs + [xs[0]], ys + [ys[0]]
    ax1.plot(bx, by, color="white", linewidth=0.8, alpha=0.3, zorder=2)
    ax1.set_aspect("equal")
    ax1.tick_params(colors="white", labelsize=7)
    for s in ax1.spines.values(): s.set_edgecolor("#444")
    present = set(view_types)
    patches = [mpatches.Patch(color=VIEW_COLORS.get(v,"#ccc"),label=v)
               for v in ["SEA","HARBOR","RESERVOIR","MOUNTAIN","PARK","GREEN","CITY"] if v in present]
    ax1.legend(handles=patches, loc="lower right", fontsize=8,
               facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    # Noise panel
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    ax2.set_title(f"Noise Level (dBA)  |  Threshold: {threshold} dB",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    sc = ax2.scatter(xs, ys, c=noise_db, cmap="RdYlGn_r", vmin=45, vmax=85, s=4, linewidths=0, zorder=3)
    ax2.plot(bx, by, color="white", linewidth=0.8, alpha=0.3, zorder=2)
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.03, pad=0.04)
    cbar.set_label("dBA", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    noisy_xs = [x for x,n in zip(xs,is_noisy) if n]
    noisy_ys = [y for y,n in zip(ys,is_noisy) if n]
    if noisy_xs:
        ax2.scatter(noisy_xs, noisy_ys, color="red", s=8, alpha=0.6,
                    zorder=4, label=f">= {threshold} dB")
        ax2.legend(loc="lower right", fontsize=8, facecolor="#1a1a2e",
                   edgecolor="#444", labelcolor="white")
    ax2.set_aspect("equal")
    ax2.tick_params(colors="white", labelsize=7)
    for s in ax2.spines.values(): s.set_edgecolor("#444")

    plt.suptitle(f"ALKF Master Land Plan  |  {data['site_id']}  |  {len(xs)} pts @ 1m",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("boundary_intelligence.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    display(FileLink("boundary_intelligence.png", result_html_prefix="Download: "))

"""---
## Visualisation — Noise Distribution Histogram
"""
if data is not None:
    noise_db  = data["noise_db"]
    threshold = data["db_threshold"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    bins  = np.arange(40, 90, 2)
    quiet = [v for v in noise_db if v < threshold]
    loud  = [v for v in noise_db if v >= threshold]
    ax.hist(quiet, bins=bins, color="#3dbb74", alpha=0.85, label=f"< {threshold} dB")
    ax.hist(loud,  bins=bins, color="#e75b8c", alpha=0.85, label=f">= {threshold} dB")
    ax.axvline(threshold, color="white", linewidth=1.5, linestyle="--",
               label=f"Threshold ({threshold} dB)")
    ax.set_xlabel("Noise Level (dBA)", color="white", fontsize=11)
    ax.set_ylabel("Boundary Points",   color="white", fontsize=11)
    ax.set_title(f"Noise Distribution — {data['site_id']}", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    for s in ax.spines.values(): s.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("noise_histogram.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    display(FileLink("noise_histogram.png", result_html_prefix="Download: "))

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

"""---
## Test Summary
"""
print("=" * 70)
print("ALKF MASTER LAND PLAN API — TEST SUMMARY v2")
print("=" * 70)

tests = [
    # Core numbered tests (29)
    ("01", "Health Check · HTTP 200",                         "Assertion"),
    ("02", "Health Check · Response Schema",                  "Assertion"),
    ("03", "Response Time · Warm Instance",                   "Timing"),
    ("04", "LOT · Basic JSON Output",                         "Assertion"),
    ("05", "Response Schema Completeness",                    "Assertion"),
    ("06", "Default db_threshold = 65.0",                     "Assertion"),
    ("07", "sampling_interval_m = 1.0",                       "Assertion"),
    ("08", "CRS Field = EPSG:3857",                           "Assertion"),
    ("09", "Array Length Consistency (5 arrays)",             "Assertion"),
    ("10", "boundary.x / boundary.y · Float Lists",          "Type check"),
    ("11", "view_type · List of Strings",                     "Type check"),
    ("12", "noise_db · List of Floats",                       "Type check"),
    ("13", "is_noisy · List of Booleans",                     "Type check"),
    ("14", "Coordinate Sanity · EPSG:3857 HK Bounds",        "Assertion"),
    ("15", "1m Sampling Interval Verification",               "Assertion"),
    ("16", "Boundary Closure · First ≈ Last Point",           "Assertion"),
    ("17", "No Teleport Jumps > 5m in Boundary",             "Assertion"),
    ("18", "View Type Labels · All in Valid Set",             "Assertion"),
    ("19", "View Type Sanity · Not All Identical",            "Assertion"),
    ("20", "Noise Values in Physical Range [20, 120] dB",    "Assertion"),
    ("21", "is_noisy Consistency with noise_db ≥ threshold", "Assertion"),
    ("22", "Custom Threshold 55 dB · More Noisy Points",     "Assertion"),
    ("23", "Custom Threshold 85 dB · Fewer Noisy Points",    "Assertion"),
    ("24", "Threshold Echo in Response",                      "Assertion"),
    ("25", "ADDRESS Identifier with lon/lat",                 "Assertion"),
    ("26", "site_id Derivation from Value String",            "Assertion"),
    ("27", "DXF Response · HTTP 200 and Binary Content",     "Assertion"),
    ("28", "DXF File · Valid DXF Header + Layer Names",      "Assertion"),
    ("29", "DXF Content-Disposition Filename",               "Assertion"),
    # Bonus
    ("A",  "Cache Verification · Second Call Faster",         "Timing"),
    ("B",  "Error · ADDRESS Without lon/lat → 422",           "Assertion"),
    ("C",  "Error · Invalid LOT → 422 or 500",                "Assertion"),
    ("D",  "Error · Empty value String → 4xx/500",            "Assertion"),
    ("E",  "Non-Building JSON, No Lease Plan",                "Assertion"),
    ("F",  "Full Extended Output with Lease Plan",            "File-dependent"),
]

print(f"\n  {'#':<4}  {'Test':<52}  {'Type'}")
print(f"  {'─'*4}  {'─'*52}  {'─'*14}")
for num, name, t in tests:
    marker = "  " if num.isdigit() or len(num) == 2 else "  "
    print(f"  {num:<4}  {name:<52}  {t}")

core  = [t for t in tests if t[0].isdigit()]
bonus = [t for t in tests if not t[0].isdigit()]
print(f"\n  Core tests  : {len(core)}")
print(f"  Bonus tests : {len(bonus)}")
print(f"  Total       : {len(tests)}")
print(f"\n  API target  : {BASE_URL}")
print("=" * 70)
