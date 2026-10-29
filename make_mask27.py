# -*- coding: utf-8 -*-
"""Generate images-2027/map_mask27.png -- the 2027 highland-layer lookup mask.

Why this file exists
--------------------
`main.py` uses the mask as a *two-level lookup table* (main.py:1455 / 1467):

    color = mask_image[y_c, x_c]
    if color[0] == color[1] == color[2] == 0:   # pure black -> ground / road layer
        ...use M_ground,   Z = 0
    else:                                       # any non-black -> highland layer
        ...use M_height_r

Two consequences that drive the format:

* The test is *exact equality with zero*, so the mask must be strictly
  bilevel.  A JPEG round-trips saturated pure green through YCbCr and lands
  on (1, 255, 0); the file this script replaced,
  images-2027/map_mask27.jpg, had 2 058 464 such pixels and only 12 429 true
  (0, 255, 0) ones.  Hence: lossless PNG, only (0,0,0) and (0,255,0).
* Only the *position* of the boundary matters for the layer decision, so the
  geometry below is kept in metres and snapped to the structural lines that
  are actually drawn on images-2027/red.jpg.

Geometry source
---------------
Not hand-drawn, and not migrated from another season.  Every coordinate is
measured on images-2027/red.jpg (1500 x 2800 px, 100 px/m, x = 15 m short
axis, y = 28 m long axis, row = (28 - y) * 100).

The annotations on that render are the only trustworthy geometry channel:
luminance is a 9-step grey relief with heavy surface texture, and the official
rules booklet (RMUC 2027 V1.0.0, 4.3) gives figure 4-26 / 4-27 only as
bitmaps with no dimension chain.  So the mask is built from the two saturated
ink colours, which are separable by *chroma* alone and by nothing else:

    RED  = r - 0.5*(g + b) > 45     50 380 px
    BLUE = b - 0.5*(g + r) > 45     60 437 px

Stroke width is 8 px p50 (distance-transform skeleton, p10 6 / p90 8), so a
centre line carries a +-4 px = +-4 cm irreducible uncertainty.  Each edge
below is the *stroke centre*, re-measured at run time by measure_lines();

Drawn outline of the red-half trapezoidal highland
--------------------------------------------------
Chromatography gives these strokes, all confirmed present by an ink probe that
reports the ink-hit fraction along the ideal line (see --verify):

    H1  y = 4.259    x 11.023 .. 14.420    far edge            100 % ink
    V1  x = 11.023   y  4.259 ..  7.050    inner edge, lower   100 % ink
    V3  x = 13.520   y 10.060 .. 12.550    inner edge, upper   100 % ink
    V2  x = 14.420   y  4.259 .. 12.550    outer edge          100 % ink
    CAP y = 12.550   x 13.520 .. 14.420    near edge           endpoints only
    SL  (11.023, 7.050) -> (13.520, 10.060)  inner slant       not drawn

Four edges are therefore *drawn*, and two are honest closures:

* the near edge CAP is undrawn but both verticals V2 and V3 terminate at
  y = 12.550 within 20 mm, so the cap is the minimum closure and is anchored
  at both ends;
* the slant SL is undrawn -- a +-0.15 m ink probe along it scores 3 / 79 hits,
  all three being the two strokes' own end caps.  It is the minimum-length
  closure between the two measured stroke terminations, and it is preferred
  over any step because a step would have to cut across the detached stroke
  x = 12.310 (y 5.400 .. 7.830), which is *interior* to this polygon.

The render is not perfectly 180-degree symmetric: the blue half's inner edge
runs 0.34 m further than its red counterpart (red 4.259..7.050 vs blue
20.60..23.73, i.e. 4.27..7.40 after the rotation).  This file takes the red
half as primary and derives the blue half by exact rotation, so the mask is
symmetric by construction; the 0.34 m asymmetry is annotation noise on the
render and bounds the achievable fidelity.

Why the previous numbers were wrong
-----------------------------------
The mask this script previously produced was built from LSD edges on the
greyscale image, which locks onto *tone* edges as readily as ink.  Three of
its six edges were not structure lines at all:

    edge     old value    measured    error
    far y      3.581       4.259      +678 mm    (old line is a ramp highlight)
    inner x   10.733      11.023      +290 mm
    outer x   14.507      14.420       -87 mm
    mid x     13.480      13.520       +40 mm
    near y    12.970      12.550      -420 mm
    inner top  7.890       7.050      -840 mm

A probe along the old far edge (y = 3.581, x 11.023..14.420) finds 0 / 69 ink
hits, so it was never drawn.

What the mask *pixels* are checked against
------------------------------------------
Every measurement above compares the metre-space polygon with the render, and
that is not enough.  An early version of to_px() used row = y * PX and forgot
that rows grow downwards, which mirrors the drawing about y = 14 and therefore
swaps the two trapezoids.  All the metrics above stayed perfect anyway --
rot180 IoU 1.000000, perimeter-on-ink 0.796 -- because the two trapezoids are
exact 180-degree images of each other, so the mask was symmetric about the
wrong axis and nothing self-referential could see it.  Two checks now read the
written pixels instead of the polygon:

* chroma support -- red ink exists only in y 0.51..14.28 and blue ink only in
  y 13.77..27.51 (measured), so the trapezoid in the red half has to be
  enclosed by red ink along all four of its drawn edges.  Measured: TRAP_R own
  1.000 / foreign 0.000, TRAP_B own 1.000 / foreign 0.000.  A swapped mask
  scores 0.000 own / 1.000 foreign, so the mistake cannot recur silently.
* assay points -- six fixed positions, inside and outside each of the three
  regions, that must classify exactly as written.

Accepted output of `python make_mask27.py --verify` (problems: []):

    green frac 0.137010    57.544 m2    3 components >= 0.5 m2
    TRAP_R 18.189 m2   TRAP_B 18.189 m2   CENTRE 20.466 m2
    perimeter-on-ink 0.796   drawn-perimeter 0.779   rot180 IoU 1.000000

Central highland
----------------
The centre has no drawn straight sides; its extent comes from two one-row
bright lips in the render (ramp crests), measured on a 0.1 m absolute-luminance
grid at y = 11.300 and y = 16.700, each 3.70 m wide.  That is cross-checked by
the two drawn 12-degree ramp-foot lines at y = 10.452 and y = 17.523: the
platform-to-ramp-foot distances are 0.848 m and 0.823 m, agreeing to 25 mm.
The span is 5.400 m exactly, centred on y = 14.000.  x keeps 5.610 / 9.400,
which is the field's width mid-line (7.505) to 5 mm and sits inside the lip
uncertainty.

Official rules facts used (RMUC 2027 rules V1.0.0, 4.3):
  highland = one trapezoidal highland per half + one central highland;
  trapezoidal highland is 200 mm above the field floor, reached by a 23 deg
  ramp;  the central highland joins the road area with 12 deg ramps at both
  ends.  Only those three regions belong in this mask -- the 制高点 and 基地区
  are separate and are deliberately excluded.

Usage
-----
    python make_mask27.py            # write the mask + check overlay
    python make_mask27.py --verify   # also print the numeric acceptance report
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(HERE, "images-2027")
RENDER = os.path.join(MAP_DIR, "red.jpg")
MASK_PNG = os.path.join(MAP_DIR, "map_mask27.png")
CHECK_PNG = os.path.join(MAP_DIR, "mask_check_2027.png")
GEOM_JSON = os.path.join(MAP_DIR, "highland_2027.json")

PX = 100.0          # pixels per metre, both for red.jpg and for the mask
W, H = 1500, 2800   # mask size: 15.00 m x 28.00 m
FIELD_W, FIELD_H = 15.0, 28.0
CX, CY = 7.5, 14.0  # field centre -> the 180 deg symmetry point

GREEN = (0, 255, 0)   # OpenCV is BGR, so this is pure green (0,255,0)
BLACK = (0, 0, 0)

# --------------------------------------------------------------------------
# Measured structure lines, metres, render frame.  Stroke centres.
# --------------------------------------------------------------------------
FAR_Y = 4.259       # H1  far edge of the red trapezoid
INNER_X = 11.023    # V1  inner edge, lower run
INNER_TOP_Y = 7.050 #     where V1 stops (the inner edge slants from here)
MID_X = 13.520      # V3  inner edge, upper run
MID_BOT_Y = 10.060  #     where V3 starts
OUTER_X = 14.420    # V2  outer edge (inner face of the road area)
NEAR_Y = 12.550     # CAP near edge; V2 and V3 both terminate here

TRAP_R = [
    (OUTER_X, FAR_Y),
    (INNER_X, FAR_Y),
    (INNER_X, INNER_TOP_Y),
    (MID_X, MID_BOT_Y),      # slant: undrawn, minimum closure
    (MID_X, NEAR_Y),
    (OUTER_X, NEAR_Y),
]

# Central highland: y from the two measured ramp crests, x symmetric about the
# field's width mid-line.
CENTRE = [
    (5.610, 11.300),
    (9.400, 11.300),
    (9.400, 16.700),
    (5.610, 16.700),
]

TOL_PX = 5.0    # ink probe / boundary support tolerance (+-5 cm)


def rot180(poly):
    """Rotate a metre-space polygon 180 deg about the field centre."""
    return [(round(2 * CX - x, 4), round(2 * CY - y, 4)) for x, y in reversed(poly)]


TRAP_B = rot180(TRAP_R)   # blue-half trapezoid, exact symmetric image

ASSAYS = [
    # label, x, y, expected green.  These exist because a trapezoid written into
    # the wrong half passes every symmetry check yet fails an assay point.
    ("red_in", 13.00, 8.00, True),
    ("red_out", 10.00, 8.00, False),
    ("blue_in", 2.00, 20.00, True),
    ("blue_out", 5.00, 20.00, False),
    ("centre_in", 7.50, 14.00, True),
    ("centre_out", 7.50, 10.00, False),
]


def to_px(poly):
    """Metres -> integer pixel polygon (col, row) for cv2.fillPoly.

    Rows grow downwards while the field's y grows upwards, so
    row = (FIELD_H - y) * PX.  Dropping that flip reflects the drawing about
    the field's horizontal centre line: the centre box (symmetric about
    y = FIELD_H / 2) survives untouched, so the two trapezoids swap halves
    silently and every symmetry metric still reads perfect.
    """
    return np.array([[int(round(x * PX)), int(round((FIELD_H - y) * PX))]
                     for x, y in poly], np.int32)


def shoelace(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def perimeter(poly):
    p = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        p += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return p


# --------------------------------------------------------------------------
# Measurement side: recover the drawn lines from the render.
# --------------------------------------------------------------------------
def load_chroma():
    """Separate the two ink families.

    Red ink is drawn in the red half only (y 0.51..14.28) and blue ink in the
    blue half (y 13.77..27.51), so the chroma of the nearest stroke states
    which half a polygon has to live in.  This is the one measurement that can
    tell the two trapezoids apart -- they are exact 180 deg images.
    """
    im = cv2.imread(RENDER, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if im is None or im.shape[:2] != (H, W):
        return None, None
    b, g, r = [im[:, :, i].astype(np.float32) for i in range(3)]
    return (r - 0.5 * (g + b)) > 45.0, (b - 0.5 * (g + r)) > 45.0


def load_ink():
    red, blue = load_chroma()
    return None if red is None else (red | blue)


def _row_of(y):
    return (FIELD_H - y) * PX


def measure_centre_h(ink, y0, x0, x1, half=0.12):
    """Median sub-pixel row centre of a horizontal stroke near y0."""
    r0 = int(round(_row_of(y0 + half)))
    r1 = int(round(_row_of(y0 - half)))
    vals = []
    for c in range(int(round(x0 * PX)), int(round(x1 * PX)) + 1):
        col = ink[max(0, r0):r1 + 1, c]
        if col.sum() < 2:
            continue
        idx = np.nonzero(col)[0]
        vals.append(FIELD_H - (r0 + float(idx.mean())) / PX)
    if not vals:
        return None, None
    v = np.array(vals)
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


def measure_centre_v(ink, x0, y0, y1, half=0.12):
    """Median sub-pixel column centre of a vertical stroke near x0."""
    c0 = int(round((x0 - half) * PX))
    c1 = int(round((x0 + half) * PX))
    vals = []
    for r in range(int(round(_row_of(y1))), int(round(_row_of(y0))) + 1):
        if r < 0 or r >= H:
            continue
        row = ink[r, max(0, c0):c1 + 1]
        if row.sum() < 2:
            continue
        idx = np.nonzero(row)[0]
        vals.append((c0 + float(idx.mean())) / PX)
    if not vals:
        return None, None
    v = np.array(vals)
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


def measure_stroke_end_v(ink, x0, y_from, y_to, half=0.05, taper=0.85):
    """Centreline y at which a vertical stroke centred on x0 stops.

    Use this where the drawing fixes a stroke's *extent* rather than its
    position: walk from y_from towards y_to and take the last station that
    still carries ink.  The strokes here are drawn with round caps, so that
    extreme pixel lies half a stroke width past the centreline end; the cap
    radius is measured from the local ink width and subtracted again, unless
    the end is square (no taper).  A few missing rows are tolerated, so a
    stroke that another stroke crosses is not truncated by the crossing.
    """
    c0 = int(round((x0 - half) * PX))
    c1 = int(round((x0 + half) * PX))
    r_from = int(round(_row_of(y_from)))
    r_to = int(round(_row_of(y_to)))
    step = 1 if r_to >= r_from else -1
    widths = []
    for i in range(abs(r_to - r_from) + 1):
        r = r_from + i * step
        if not (0 <= r < H):
            break
        widths.append(int(ink[r, max(0, c0):c1 + 1].sum()))
    if not widths:
        return None
    tip = None
    miss = 0
    for i, w in enumerate(widths):
        if w:
            tip = i
            miss = 0
        else:
            miss += 1
            if miss > 3:
                break
    if tip is None:
        return None
    inner = [w for w in widths[:max(1, int(0.6 * len(widths)))] if w]
    full = float(np.median(inner)) if inner else float(widths[tip])
    r_cap = 0.5 * full if (full >= 4.0 and widths[tip] / full <= taper) else 0.0
    row_end = r_from + tip * step - step * r_cap
    return FIELD_H - row_end / PX


def ink_hit_fraction(ink, p0, p1, half=PX * 0.03 / PX):
    """Fraction of stations along p0->p1 with ink within +-half metres."""
    n = max(2, int(round(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.05)) + 1)
    hit = 0
    for i in range(n):
        t = i / (n - 1)
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        c0 = int(round((x - half) * PX)); c1 = int(round((x + half) * PX))
        r0 = int(round(_row_of(y + half))); r1 = int(round(_row_of(y - half)))
        sub = ink[max(0, r0):r1 + 1, max(0, c0):c1 + 1]
        hit += 1 if sub.any() else 0
    return hit / float(n), n


def perimeter_ink_support(ink, poly, tol=TOL_PX):
    """Fraction of the polygon outline that lies on drawn ink (+-tol px)."""
    near = cv2.dilate(ink.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (int(2 * tol + 1), int(2 * tol + 1)))) > 0
    n = len(poly)
    tot = on = 0
    per_edge = []
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        steps = max(2, int(round(np.hypot(x2 - x1, y2 - y1) * PX)))
        e_on = 0
        for k in range(steps):
            t = k / float(steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            c = int(round(x * PX)); r = int(round(_row_of(y)))
            if 0 <= r < H and 0 <= c < W and near[r, c]:
                e_on += 1
        per_edge.append(e_on / float(steps))
        on += e_on
        tot += steps
    return on / float(tot), per_edge


def segment_ink_fraction(ink, p0, p1, tol=TOL_PX):
    """Fraction of 5 cm stations along p0->p1 with ink within +-tol px.

    Same sampling as perimeter_ink_support, but restricted to one segment so
    that the two ink families can be scored separately.
    """
    near = cv2.dilate(ink.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (int(2 * tol + 1), int(2 * tol + 1)))) > 0
    n = max(2, int(round(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.05)) + 1)
    hits = 0
    for k in range(n):
        t = k / (n - 1.0)
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        c = int(round(x * PX)); r = int(round(_row_of(y)))
        if 0 <= r < H and 0 <= c < W and near[r, c]:
            hits += 1
    return hits / float(n)


# --------------------------------------------------------------------------
# Mask construction.
# --------------------------------------------------------------------------
def build():
    mask = np.zeros((H, W, 3), np.uint8)
    for poly in (TRAP_R, TRAP_B, CENTRE):
        cv2.fillPoly(mask, [to_px(poly)], GREEN, lineType=cv2.LINE_8)

    # Enforce the physical 180 deg symmetry exactly at pixel level.
    g = (mask[:, :, 1] > 127)
    g_sym = g | g[::-1, ::-1]
    mask[g_sym] = GREEN
    return mask, int(g_sym.sum()) - int(g.sum())


def boundary_pixels(green):
    """1-px inner boundary of the green region."""
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(green.astype(np.uint8), k, iterations=1)
    return (green.astype(np.uint8) - er).astype(bool)


def verify(mask, sym_delta):
    rep = {}
    green = mask[:, :, 1] > 127
    colors = np.unique(mask.reshape(-1, 3), axis=0)
    rep["unique_colors"] = [tuple(int(v) for v in c) for c in colors]
    rep["green_px"] = int(green.sum())
    rep["green_frac"] = float(green.sum()) / (H * W)
    rep["area_total_m2"] = rep["green_px"] / (PX * PX)
    rep["symmetry_px_added"] = sym_delta
    rep["per_polygon"] = {}
    for name, poly in (("TRAP_R", TRAP_R), ("TRAP_B", TRAP_B), ("CENTRE", CENTRE)):
        rep["per_polygon"][name] = {
            "area_m2": round(shoelace(poly), 3),
            "perimeter_m": round(perimeter(poly), 3),
            "bbox": [
                round(min(p[0] for p in poly), 3), round(max(p[0] for p in poly), 3),
                round(min(p[1] for p in poly), 3), round(max(p[1] for p in poly), 3),
            ],
        }

    ink = load_ink()
    if ink is None:
        rep["lines"] = None
        rep["perimeter_ink_support"] = None
    else:
        # --- re-derive every drawn edge from the render, report the residual --
        old = {"far_y": 3.581, "inner_x": 10.733, "inner_top_y": 7.890,
               "mid_x": 13.480, "mid_bot_y": 9.850, "near_y": 12.970,
               "outer_x": 14.507}
        new = {"far_y": FAR_Y, "inner_x": INNER_X, "inner_top_y": INNER_TOP_Y,
               "mid_x": MID_X, "mid_bot_y": MID_BOT_Y, "near_y": NEAR_Y,
               "outer_x": OUTER_X}
        # Lines whose *position* the drawing fixes: take the stroke centre.
        plan = [
            ("far_y", "h", (FAR_Y, INNER_X + 0.05, OUTER_X - 0.05)),
            ("inner_x", "v", (INNER_X, FAR_Y + 0.05, INNER_TOP_Y - 0.05)),
            ("mid_x", "v", (MID_X, MID_BOT_Y + 0.05, NEAR_Y - 0.05)),
            ("outer_x", "v", (OUTER_X, FAR_Y + 0.05, NEAR_Y - 0.05)),
        ]
        rows = {}
        for key, kind, args in plan:
            if kind == "h":
                med, mad = measure_centre_h(ink, *args)
            else:
                med, mad = measure_centre_v(ink, *args)
            rows[key] = {
                "used": new[key],
                "measured": None if med is None else round(med, 3),
                "mad_mm": None if mad is None else round(mad * 1000.0, 1),
                "residual_mm": None if med is None else round((new[key] - med) * 1000.0, 1),
                "old_value": old[key],
                "old_error_mm": None if med is None else round((old[key] - med) * 1000.0, 1),
            }
        rep["lines"] = rows

        # Lines whose *extent* the drawing fixes: where a stroke stops.  V1
        # stops at the slant foot, V3 starts at its own foot, and V2/V3 both
        # stop short of the near field boundary.  The near edge is never drawn:
        # it exists only as the common terminus of V2 and V3.
        span_plan = [
            ("V1_end_y", "inner_top_y", INNER_X, INNER_TOP_Y - 1.0, INNER_TOP_Y + 1.0),
            ("V3_start_y", "mid_bot_y", MID_X, MID_BOT_Y + 1.0, MID_BOT_Y - 1.0),
            ("V2_end_y", "near_y", OUTER_X, NEAR_Y - 1.2, NEAR_Y + 0.7),
            ("V3_end_y", "near_y", MID_X, NEAR_Y - 1.2, NEAR_Y + 0.7),
        ]
        termini = {}
        for key, target, x, y_from, y_to in span_plan:
            got = measure_stroke_end_v(ink, x, y_from, y_to)
            ref = new[target]
            termini[key] = {
                "target": target,
                "used": ref,
                "measured": None if got is None else round(got, 3),
                "residual_mm": None if got is None else round((ref - got) * 1000.0, 1),
            }
        rep["termini"] = termini
        ends = [termini[k]["measured"] for k in ("V2_end_y", "V3_end_y")]
        rep["near_y_from_termini"] = (round(float(np.mean(ends)), 3)
                                      if None not in ends else None)
        fr, n = ink_hit_fraction(ink, (INNER_X, FAR_Y), (OUTER_X, FAR_Y))
        rep["probe_far_edge"] = {"hit": round(fr, 3), "stations": n}
        fr, n = ink_hit_fraction(ink, (INNER_X, INNER_TOP_Y), (MID_X, MID_BOT_Y))
        rep["probe_slant"] = {"hit": round(fr, 3), "stations": n}
        fr, n = ink_hit_fraction(ink, (INNER_X, 3.581), (OUTER_X, 3.581))
        rep["probe_old_far_edge"] = {"hit": round(fr, 3), "stations": n}

        sup, per_edge = perimeter_ink_support(ink, TRAP_R)
        rep["perimeter_ink_support"] = round(sup, 4)
        rep["perimeter_ink_support_per_edge"] = [round(v, 3) for v in per_edge]
        rep["edges"] = ["far", "inner", "slant", "mid", "cap", "outer"]
        rep["slant_len_m"] = round(np.hypot(MID_X - INNER_X, MID_BOT_Y - INNER_TOP_Y), 3)
        rep["drawn_perimeter_frac"] = round(
            (perimeter(TRAP_R) - rep["slant_len_m"] - (OUTER_X - MID_X)) / perimeter(TRAP_R), 4)

        g = cv2.cvtColor(cv2.imread(RENDER, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
                         cv2.COLOR_BGR2GRAY)
        gs = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
        mag = cv2.magnitude(cv2.Sobel(gs, cv2.CV_32F, 1, 0, 3),
                            cv2.Sobel(gs, cv2.CV_32F, 0, 1, 3))
        strong = (mag > float(np.percentile(mag, 88))).astype(np.uint8)
        near_e = cv2.dilate(strong, np.ones((5, 5), np.uint8)) > 0
        bp = boundary_pixels(green)
        rep["boundary_px"] = int(bp.sum())
        rep["boundary_on_edge_frac"] = float(np.logical_and(bp, near_e).sum()) / max(1, int(bp.sum()))
        rep["edge_density_frac"] = float(strong.mean())
    # --- the written mask itself must sit on the drawing ---------------------
    # Everything above compares the metre-space polygon against the render, so
    # it is blind to a wrong metre->pixel mapping: a trapezoid drawn into the
    # other half still scores rot180 IoU 1.0 and 0.99 perimeter ink support,
    # because the two trapezoids are exact 180 deg images of each other.  These
    # checks look at the mask pixels instead.
    def _at(x, y):
        return bool(green[int(round((FIELD_H - y) * PX)), int(round(x * PX))])

    rep["assays"] = {name: _at(ax, ay) for name, ax, ay, _want in ASSAYS}
    if ink is not None:
        dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5) / PX
        bp_all = boundary_pixels(green)
        per_region = {}
        for name, poly in (("TRAP_R", TRAP_R), ("TRAP_B", TRAP_B), ("CENTRE", CENTRE)):
            raster = cv2.fillPoly(np.zeros((H, W), np.uint8), [to_px(poly)], 1) > 0
            sel = bp_all & raster
            if not sel.any():
                per_region[name] = None
                continue
            d = dist[sel]
            per_region[name] = {
                "n": int(sel.sum()),
                "p50_mm": round(float(np.percentile(d, 50)) * 1000.0, 1),
                "p90_mm": round(float(np.percentile(d, 90)) * 1000.0, 1),
                "on_ink_frac": round(float((d <= 0.06).mean()), 3),
            }
        rep["boundary_to_ink"] = per_region

        # Chroma of the nearest stroke: the decisive half-swap check.  The
        # boundary_to_ink above cannot serve that purpose -- on the 50 deg
        # slant the rasterised staircase lets the green edge bulge one pixel
        # outside the ideal polygon, so those ring pixels drop out of `raster`
        # in the red half and stay in it in the blue half.  That asymmetry is
        # rounding, not geometry, which is why it is reported but never asserted.
        red_ch, blue_ch = load_chroma()
        drawn = [
            ((OUTER_X, FAR_Y), (INNER_X, FAR_Y)),
            ((INNER_X, FAR_Y), (INNER_X, INNER_TOP_Y)),
            ((MID_X, MID_BOT_Y), (MID_X, NEAR_Y)),
            ((OUTER_X, NEAR_Y), (OUTER_X, FAR_Y)),
        ]
        segs = {"TRAP_R": drawn,
                "TRAP_B": [tuple(rot180([a, b])) for a, b in drawn]}
        chroma = {}
        for name, ss in segs.items():
            own, other = (red_ch, blue_ch) if name == "TRAP_R" else (blue_ch, red_ch)
            row = {}
            for label, ch_img in (("own", own), ("other", other)):
                vals = [segment_ink_fraction(ch_img, a, b) for a, b in ss]
                row[label] = {"per_edge": [round(v, 3) for v in vals],
                              "mean": round(float(np.mean(vals)), 4)}
            chroma[name] = row
        rep["chroma_support"] = chroma

    # --- exact 180 deg self check -------------------------------------------
    a = green
    b = green[::-1, ::-1]
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    rep["rot180_self_iou"] = inter / union if union else 0.0

    # --- contract checks ----------------------------------------------------
    problems = []
    if rep["unique_colors"] != [(0, 0, 0), (0, 255, 0)]:
        problems.append("mask is not strictly bilevel: %s" % (rep["unique_colors"],))
    if rep["rot180_self_iou"] < 0.999:
        problems.append("180 deg symmetry broken: %.6f" % rep["rot180_self_iou"])
    for key, row in (rep.get("lines") or {}).items():
        if row["residual_mm"] is None:
            problems.append("could not re-measure line %s" % key)
        elif abs(row["residual_mm"]) > 20.0:
            problems.append("line %s off by %.1f mm" % (key, row["residual_mm"]))
    for key, row in (rep.get("termini") or {}).items():
        if row["residual_mm"] is None:
            problems.append("could not re-measure stroke terminus %s" % key)
        # +-4 px = +-40 mm is the native resolution of an 8 px stroke end
        elif abs(row["residual_mm"]) > 40.0:
            problems.append("terminus %s off by %.1f mm" % (key, row["residual_mm"]))
    # The superseded JPEG copy gave half the field away.  Never allow that again.
    if not (0.10 <= rep["green_frac"] <= 0.20):
        problems.append("green fraction %.4f outside the plausible 0.10-0.20 band"
                        % rep["green_frac"])
    pe = rep.get("perimeter_ink_support_per_edge")
    if pe:
        for name, val in zip(rep["edges"], pe):
            if name in ("far", "inner", "mid", "outer"):
                if val < 0.90:
                    problems.append("drawn edge %s only %.0f%% on ink" % (name, val * 100.0))
            elif val > 0.50:
                problems.append("closure edge %s unexpectedly %.0f%% on ink"
                                % (name, val * 100.0))
    if (rep.get("probe_old_far_edge") or {}).get("hit", 0.0) > 0.10:
        problems.append("the superseded far edge y=3.581 carries ink again; re-audit")
    ch = rep.get("chroma_support") or {}
    for name, own, other in (("TRAP_R", "red", "blue"), ("TRAP_B", "blue", "red")):
        row = ch.get(name) or {}
        ownm = (row.get("own") or {}).get("mean")
        othm = (row.get("other") or {}).get("mean")
        if ownm is None or othm is None:
            problems.append("could not measure ink chroma for %s" % name)
            continue
        # The two trapezoids are exact 180 deg images of each other, so every
        # geometric and symmetry metric is blind to swapping them: rot180 IoU
        # stays 1.000 and the metre-space ink support stays 0.796.  The ink is
        # not blind -- red ink is drawn in the red half only (y 0.51..14.28),
        # blue ink in the blue half (y 13.77..27.51) -- so a swapped mask scores
        # own-chroma 0.00 and foreign-chroma 1.00 on all four drawn edges.
        if ownm < 0.85:
            problems.append("region %s: own-chroma ink support %.2f < 0.85" % (name, ownm))
        if othm > 0.15:
            problems.append("region %s: foreign-chroma ink support %.2f > 0.15"
                            " -- trapezoid sits in the wrong half" % (name, othm))
    for name, ax, ay, want in ASSAYS:
        got = (rep.get("assays") or {}).get(name)
        if got != want:
            problems.append("assay %s(%.2f, %.2f) = %s, expected %s"
                            % (name, ax, ay, got, want))
    rep["problems"] = problems
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="print the numeric report")
    args = ap.parse_args()

    mask, sym_delta = build()
    ok = cv2.imwrite(MASK_PNG, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # human-checkable overlay: render + green wash on the highland layer
    if os.path.exists(RENDER):
        base = cv2.imread(RENDER, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if base is not None and base.shape == mask.shape:
            ov = base.copy()
            green = mask[:, :, 1] > 127
            ov[green] = (0.45 * ov[green] + 0.55 * np.array(GREEN, np.float32)).astype(np.uint8)
            for poly in (TRAP_R, TRAP_B, CENTRE):
                cv2.polylines(ov, [to_px(poly)], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(CHECK_PNG, ov, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    geom = {"frame": {"px_per_m": PX, "width_px": W, "height_px": H,
                      "field_m": [FIELD_W, FIELD_H], "centre_m": [CX, CY]},
            "measured_lines": {"far_y": FAR_Y, "inner_x": INNER_X,
                               "inner_top_y": INNER_TOP_Y, "mid_x": MID_X,
                               "mid_bot_y": MID_BOT_Y, "near_y": NEAR_Y,
                               "outer_x": OUTER_X},
            "TRAP_R": TRAP_R, "TRAP_B": TRAP_B, "CENTRE": CENTRE}
    with open(GEOM_JSON, "w", encoding="utf-8") as f:
        json.dump(geom, f, indent=2)

    rep = verify(mask, sym_delta)
    rep["wrote"] = {"mask": MASK_PNG, "check": CHECK_PNG, "geom": GEOM_JSON,
                    "imwrite_ok": bool(ok)}
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if rep["problems"]:
        raise SystemExit("MASK REJECTED: " + "; ".join(rep["problems"]))
    print("\nOK  green %.4f  area %.2f m2  rot180 IoU %.6f  perimeter-on-ink %.3f  drawn-perimeter %.3f"
          % (rep["green_frac"], rep["area_total_m2"], rep["rot180_self_iou"],
             rep["perimeter_ink_support"] or 0.0, rep["drawn_perimeter_frac"]))


if __name__ == "__main__":
    main()
