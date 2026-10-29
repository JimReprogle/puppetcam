#!/usr/bin/env python3
"""puppetcam: real-time felt-puppet face filter -> virtual camera (Teams/Zoom/etc).

Oval felt head, natural skin tone, glasses, goatee, heavy-lidded ping-pong eyes.
Head turn/nod: 2.5D parallax - every feature has a depth and is rotated by your yaw/pitch.
Fleece look: fine pile + coarse fibers, velvet shading (bright rims, not dark), fiber-displaced
color edges, and a fuzzy fiber halo on the silhouette.
"""
import argparse
import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions, ImageSegmenter,
                                           ImageSegmenterOptions, RunningMode)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

S = 512                 # local puppet canvas (square); on-screen head is usually smaller
ANCHOR = (0.50, 0.40)   # canvas point pinned to the real face center
PADT, PADB = int(0.19 * S), int(0.08 * S)   # headroom for the Pez head lift / jaw drop
OUT_H = S + PADT + PADB
LIGHT = np.array([-0.35, -0.55, 0.76], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
MAX_YAW, MAX_PITCH = math.radians(38), math.radians(25)
MOUTH_OPEN = 0.06        # below this the mouth is a closed line, above it the head hinges
DEPTH = 0.6             # parallax depth scale

# BGR felt colors
SKIN = {"light": (150, 180, 225), "medium": (105, 150, 210),
        "tan": (80, 125, 185), "deep": (55, 85, 135)}
HAIR = {"gray": (165, 165, 170), "salt": (125, 130, 135), "white": (225, 225, 230),
        "brown": (40, 65, 95), "black": (35, 35, 40)}

# FaceLandmarker indices (478-point mesh, iris centers 468/473)
R_OUT, R_IN, R_IRIS = 33, 133, 468
L_IN, L_OUT, L_IRIS = 362, 263, 473
R_UP, R_DN, L_UP, L_DN = 159, 145, 386, 374
LIP_UP, LIP_DN, FOREHEAD, CHIN, CHEEK_R, CHEEK_L, NOSE_TIP = 13, 14, 10, 152, 234, 454, 1


def L(v):
    return max(1, int(v * S))


def shade(col, k):
    return tuple(min(255, max(0, int(v * k))) for v in col)


class Smooth:
    def __init__(self, a):
        self.a, self.v = a, None

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        self.v = x if self.v is None else self.a * x + (1 - self.a) * self.v
        return self.v


class Prof:
    """--profile: wall time per pipeline stage; prints mean / p95 ms every `every` frames."""

    def __init__(self, on, every=150):
        self.on, self.every, self.t, self.frames, self.d = on, every, 0.0, 0, {}

    def start(self):
        if self.on:
            self.t = self.t0 = time.perf_counter()

    def lap(self, name):
        if not self.on:
            return
        now = time.perf_counter()
        self.d.setdefault(name, []).append((now - self.t) * 1000)
        self.t = now

    def end(self):
        if not self.on:
            return
        self.d.setdefault("frame", []).append((time.perf_counter() - self.t0) * 1000)
        self.frames += 1
        if self.frames % self.every == 0:
            self.report()

    def report(self):
        if not self.on or not self.d:
            return
        print(f"--- profile, last {len(self.d.get('frame', []))} frames (ms: mean / p95 / samples)")
        for name, v in self.d.items():
            a = np.array(v)
            print(f"  {name:12s} {a.mean():7.2f} {np.percentile(a, 95):7.2f} {len(a):5d}")
        self.d = {}


# ---------------------------------------------------------------- fleece material (built once)

def _strokes(rng, n, lmin, lmax, bright_frac):
    pos = np.zeros((S, S), np.uint8)
    neg = np.zeros((S, S), np.uint8)
    xs, ys = rng.uniform(0, S, n), rng.uniform(0, S, n)
    ang = rng.uniform(0, math.pi, n)
    ln = rng.uniform(lmin, lmax, n)
    val = rng.integers(80, 255, n)
    bright = rng.random(n) < bright_frac
    for x, y, a, l, v, b in zip(xs, ys, ang, ln, val, bright):
        cv2.line(pos if b else neg, (int(x), int(y)),
                 (int(x + l * math.cos(a)), int(y + l * math.sin(a))), int(v), 1, cv2.LINE_AA)
    f = (pos.astype(np.float32) - neg.astype(np.float32)) / 255.0
    return f / (np.abs(f).max() + 1e-6)


def build_felt(seed=7):
    rng = np.random.default_rng(seed)
    pile = cv2.GaussianBlur(_strokes(rng, 70000, 2, 6, 0.7), (0, 0), 0.7)     # fine fuzz, mostly bright tips
    fiber = cv2.GaussianBlur(_strokes(rng, 9000, 6, 16, 0.5), (0, 0), 0.8)    # longer loose fibers
    mottle = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 14)
    mottle /= mottle.std() + 1e-6
    for f in (pile, fiber):
        f /= np.abs(f).max() + 1e-6
    # fiber-driven displacement field: color edges interpenetrate like real fleece seams
    dx = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 1.2)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 1.2)
    gx, gy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    mapx = gx + dx / (dx.std() + 1e-6) * 1.3
    mapy = gy + dy / (dy.std() + 1e-6) * 1.3
    return {"pile": pile, "fiber": fiber, "mottle": mottle, "mapx": mapx, "mapy": mapy}


def velvet_shading(mask, bumps=None, relief=2.0, base=0.62, key=0.55):
    """Fleece reflectance: soft Lambert plus a grazing-angle sheen. Fuzz scatters light at the
    rims, so edges get BRIGHTER - that (not dark edges) is what separates felt from cork/leather."""
    h, w = mask.shape
    hs, ws = h // 2, w // 2
    small = cv2.resize(mask, (ws, hs), interpolation=cv2.INTER_AREA)
    d = cv2.distanceTransform((small > 127).astype(np.uint8), cv2.DIST_L2, 5)
    height = np.sqrt(d) * 2.2
    if bumps is not None:
        height += cv2.resize(bumps, (ws, hs), interpolation=cv2.INTER_AREA)
    height = cv2.GaussianBlur(height, (0, 0), 3.0)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) * 0.18 * relief
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) * 0.18 * relief
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    lam = np.clip((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) * nz, 0, 1)
    rim = (1 - nz) ** 1.4
    sh = base + key * lam + 0.30 * rim
    return cv2.resize(sh, (w, h), interpolation=cv2.INTER_LINEAR)


_rng = np.random.default_rng(11)
BEARD_STRANDS = np.column_stack([_rng.uniform(-1, 1, 380), _rng.uniform(0, 1, 380),
                                 _rng.uniform(-0.3, 0.3, 380), _rng.integers(0, 3, 380)])
_rng = np.random.default_rng(9)
BROW_STRANDS = np.column_stack([_rng.uniform(-1, 1, 70), _rng.uniform(-1, 1, 70), _rng.integers(0, 3, 70)])
_rng = np.random.default_rng(5)
HAIR_STRANDS = np.column_stack([_rng.uniform(-1, 1, 160), _rng.uniform(0, 1, 160), _rng.integers(0, 3, 160)])


# ---------------------------------------------------------------- sculpted lighting

FILL = np.array([0.60, -0.10, 0.79], np.float32)
FILL /= np.linalg.norm(FILL)
_HALF = np.array([LIGHT[0], LIGHT[1], LIGHT[2] + 1.0], np.float32)
_HALF /= np.linalg.norm(_HALF)


def add_ellipsoid(hf, c, ax, ay, depth, n=2.0):
    """Add (depth>0) or carve (depth<0) a rounded (super)ellipsoid into a height field.
    c, ax, ay are in height-field pixels."""
    H, W = hf.shape
    ax, ay = max(ax, 1.0), max(ay, 1.0)
    x0, x1 = max(0, int(c[0] - ax - 1)), min(W, int(c[0] + ax + 2))
    y0, y1 = max(0, int(c[1] - ay - 1)), min(H, int(c[1] + ay + 2))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    r = (np.abs((xx - c[0]) / ax) ** n + np.abs((yy - c[1]) / ay) ** n) ** (1.0 / n)
    t = np.clip(1 - r * r, 0, 1)
    prof = np.sqrt(t) if n > 2.0 else t * t * (3 - 2 * np.sqrt(t))   # domed block vs. soft-rimmed bump
    hf[y0:y1, x0:x1] += (depth * prof).astype(np.float32)


def sculpt_light(hf, out_size):
    """Full lighting from a height field (pixel units):
    key (upper-left) + cool fill (right) + ambient occlusion in creases + height-field cast
    shadows + fleece rim sheen + a soft broad highlight. Returns (shade, spec) at out_size."""
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx * inv, -gy * inv, inv
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    fil = np.clip(nx * FILL[0] + ny * FILL[1] + nz * FILL[2], 0, 1)
    cav = cv2.GaussianBlur(hf, (0, 0), 7) - hf                  # >0 in creases and sockets
    ao = np.clip(1.0 - 0.045 * cav, 0.45, 1.0)
    # shadows: march toward the light; anything rising above the ray blocks it
    d = np.array(LIGHT[:2]) / np.linalg.norm(LIGHT[:2])
    tan_e = float(LIGHT[2] / np.linalg.norm(LIGHT[:2]))
    occ = np.zeros_like(hf)
    for k in (2, 4, 7, 10, 14, 19, 25, 32):
        M = np.float32([[1, 0, d[0] * k], [0, 1, d[1] * k]])
        ahead = cv2.warpAffine(hf, M, (hf.shape[1], hf.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        occ = np.maximum(occ, ahead - (hf + k * tan_e))
    shadow = cv2.GaussianBlur(np.clip(occ / 5.0, 0, 1), (0, 0), 1.5)
    rim = (1 - nz) ** 2
    shade = 0.30 * ao + 0.82 * lam * (1 - 0.80 * shadow) + 0.22 * fil * ao + 0.32 * rim
    nh = np.clip(nx * _HALF[0] + ny * _HALF[1] + nz * _HALF[2], 0, 1)
    spec = 0.16 * nh ** 14 * (1 - shadow)
    up = lambda a: cv2.resize(a, out_size, interpolation=cv2.INTER_LINEAR)
    return up(shade), up(spec)


# ---------------------------------------------------------------- 2.5D projection

def make_proj(yaw, pitch):
    """Rotate canvas point (x, y) at depth z (toward camera, canvas units) by yaw then pitch,
    orthographic. +yaw = nose toward image-right, +pitch = nose down."""
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ax, ay = ANCHOR

    def pr(x, y, z=0.0):
        X, Y, z = x - ax, y - ay, z * DEPTH
        X2, Z2 = X * cyw + z * syw, -X * syw + z * cyw
        Y2 = Y * cp + Z2 * sp
        return int((ax + X2) * S), int((ay + Y2) * S)
    return pr, cyw


# ---------------------------------------------------------------- parts

def hairline_y(u, hx, top):
    """Hairline height at head-local x offset u (-hx..+hx): high at the front, down to the ears."""
    return top + .04 + (.38 - top - .04) * (abs(u) / hx) ** 3


def hairline_z(u, hx):
    """Depth of the scalp at the hairline: forward at the center, at the silhouette on the sides."""
    return 0.20 * (1 - (u / hx) ** 2)


def draw_eye(c, ex, ey, r, lid, gaze, crease_col):
    y1 = int(ey - r + 2 * r * lid)
    ya, yb, xa, xb = ey - r - 2, ey + r + 3, ex - r - 2, ex + r + 3
    if ya < 0 or xa < 0 or yb > S or xb > S:
        return
    roi = c[ya:yb, xa:xb].astype(np.float32)
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    rr = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2) / r
    ball = 250 - 50 * rr ** 2
    ball *= 1 - 0.40 * np.clip(1 - (yy - y1) / (0.45 * r), 0, 1)
    px, py = ex + gaze[0] * r * 0.45, ey + gaze[1] * r * 0.30 + r * 0.15
    pa = np.clip(0.42 * r - np.sqrt((xx - px) ** 2 + (yy - py) ** 2) + 0.5, 0, 1)
    ball = ball * (1 - pa) + 18 * pa
    sa = np.clip(0.13 * r - np.sqrt((xx - (ex - 0.38 * r)) ** 2 + (yy - (ey + 0.05 * r)) ** 2) + 0.5, 0, 1)
    ball = ball * (1 - 0.8 * sa) + 255 * 0.8 * sa
    va = np.clip((1 - rr) * r + 0.5, 0, 1) * np.clip(yy - y1 + 0.5, 0, 1)
    out = roi * (1 - va[..., None]) + ball[..., None] * va[..., None]
    c[ya:yb, xa:xb] = np.clip(out, 0, 255).astype(np.uint8)
    if y1 < ey + r:
        hw = int(math.sqrt(max(0, r * r - (y1 - ey) ** 2)))
        cv2.ellipse(c, (ex, y1), (hw, 2), 0, 0, 180, crease_col, 2, cv2.LINE_AA)


def draw_glasses(c, pr, yaw, hx):
    fc, hl, th = (52, 52, 55), (120, 120, 125), L(.010)
    z = .27
    for sx in (-1, 1):
        cx = .5 + sx * .085
        quad = np.array([pr(cx - .075, .28, z), pr(cx + .075, .28, z),
                         pr(cx + .075, .385, z), pr(cx - .075, .385, z)], np.int32)
        lens = np.zeros(c.shape[:2], np.uint8)
        cv2.fillPoly(lens, [quad], 255)
        sel = lens > 0
        c[sel] = np.clip(c[sel].astype(np.float32) * 0.93 + 8, 0, 255).astype(np.uint8)
        cv2.polylines(c, [quad], True, fc, th, cv2.LINE_AA)
        cv2.line(c, tuple(quad[0] + (3, -th // 2 + 1)), tuple(quad[1] + (-3, -th // 2 + 1)), hl, 1, cv2.LINE_AA)
        if sx * yaw < 0.17:  # far-side temple arm disappears behind the head
            cv2.line(c, pr(cx + sx * .075, .30, z), pr(.5 + sx * hx, .31, -0.02), fc, th, cv2.LINE_AA)
    cv2.line(c, pr(.49, .315, z), pr(.51, .315, z), fc, th, cv2.LINE_AA)


def superellipse(center, ax, ay, n, num=120):
    """Boxy rounded outline (n=2 ellipse, n~3.5 Pez-dispenser block)."""
    t = np.linspace(0, 2 * np.pi, num, endpoint=False)
    ct, sn = np.cos(t), np.sin(t)
    x = center[0] + ax * np.sign(ct) * np.abs(ct) ** (2 / n)
    y = center[1] + ay * np.sign(sn) * np.abs(sn) ** (2 / n)
    return np.stack([x, y], 1).astype(np.int32)


def se_top(u, ax, ay, n):
    """Height of a superellipse's upper edge above its center at horizontal offset u."""
    return ay * max(0.0, 1 - abs(u / ax) ** n) ** (1 / n)


def over(dst_c, dst_a, src_c, src_a, y0):
    """Alpha-composite src (rows) onto dst starting at row y0 (clipped)."""
    h = src_c.shape[0]
    a0, a1 = max(0, y0), min(dst_c.shape[0], y0 + h)
    if a1 <= a0:
        return
    sc = src_c[a0 - y0:a1 - y0].astype(np.float32)
    sa = src_a[a0 - y0:a1 - y0][..., None]
    dst_c[a0:a1] = sc * sa + dst_c[a0:a1] * (1 - sa)
    dst_a[a0:a1] = sa[..., 0] + dst_a[a0:a1] * (1 - sa[..., 0])


def draw_headphones(c, m, hc, cax, cay, n, ears, syw, col):
    """Generic over-ear headphones. The band is drawn straight onto the head. Each cup (with its
    arm) goes on its own layer with a continuous front/behind weight, composited later against the
    head's final alpha - so as you turn, a cup slides behind the skull smoothly instead of popping.
    Returns (band mask, [(cup_bgr, cup_alpha, front_weight), ...])."""
    hp = np.zeros((S, S), np.uint8)
    dark, hi = shade(col, 0.70), shade(col, 1.9)
    th = L(.040)
    cx, cy = hc[0], hc[1] - L(.012)
    ax, ay = L(cax) + th // 2, L(cay) + th // 2
    t = np.linspace(np.pi + 0.30, 2 * np.pi - 0.30, 48)
    ct, sn = np.cos(t), np.sin(t)
    band = np.stack([cx + ax * np.sign(ct) * np.abs(ct) ** (2 / n),
                     cy + ay * np.sign(sn) * np.abs(sn) ** (2 / n)], 1).astype(np.int32)
    cv2.polylines(c, [band], False, col, th, cv2.LINE_AA)
    cv2.polylines(hp, [band], False, 255, th, cv2.LINE_AA)
    cv2.polylines(c, [band - (0, th // 3)], False, hi, max(1, th // 6), cv2.LINE_AA)   # top-edge sheen
    cv2.polylines(c, [band + (0, th // 3)], False, dark, max(1, th // 5), cv2.LINE_AA)  # underside
    ends = {-1: tuple(band[0]), 1: tuple(band[-1])}   # band[0] is the left end
    layers = []
    for (ec, _), sx in zip(ears, (-1, 1)):
        away = sx * syw                                 # >0: this side is turning away from camera
        front = float(np.clip(1.0 - (away - 0.05) / 0.15, 0, 1))  # 1 = in front (facing/near), 0 = behind
        cup_c = (ec[0] + sx * L(.035), ec[1] + L(.01))
        lc = np.zeros((S, S, 3), np.uint8)
        lm = np.zeros((S, S), np.uint8)
        top = (cup_c[0], cup_c[1] - L(.095))
        cv2.line(lc, ends[sx], top, dark, max(2, L(.014)), cv2.LINE_AA)
        cv2.line(lm, ends[sx], top, 255, max(2, L(.014)), cv2.LINE_AA)
        _cup(lc, lm, cup_c, sx, col, dark, hi)
        layers.append((lc, lm.astype(np.float32) / 255.0, front))
    return hp, layers


def composite_cups(c, a, layers):
    """Cups always draw opaque. A cup that is turning away is occluded by the head starting from the
    head's silhouette and sweeping inward as 'front' drops - a moving edge, never a see-through blend."""
    head = (a > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(head, cv2.DIST_L2, 3)          # px inside the head outline
    reach = float(L(.16))                                        # about one cup width
    cf = c.astype(np.float32)
    cup_m = np.zeros(a.shape, np.float32)
    for lc, la, front in layers:
        ys, xs = np.nonzero(la > 0)
        if not ys.size:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        occ = np.clip((dist[y0:y1, x0:x1] - front * reach) / 2.0 + 0.5, 0, 1) * (head[y0:y1, x0:x1] > 0)
        if front >= 1.0:
            occ[:] = 0
        vis = la[y0:y1, x0:x1] * (1 - occ)
        cf[y0:y1, x0:x1] = lc[y0:y1, x0:x1] * vis[..., None] + cf[y0:y1, x0:x1] * (1 - vis[..., None])
        a[y0:y1, x0:x1] = np.maximum(a[y0:y1, x0:x1], vis)
        cup_m[y0:y1, x0:x1] = np.maximum(cup_m[y0:y1, x0:x1], vis)
    return np.clip(cf, 0, 255).astype(np.uint8), a, cup_m


def _cup(c, mk, ctr, sx, col, dark, hi):
    ax, ay = L(.075), L(.112)                      # larger cup, rounded-square (superellipse n=4)
    x0, x1 = max(0, ctr[0] - ax - 2), min(S, ctr[0] + ax + 3)
    y0, y1 = max(0, ctr[1] - ay - 2), min(S, ctr[1] + ay + 3)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u, v = (xx - ctr[0]) / ax, (yy - ctr[1]) / ay
    r = (np.abs(u) ** 4 + np.abs(v) ** 4) ** 0.25
    inside = np.clip((1 - r) * min(ax, ay) + 0.5, 0, 1)
    # domed shell lit from the upper left, darker cushion ring on the head side
    nz = np.sqrt(np.clip(1 - r * r, 0, 1))
    lam = np.clip(u * LIGHT[0] + v * LIGHT[1] + nz * LIGHT[2], 0, 1)
    shell = np.array(col, np.float32) * (0.35 + 0.75 * lam)[..., None]
    cushion = (r > 0.72) & ((u * -sx) > 0.15)
    shell[cushion] = np.array(dark, np.float32) * 0.8
    sp = np.clip(1 - np.hypot(u + 0.35, v + 0.45) / 0.22, 0, 1) ** 2
    shell = shell + np.array(hi, np.float32) * 0.5 * sp[..., None]
    roi = c[y0:y1, x0:x1].astype(np.float32)
    c[y0:y1, x0:x1] = np.clip(shell * inside[..., None] + roi * (1 - inside[..., None]), 0, 255).astype(np.uint8)
    mk[y0:y1, x0:x1] = np.maximum(mk[y0:y1, x0:x1], (inside * 255).astype(np.uint8))


def draw_puppet(st, cfg, felt):
    """Render the closed head, then split it on the mouth hinge like a Muppet / Pez head:
    the whole upper head tips back (rises + foreshortens), the jaw drops a little, and the
    cavity, tongue and stretched cheek fabric fill the gap."""
    skin, hair, beard = cfg["skin"], cfg["hair"], cfg["beard"]
    yaw, pitch = st["yaw"], st["pitch"]
    pr, fx = make_proj(yaw, pitch)
    syw = math.sin(yaw)
    c = np.zeros((S, S, 3), np.uint8)
    m = np.zeros((S, S), np.uint8)
    bumps = np.zeros((S, S), np.float32)
    cast = np.zeros((S, S), np.uint8)       # hard props that throw shadows (eyes, glasses)
    glasses_m = None
    fur_m = np.zeros((S, S), np.uint8)      # raised fur masses (brows, beard, mustache)
    ears = []

    def both(f, col):
        f(c, col)
        f(m, 255)

    y0 = 0.58                               # mouth hinge line
    n = 2.0 + 1.6 * cfg["pez"]              # head squareness
    hx = .25 - .05 * cfg["long"]
    hy = .33 + .025 * cfg["long"]
    hcy = .40
    htop = hcy - hy
    hxw = hx * (1 + 0.30 * abs(syw))
    mw = .17 * fx                           # wide Muppet mouth

    # no ears (the headphone cups cover where they'd be); just record the cup anchor points
    for sx in (-1, 1):
        away = max(0.0, sx * syw)
        ears.append((pr(.5 + sx * hxw * (1 - 0.45 * away), .40, -0.06), 0.0))
    hc = pr(.5, hcy, 0.06)
    both(lambda i, k: cv2.fillPoly(i, [superellipse(hc, L(hxw), L(hy), n)], k, cv2.LINE_AA), skin)

    cap = np.zeros((S, S), np.uint8)
    cax, cay = hxw + .018, hy + .022
    cv2.fillPoly(cap, [superellipse((hc[0], hc[1] - L(.012)), L(cax), L(cay), n)], 255, cv2.LINE_AA)
    line = [pr(.5 + u, hairline_y(u, hx, htop), hairline_z(u, hx)) for u in np.linspace(-hx, hx, 25)]
    (x0h, y0h), (x1h, y1h) = line[0], line[-1]
    poly = np.array(line + [(x1h + L(.1), y1h), (x1h + L(.1), -10), (x0h - L(.1), -10), (x0h - L(.1), y0h)],
                    np.int32)
    above = np.zeros((S, S), np.uint8)
    cv2.fillPoly(above, [poly], 255, cv2.LINE_AA)
    hair_m = cv2.min(cap, above)
    ha = (hair_m.astype(np.float32) / 255.0)[..., None]
    c = (c * (1 - ha) + np.array(hair, np.float32) * ha).astype(np.uint8)
    m = cv2.max(m, hair_m)

    # sculpted foam: big bulb nose, cheeks, brow ridge, lip roll
    nose = pr(.5, .47, .32)
    cv2.ellipse(c, nose, (L(.075 * (0.8 + 0.2 * fx)), L(.09)), 0, 0, 360, shade(skin, 0.90), -1, cv2.LINE_AA)
    cv2.ellipse(bumps, nose, (L(.07), L(.085)), 0, 0, 360, 24.0, -1)
    for sx in (-1, 1):
        cv2.ellipse(bumps, pr(.5 + sx * .13, .50, .16), (L(.08 * fx), L(.07)), 0, 0, 360, 7.0, -1)
    cv2.ellipse(bumps, pr(.5, .29, .20), (L(.20 * fx), L(.05)), 0, 0, 360, 5.0, -1)
    cv2.ellipse(bumps, pr(.5, y0 - .02, .22), (L(mw), L(.02)), 0, 0, 360, 3.0, -1)
    cv2.ellipse(bumps, pr(.5, y0 + .025, .22), (L(mw * .9), L(.022)), 0, 0, 360, 3.0, -1)
    bumps = cv2.GaussianBlur(bumps, (0, 0), 9)

    # brows: felt base (strands later), driven by browInnerUp / browDown
    bup, bdn = st["brow"]
    brows = []
    for sx in (-1, 1):
        bc = pr(.5 + sx * .09, .252 - .055 * bup + .030 * bdn, .24)
        ang = -sx * (36 * bdn - 28 * bup)
        brows.append((bc, ang))
        both(lambda i, k: cv2.ellipse(i, bc, (L(.064 * fx), L(.021)), ang, 0, 360, k, -1, cv2.LINE_AA), hair)
        cv2.ellipse(fur_m, bc, (L(.064 * fx), L(.021)), ang, 0, 360, 255, -1)

    # goatee (lower jaw) + mustache (upper lip, kept above the hinge)
    by = y0 + .02
    if cfg["beard_on"]:
        pts = [(.40, by, .20), (.60, by, .20), (.585, by + .10, .19), (.535, by + .22, .16), (.505, by + .30, .14),
               (.48, by + .22, .16), (.415, by + .10, .19)]
        poly = np.array([pr(*q) for q in pts], np.int32)
        both(lambda i, k: cv2.fillPoly(i, [poly], k, cv2.LINE_AA), beard)
        cv2.fillPoly(fur_m, [poly], 255)
        for sx in (-1, 1):
            cv2.ellipse(fur_m, pr(.5 + sx * .06, y0 - .022, .25), (L(.07 * fx), L(.017)), -sx * 12, 0, 360, 255, -1)
            both(lambda i, k: cv2.ellipse(i, pr(.5 + sx * .06, y0 - .022, .25), (L(.07 * fx), L(.017)), -sx * 12,
                                          0, 360, k, -1, cv2.LINE_AA), beard)

    # ---- fleece
    grow = cv2.dilate(c, np.ones((15, 15), np.uint8))
    c = np.where(m[..., None] > 0, c, grow)
    c = cv2.remap(c, felt["mapx"], felt["mapy"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    for sx in (-1, 1):
        cv2.ellipse(c, pr(.5 + sx * .028, .535, .30), (L(.015 * fx), L(.009)), 0, 0, 360, shade(skin, 0.5), -1,
                    cv2.LINE_AA)
    mc = pr(.5, y0, .22)
    bend = .014 + .045 * st.get("corners", 0.0)            # + smile, - frown
    curve = np.array([(mc[0] + int(u * L(mw)), mc[1] - int(L(1) * bend * u ** 4)) for u in np.linspace(-1, 1, 21)],
                     np.int32)
    if st["mouth"] < MOUTH_OPEN:
        cv2.polylines(c, [curve], False, shade(skin, 0.40), L(.006), cv2.LINE_AA)
    # open: the cavity is the whole mouth - no extra lines (corner creases read as a handlebar mustache)
    # sculpt: a height field built from foam-like primitives at half resolution
    h2 = S // 2
    hf = np.zeros((h2, h2), np.float32)
    q = lambda pt: (pt[0] / 2.0, pt[1] / 2.0)
    hw2 = L(hxw) / 2.0
    add_ellipsoid(hf, q(hc), hw2, L(hy) / 2.0, 0.85 * hw2, n=n)                  # head block, domed
    add_ellipsoid(hf, q(nose), L(.075) / 2.0, L(.09) / 2.0, 0.42 * hw2)           # bulb nose
    for sx in (-1, 1):
        add_ellipsoid(hf, q(pr(.5 + sx * .12, .50, .16)), L(.10 * fx) / 2.0, L(.085) / 2.0, 0.06 * hw2)   # cheeks
        add_ellipsoid(hf, q(pr(.5 + sx * .085, .33, .20)), L(.075) / 2.0, L(.06) / 2.0, -0.14 * hw2)    # sockets
    add_ellipsoid(hf, q(pr(.5, .285, .20)), L(.20 * fx) / 2.0, L(.045) / 2.0, 0.10 * hw2)              # brow ridge
    add_ellipsoid(hf, q(pr(.5, y0 - .018, .22)), L(mw) / 2.0, L(.022) / 2.0, 0.08 * hw2)              # upper lip
    add_ellipsoid(hf, q(pr(.5, y0 + .028, .22)), L(mw * .9) / 2.0, L(.024) / 2.0, 0.08 * hw2)         # lower lip
    add_ellipsoid(hf, q(pr(.5, .68, .18)), L(.10 * fx) / 2.0, L(.06) / 2.0, 0.10 * hw2)               # chin ball
    # (no smile-fold grooves: with head pitch they slid into the nose and read as lines through it)
    small = lambda mk: cv2.resize(mk, (h2, h2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    hair_s, fur_s = small(hair_m), small(fur_m)
    fib_s = cv2.resize(felt["fiber"], (h2, h2), interpolation=cv2.INTER_AREA)
    hf += cv2.GaussianBlur(hair_s, (0, 0), 2.0) * (0.12 * hw2 + 2.5 * fib_s)    # hair is a raised mass
    hf += cv2.GaussianBlur(fur_s, (0, 0), 1.5) * (0.07 * hw2 + 2.0 * fib_s)     # brows, beard, mustache
    hf *= small(m)                                                              # nothing outside the head
    s, spec = sculpt_light(hf, (S, S))
    form = s
    mod = (1.0 + 0.15 * felt["pile"] + 0.08 * felt["fiber"] + 0.05 * felt["mottle"]) * s
    c = np.clip(c.astype(np.float32) * mod[..., None] + 255.0 * spec[..., None], 0, 255).astype(np.uint8)
    c = cv2.GaussianBlur(c, (0, 0), 0.55)
    spot = np.zeros((S, S), np.float32)                      # soft sheen where the key light hits the nose
    cv2.circle(spot, (nose[0] - L(.024), nose[1] - L(.032)), L(.022), 1.0, -1)
    spot = cv2.GaussianBlur(spot, (0, 0), L(.012))[..., None]
    c = np.clip(c.astype(np.float32) + 55 * spot, 0, 255).astype(np.uint8)

    # hair, brow and beard fur
    hair_tones = [shade(hair, k) for k in (0.8, 1.0, 1.35)]
    brow_tones = [shade(hair, k) for k in (0.75, 1.0, 1.4)]
    beard_tones = [shade(beard, k) for k in (0.75, 1.0, 1.3)]
    sl = c.copy()
    for u, v, t in HAIR_STRANDS:
        uu = u * (hx - .005)
        top = hcy - .012 - se_top(uu, cax, cay, n)
        y = top + .01 + v * (hairline_y(uu, hx, htop) - top - .01)
        z = 0.7 * hairline_z(uu, hx)
        col = hair_tones[int(t)]
        cv2.line(sl, pr(.5 + uu, y, z), pr(.5 + uu * 1.06, y - .045, z * 0.8), col, 1, cv2.LINE_AA)
    clip = (cv2.erode(hair_m, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0)[..., None]
    c = (sl * clip + c * (1 - clip)).astype(np.uint8)
    for (bx, byy), ang in brows:
        ca, sa_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for u, v, t in BROW_STRANDS:
            ox, oy = u * L(.060 * fx), v * L(.019)
            x, y = bx + ox * ca - oy * sa_, byy + ox * sa_ + oy * ca
            col = brow_tones[int(t)]
            both(lambda i, k: cv2.line(i, (int(x), int(y)), (int(x + u * L(.012)), int(y - L(.016))), k, 1,
                                       cv2.LINE_AA), col)
    if cfg["beard_on"]:
        for u, v, dx, t in BEARD_STRANDS:
            half = .095 * (1 - v * .85)
            x, y, z = .5 + u * half, by + v * .27, .20 - v * .06
            col = beard_tones[int(t)]
            both(lambda i, k: cv2.line(i, pr(x, y, z), pr(x + dx * .02 - u * .004, y + .05, z), k, 1, cv2.LINE_AA),
                 col)

    # hard props
    for sx in (-1, 1):
        ex, ey = pr(.5 + sx * .085, .33, .20)
        draw_eye(c, ex, ey, L(.052), st["lid"], st["gaze"], shade(skin, 0.45))
        cv2.circle(cast, (ex, ey), L(.052), 255, -1)
    if cfg["glasses"]:
        draw_glasses(c, pr, yaw, hx)
        g = np.zeros((S, S, 3), np.uint8)
        draw_glasses(g, pr, yaw, hx)
        glasses_m = (g.max(2) > 0).astype(np.uint8) * 255
        cast = cv2.max(cast, glasses_m)

    hp_m, cup_layers, cup_m = None, [], None
    if cfg.get("headphones", True):
        hp_m, cup_layers = draw_headphones(c, m, hc, cax, cay, n, ears, syw, cfg.get("hp_color", (62, 62, 66)))
        cast = cv2.max(cast, hp_m)
        for _, la, front in cup_layers:              # cups shade the head only while in front of it
            cast = cv2.max(cast, (la * 255 * front).astype(np.uint8))

    # contact shadows: features are raised off the face, key light is upper-left
    tx = np.float32([[1, 0, L(.014)], [0, 1, L(.022)]])
    sh = cv2.GaussianBlur(cv2.warpAffine(cast, tx, (S, S)), (0, 0), L(.012)).astype(np.float32) / 255.0
    sh *= 1.0 - cast.astype(np.float32) / 255.0
    c = np.clip(c.astype(np.float32) * (1 - 0.42 * sh[..., None]), 0, 255).astype(np.uint8)

    # Antron-fleece halo: wide fuzzy silhouette with sheen
    soft = cv2.GaussianBlur(m, (0, 0), 3.6).astype(np.float32) / 255.0
    band = np.clip(1 - np.abs(soft - 0.5) * 2.2, 0, 1)
    c = np.clip(c.astype(np.float32) * (1 + 0.24 * band[..., None]), 0, 255).astype(np.uint8)
    a = np.clip((soft - 0.5) * 2.4 + 0.5 + felt["fiber"] * 0.65 + felt["pile"] * 0.30, 0, 1)
    a[soft > 0.80] = 1.0            # solid inside: fuzz noise was punching see-through holes in the face
    if hp_m is not None:
        a = np.maximum(a, hp_m.astype(np.float32) / 255.0)       # hard-edged, not fuzzy
    if glasses_m is not None:
        a = np.maximum(a, glasses_m.astype(np.float32) / 255.0)
    if cup_layers:
        c, a, cup_m = composite_cups(c, a, cup_layers)

    # ---- hinge split
    mo = st["mouth"]
    yh = int(np.clip(mc[1] + 1, 1, S - 1))
    is_open = mo >= MOUTH_OPEN
    lift, sq, drop = (L(0.17 * mo), 1 - 0.22 * mo, L(0.05 * mo)) if is_open else (0, 1.0, 0)
    oc = np.zeros((OUT_H, S, 3), np.float32)
    oa = np.zeros((OUT_H, S), np.float32)
    top_edge, bot_edge = yh + PADT - lift, yh + PADT + drop
    gap = bot_edge - top_edge
    if gap > 0:
        # cheek fabric stretches across the gap; the cavity sits inside it
        rows = slice(yh, yh + gap) if yh + gap <= S else slice(S - gap, S)
        # cheek fabric: the head's own shaded color at the hinge, fresh fleece texture, shadowed under the lip
        # built from skin x the head's shading at this height - NOT sampled from the image row, which
        # picked up the mustache and smeared it across the gap as gray streaks
        rowcol = (np.array(skin, np.float32)[None, None, :] *
                  cv2.blur(form[yh - 2:yh - 1], (21, 1))[..., None])
        grad = np.linspace(0.88, 1.0, gap, dtype=np.float32)[:, None, None]
        web = rowcol * grad * (1.0 + 0.12 * felt["pile"][rows] + 0.06 * felt["fiber"][rows])[..., None]
        # gap width = the HEAD's width at the hinge (felt mask only). Using the final alpha here
        # also picked up headphone cups / glasses when the hinge lined up with them -> a skin slab.
        head_row = np.minimum(m[max(0, yh - 3)], m[min(S - 1, yh + 2)]).astype(np.float32) / 255.0
        web_a = np.repeat(cv2.GaussianBlur(head_row[None, :], (0, 0), 1.5), gap, 0)
        cav = np.zeros((gap, S, 3), np.uint8)
        cav_m = np.zeros((gap, S), np.uint8)
        row = np.nonzero(head_row > 0.5)[0]            # Muppet mouth: nearly cheek to cheek
        half_w = max(L(mw), int(0.465 * (row[-1] - row[0]))) if row.size else L(mw)
        cx_cav = int(np.clip(mc[0], row[0] + half_w + 2, row[-1] - half_w - 2)) if row.size else mc[0]
        ctr = (cx_cav, gap // 2)
        axes = (half_w, max(1, gap // 2 + 2))
        cv2.ellipse(cav, ctr, axes, 0, 0, 360, (40, 22, 95), -1, cv2.LINE_AA)
        cv2.ellipse(cav_m, ctr, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(cav, (ctr[0], ctr[1] + gap // 8), (int(axes[0] * .65), int(axes[1] * .55)), 0, 0, 360,
                    (22, 10, 50), -1, cv2.LINE_AA)
        cv2.ellipse(cav, (ctr[0], gap), (int(axes[0] * .55), max(1, int(gap * .38))), 0, 180, 360,
                    (125, 105, 225), -1, cv2.LINE_AA)
        cv2.ellipse(cav, ctr, axes, 0, 0, 360, shade(skin, 0.55), max(2, L(.007)), cv2.LINE_AA)  # felt lip rim
        cv2.ellipse(cav_m, ctr, axes, 0, 0, 360, 255, max(2, L(.007)), cv2.LINE_AA)
        ca_ = (cav_m.astype(np.float32) / 255.0)[..., None]
        web = cav * ca_ + web * (1 - ca_)
        over(oc, oa, web, web_a, top_edge)
    # Headphone cups are attached to the UPPER head: any cup pixels below the hinge row travel
    # with the upper part (and are removed from the jaw), so no cup sliver is left behind on the jaw.
    ext = 0
    if cup_m is not None:
        rows_c = np.nonzero(cup_m[yh:].max(1) > 0.01)[0]
        ext = int(rows_c.max()) + 1 if rows_c.size else 0
    low_a = a[yh:] * (cup_m[yh:] < 0.01) if cup_m is not None else a[yh:]   # hard cut: no AA residue
    over(oc, oa, c[yh:], low_a, bot_edge)
    up_a_src = a[:yh + ext].copy()
    up_c_src = c[:yh + ext].copy()
    if ext:
        up_a_src[yh:] = cup_m[yh:yh + ext]
        off = cup_m[yh:yh + ext] < 0.5                       # non-cup pixels below the hinge: reuse the
        up_c_src[yh:][off] = np.broadcast_to(c[yh - 1], (ext, S, 3))[off]   # row above so resizing adds no dark seam
    up_h = max(1, int((yh + ext) * sq))
    up_c = cv2.resize(up_c_src, (S, up_h), interpolation=cv2.INTER_AREA)
    up_a = cv2.resize(up_a_src, (S, up_h), interpolation=cv2.INTER_AREA)
    over(oc, oa, up_c, up_a, top_edge - int(yh * sq))
    nose_out = (nose[0], top_edge - (yh - nose[1]) * sq)
    return np.clip(oc, 0, 255).astype(np.uint8), oa, nose_out


# ---------------------------------------------------------------- body

SHIRT = {"blue": (170, 100, 45), "navy": (95, 50, 25), "red": (45, 45, 170),
         "green": (60, 120, 50), "plaid-gray": (120, 115, 110)}


def tile(f, w, h):
    return np.tile(f, (math.ceil(h / f.shape[0]), math.ceil(w / f.shape[1])))[:h, :w]


def build_body(u, cfg, felt):
    """Felt torso in two layers so the neck sits INSIDE the collar:
       back  = inside of the shirt + collar band standing up behind the neck
       front = torso/sleeves with the neck opening cut out, collar points lying over it.
    Draw order: back -> neck -> front. u = size unit in px.
    Returns (front_bgr, front_a, back_bgr, back_a, collar_x, collar_y)."""
    u = float(u)
    W, H = int(3.2 * u), int(2.7 * u)
    ox = W // 2
    wf = cfg.get("shoulders", 0.80)       # shoulder width factor
    TY, TA, TW = 0.56, 0.40, 1.12 * wf    # torso shoulder ellipse: flatter = squarer, higher shoulders
    lw = max(1, int(u * 0.012))

    def Q(x, y):
        return int(ox + x * u), int(y * u)

    def felt_it(img, sh):
        mod = (1.0 + 0.13 * tile(felt["pile"], W, H) + 0.07 * tile(felt["fiber"], W, H)
               + 0.03 * tile(felt["mottle"], W, H)) * sh
        return cv2.GaussianBlur(np.clip(img * mod[..., None], 0, 255).astype(np.uint8), (0, 0), 0.55)

    def fuzzy_alpha(mk):
        soft = cv2.GaussianBlur(mk, (0, 0), 2.2).astype(np.float32) / 255.0
        return np.clip((soft - 0.5) * 2.4 + 0.5 + tile(felt["fiber"], W, H) * 0.5
                       + tile(felt["pile"], W, H) * 0.2, 0, 1)

    shirt = cfg["shirt"]

    # ---- back layer: shirt interior + stand-up collar band behind the neck
    cb = np.zeros((H, W, 3), np.uint8)
    mb = np.zeros((H, W), np.uint8)
    cv2.ellipse(cb, Q(0, 0.20), (int(0.30 * u), int(0.13 * u)), 0, 0, 360, shade(shirt, 0.42), -1, cv2.LINE_AA)
    cv2.ellipse(mb, Q(0, 0.20), (int(0.30 * u), int(0.13 * u)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    band = np.zeros((H, W), np.uint8)
    cv2.ellipse(band, Q(0, 0.20), (int(0.29 * u), int(0.15 * u)), 0, 180, 360, 255, int(0.075 * u), cv2.LINE_AA)
    cb[band > 0] = shade(shirt, 1.05)
    mb = cv2.max(mb, band)
    cb = felt_it(cb, velvet_shading(mb))
    ab = fuzzy_alpha(mb)

    # ---- front layer
    c = np.zeros((H, W, 3), np.uint8)
    torso = np.zeros((H, W), np.uint8)
    sleeves = np.zeros((H, W), np.uint8)
    for sx in (-1, 1):
        xs_ = [(0.80 + 0.42 * t) * wf for t in np.linspace(0, 1, 8)]
        top = [(sx * x, TY - TA * math.sqrt(max(0.0, 1 - (x / (TW + .04 * wf)) ** 2)) + 0.02) for x in xs_]
        poly = np.array([Q(*q) for q in top] + [Q(sx * 1.40 * wf, 2.8), Q(sx * 0.92 * wf, 2.8)], np.int32)
        cv2.fillPoly(sleeves, [poly], 255, cv2.LINE_AA)
    cv2.ellipse(torso, Q(0, TY), (int(TW * u), int(TA * u)), 0, 180, 360, 255, -1, cv2.LINE_AA)
    cv2.rectangle(torso, Q(-TW, TY), Q(TW, 2.8), 255, -1)
    sleeves = cv2.min(sleeves, cv2.bitwise_not(cv2.erode(torso, np.ones((3, 3), np.uint8))))
    c[sleeves > 0] = shade(shirt, 0.92)
    c[torso > 0] = shirt

    # neck opening + open placket V (cut out later), placket below it
    hole = np.zeros((H, W), np.uint8)
    cv2.ellipse(hole, Q(0, 0.20), (int(0.25 * u), int(0.10 * u)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.fillPoly(hole, [np.array([Q(-0.13, 0.22), Q(0.13, 0.22), Q(0, 0.44)], np.int32)], 255, cv2.LINE_AA)
    pl = np.array([Q(-0.075, 0.44), Q(0.075, 0.44), Q(0.075, 1.15), Q(-0.075, 1.15)], np.int32)
    cv2.fillPoly(c, [pl], shade(shirt, 1.06), cv2.LINE_AA)
    cv2.polylines(c, [pl], True, shade(shirt, 0.6), lw, cv2.LINE_AA)
    flaps = np.zeros((H, W), np.uint8)
    for sx in (-1, 1):
        flap = np.array([Q(sx * 0.31, 0.12), Q(sx * 0.02, 0.30), Q(sx * 0.03, 0.46), Q(sx * 0.26, 0.48)], np.int32)
        cv2.fillPoly(c, [flap], shade(shirt, 1.10), cv2.LINE_AA)
        cv2.fillPoly(flaps, [flap], 255, cv2.LINE_AA)
        cv2.polylines(c, [flap], True, shade(shirt, 0.55), lw, cv2.LINE_AA)
        cv2.line(c, Q(sx * 0.86 * wf, TY - 0.20), Q(sx * 0.98 * wf, 2.8), shade(shirt, 0.62), lw, cv2.LINE_AA)
    m = cv2.max(cv2.max(torso, sleeves), flaps)
    # torso: analytic barrel (cylinder across, rounded over the shoulders) - no distance-field ridges
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = np.clip((xx - ox) / (TW * u), -1, 1) * 0.88
    tops = np.argmax(torso > 0, axis=0).astype(np.float32)
    tops[~(torso > 0).any(0)] = H
    ny = -0.75 * np.clip(1 - (yy - tops[None, :]) / (0.35 * u), 0, 1)
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.02, 1))
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = np.clip((nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / nrm, 0, 1)
    barrel = 0.62 + 0.55 * lam + 0.30 * (1 - nz / nrm) ** 1.4
    sh = np.where(torso > 0, barrel, velvet_shading(sleeves, relief=2.6))
    # sleeves sit on top of the torso edge: occlusion shadow where they meet
    ao = cv2.GaussianBlur(sleeves, (0, 0), 0.05 * u).astype(np.float32) / 255.0
    sh = sh * np.where(torso > 0, 1 - 0.30 * ao, 1.0)
    c = felt_it(c, sh)

    br = max(3, int(0.038 * u))
    for y in (0.62, 0.85, 1.06):
        bx, by = Q(0, y)
        cv2.circle(c, (bx, by), br, (225, 228, 230), -1, cv2.LINE_AA)
        cv2.circle(c, (bx, by), br, (150, 150, 155), 1, cv2.LINE_AA)
        for d in (-1, 1):
            cv2.circle(c, (bx + d * br // 3, by), max(1, br // 5), (120, 120, 125), -1, cv2.LINE_AA)
        cv2.circle(c, (bx - br // 3, by - br // 3), max(1, br // 4), (255, 255, 255), -1, cv2.LINE_AA)

    a = fuzzy_alpha(m)
    cut = cv2.GaussianBlur(cv2.subtract(hole, flaps), (0, 0), 1.2).astype(np.float32) / 255.0
    a = a * (1 - cut)                                     # neck shows through the opening
    a[int(0.9 * H):] = np.maximum(a[int(0.9 * H):], m[int(0.9 * H):] / 255.0)
    return c, a, cb, ab, ox, int(0.20 * u)


def cast_head_shadow(frame, head_alpha, M, body_alpha, bx0, by0, size):
    """Soft shadow of the head (key light upper-left) falling on the neck and shirt only."""
    H, W = frame.shape[:2]
    Ms = M.copy()
    Ms[:, 2] += (0.05 * size, 0.09 * size)
    rx0, ry0, rx1, ry1, Mr = warp_roi(Ms, head_alpha.shape, frame.shape)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    warped = cv2.warpAffine(head_alpha, Mr, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderValue=0)
    h, w = body_alpha.shape
    fx0, fy0, fx1, fy1 = max(0, bx0), max(0, by0), min(W, bx0 + w), min(H, by0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    ys, xs = np.nonzero(warped[::4, ::4] > 0.01)
    if not ys.size:
        return
    # every buffer below is window-sized, not frame-sized
    y0, y1 = max(0, ry0 + ys.min() * 4 - 40), min(H, ry0 + ys.max() * 4 + 60)
    x0, x1 = max(0, rx0 + xs.min() * 4 - 40), min(W, rx0 + xs.max() * 4 + 60)
    sh = np.zeros((y1 - y0, x1 - x0), np.float32)
    sy0, sx0, sy1, sx1 = max(y0, ry0), max(x0, rx0), min(y1, ry1), min(x1, rx1)
    sh[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = warped[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0]
    recv = np.zeros_like(sh)                          # where the body is
    qy0, qx0, qy1, qx1 = max(y0, fy0), max(x0, fx0), min(y1, fy1), min(x1, fx1)
    if qy1 <= qy0 or qx1 <= qx0:
        return
    recv[qy0 - y0:qy1 - y0, qx0 - x0:qx1 - x0] = body_alpha[qy0 - by0:qy1 - by0, qx0 - bx0:qx1 - bx0]
    k = max(3.0, 0.03 * size)
    # wide soft blur: do it at 1/4 res and upsample (same look, ~16x fewer pixels)
    wq, hq = max(1, (x1 - x0) // 4), max(1, (y1 - y0) // 4)
    blur = cv2.GaussianBlur(cv2.resize(sh, (wq, hq), interpolation=cv2.INTER_AREA), (0, 0), k / 4)
    keep = 1.0 - 0.45 * cv2.resize(blur, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) * recv
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.blendLinear(roi, np.zeros_like(roi), keep, 1.0 - keep)


def blit(frame, spr, alpha, x0, y0):
    H, W = frame.shape[:2]
    h, w = spr.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = np.ascontiguousarray(alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0])
    frame[fy0:fy1, fx0:fx1] = cv2.blendLinear(s, frame[fy0:fy1, fx0:fx1], a, 1.0 - a)


def draw_neck(frame, skin_tex, fiber_tex, base, top, half_base, half_top):
    """Felt neck from the collar up to the puppet's head center (the head covers the top)."""
    (bx, by), (tx, ty) = base, top
    pts = np.array([(bx - half_base, by), (bx + half_base, by), (tx + half_top, ty), (tx - half_top, ty)], np.int32)
    x0, y0 = pts.min(0) - 6
    x1, y1 = pts.max(0) + 6
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    mk = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mk, [pts - (x0, y0)], 255, cv2.LINE_AA)
    sh = velvet_shading(mk, relief=2.6) * 0.92
    a = np.clip((cv2.GaussianBlur(mk, (0, 0), 1.6) / 255.0 - 0.5) * 2.4 + 0.5 + fiber_tex[y0:y1, x0:x1] * 0.4, 0, 1)
    col = skin_tex[y0:y1, x0:x1] * sh[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (col * a[..., None] + roi * (1 - a[..., None])).astype(np.uint8)


# ---------------------------------------------------------------- erase the real person

SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_segmenter/float16/latest/selfie_segmenter.tflite")


class Eraser:
    """Replaces every pixel the person segmenter calls 'you' with a saved photo of the empty room.
    No inpainting, no guessing: a real background, exposure-matched every frame, so nothing smears."""

    def __init__(self, w, h, path):
        self.w, self.h, self.path = w, h, path
        self.plate = None
        self.polarity = None          # does confidence mask 0 mean person (True) or background?
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None and img.shape[:2] == (h, w):
                self._set_plate(img)
                print(f"loaded empty-room plate {path}")
            else:
                print(f"ignoring {path}: camera size changed, press b to recapture")

    def _set_plate(self, img):
        self.plate = img.copy()                       # uint8; quarter-res copy for the gain match
        self.plate_q = cv2.resize(img, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)

    def grab(self, frame):
        self._set_plate(frame)
        cv2.imwrite(self.path, frame)
        print(f"empty-room plate saved -> {self.path}")

    def person_mask(self, conf, face_box):
        if self.polarity is None and face_box is not None:
            x0, y0, x1, y1 = (int(v) for v in face_box)
            roi = conf[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
            if roi.size:
                self.polarity = bool(roi.mean() > 0.5)
        return conf if self.polarity in (None, True) else 1.0 - conf

    def apply(self, frame, person):
        if self.plate is None:
            return frame
        # mask work at 1/4 resolution, then upsample only the bounding box
        q = cv2.resize(person, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        hard = (q > 0.35).astype(np.uint8)
        if not hard.any():
            return frame
        bg = q < 0.05
        if bg.sum() > 200:   # exposure/white-balance drift: match plate to the frame on clear background
            fq = cv2.resize(frame, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
            gain = np.clip(fq[bg].astype(np.float32).mean(0) / np.maximum(self.plate_q[bg].mean(0), 1), 0.7, 1.4)
        else:
            gain = np.ones(3, np.float32)
        hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        pmq = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.0)
        ys, xs = np.nonzero(hard)
        y0, y1 = max(0, ys.min() * 4 - 8), min(self.h, ys.max() * 4 + 12)
        x0, x1 = max(0, xs.min() * 4 - 8), min(self.w, xs.max() * 4 + 12)
        pm = cv2.resize(pmq, (self.w, self.h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        # uint8 end to end: saturating gain, then one weighted blend (no float32 frame copies)
        plate = cv2.multiply(self.plate[y0:y1, x0:x1], (float(gain[0]), float(gain[1]), float(gain[2]), 0.0))
        frame[y0:y1, x0:x1] = cv2.blendLinear(plate, frame[y0:y1, x0:x1], pm, 1.0 - pm)
        return frame


# ---------------------------------------------------------------- tracking

def measure(lm, bs, w, h, args):
    def p(i):
        return np.array((lm[i].x * w, lm[i].y * h), np.float32)

    def p3(i):  # MediaPipe z: same scale as x, smaller = closer to camera
        return np.array((lm[i].x * w, lm[i].y * h, lm[i].z * w), np.float32)

    def d(a, b):
        return float(np.linalg.norm(p(a) - p(b)))

    # 3D forehead-chin distance: doesn't shrink when you nod or look up, so the puppet's size
    # (and the neck/body built from it) stays constant through posture changes
    fh = float(np.linalg.norm(p3(FOREHEAD) - p3(CHIN)))
    face_mid = (p(FOREHEAD) + p(CHIN)) / 2
    ev = p(L_OUT) - p(R_OUT)
    roll = math.degrees(math.atan2(ev[1], ev[0]))
    v = p3(CHEEK_L) - p3(CHEEK_R)
    yaw = math.atan2(v[2], v[0])
    u = p3(CHIN) - p3(FOREHEAD)
    pitch = math.atan2(u[2], u[1])
    # Vertical anchor from the face; horizontal placement is done in composite() by lining the
    # puppet's nose up with your real nose tip (landmark 1), so it can't drift off on turns.
    center = np.array((face_mid[0], face_mid[1] - 0.25 * fh * math.sin(pitch)), np.float32)
    nose = p(NOSE_TIP)

    if bs:
        mouth = float(np.clip(bs.get("jawOpen", 0) * args.mouth_gain, 0, 1))
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        wide = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
        squint = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
        lid = args.lid * (1 - wide) + 0.35 * squint
        lid = 1.0 if blink > args.blink else lid + (1 - lid) * blink * 0.5
        outer = (bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0)) / 2
        brow_up = min(1.0, args.brow_gain * max(bs.get("browInnerUp", 0), outer))
        brow_dn = min(1.0, args.brow_gain * (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2)
        smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
        corners = float(np.clip(1.6 * (smile - frown), -1, 1))
    else:
        mouth = float(np.clip((d(LIP_UP, LIP_DN) / fh - 0.015) / 0.12, 0, 1))
        er = (d(R_UP, R_DN) / d(R_OUT, R_IN) + d(L_UP, L_DN) / d(L_OUT, L_IN)) / 2
        lid = 1.0 if er < 0.16 else args.lid
        brow_up = brow_dn = corners = 0.0

    def iris(o, i, ir):
        return (p(ir) - (p(o) + p(i)) / 2) / (d(o, i) * 0.25)

    g = np.clip((iris(R_OUT, R_IN, R_IRIS) + iris(L_OUT, L_IN, L_IRIS)) / 2, -1, 1)
    return (center[0], center[1], fh, roll, yaw, pitch, nose[0], nose[1]), (mouth, float(np.clip(lid, 0, 1)), g[0], g[1], brow_up, brow_dn, corners)


def head_matrix(cx, cy, fh, roll, k, lift=0.0, nose_canvas=None, nose_real=None):
    r = math.radians(roll)
    cx, cy = cx + math.sin(r) * lift * fh, cy - math.cos(r) * lift * fh
    ax, ay = ANCHOR[0] * S, ANCHOR[1] * S + PADT
    M = cv2.getRotationMatrix2D((ax, ay), -roll, fh * k / S)
    M[:, 2] += np.array((cx - ax, cy - ay), np.float64)
    if nose_canvas is not None:
        npx = M @ np.array((nose_canvas[0], nose_canvas[1], 1.0))
        axis = np.array((math.cos(r), math.sin(r)))
        M[:, 2] += axis * float(np.dot(np.asarray(nose_real) - npx, axis))
    return M


def warp_roi(M, src_shape, frame_shape, pad=2):
    """Frame-space bounding box of a warped canvas, and M re-based to that box."""
    sh, sw = src_shape[:2]
    H, W = frame_shape[:2]
    corners = M @ np.array([[0, 0, 1], [sw, 0, 1], [0, sh, 1], [sw, sh, 1]], np.float64).T
    x0, y0 = np.floor(corners.min(1)).astype(int) - pad
    x1, y1 = np.ceil(corners.max(1)).astype(int) + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    Mr = M.copy()
    Mr[:, 2] -= (x0, y0)
    return x0, y0, x1, y1, Mr


def composite(frame, canvas, alpha, M):
    """Warp + blend only inside the head's bounding box (not the whole frame)."""
    x0, y0, x1, y1, Mr = warp_roi(M, canvas.shape, frame.shape)
    if x1 <= x0 or y1 <= y0:
        return frame
    size = (x1 - x0, y1 - y0)
    pc = cv2.warpAffine(canvas, Mr, size, flags=cv2.INTER_LINEAR, borderValue=0)
    pa = cv2.warpAffine(alpha, Mr, size, flags=cv2.INTER_LINEAR, borderValue=0)
    frame[y0:y1, x0:x1] = cv2.blendLinear(pc, frame[y0:y1, x0:x1], pa, 1.0 - pa)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", help="read frames from a video file instead of the camera (benchmarking)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--skin", choices=SKIN, default="medium")
    ap.add_argument("--hair", choices=HAIR, default="salt")
    ap.add_argument("--beard", choices=HAIR, default="gray")
    ap.add_argument("--no-beard", action="store_true")
    ap.add_argument("--no-glasses", action="store_true")
    ap.add_argument("--lid", type=float, default=0.35, help="resting upper-lid coverage 0..0.9")
    ap.add_argument("--scale", type=float, default=2.4, help="puppet size vs face height")
    ap.add_argument("--shoulders", type=float, default=0.80, help="shoulder width (1.0 = old wide body)")
    ap.add_argument("--neck", type=float, default=0.30, help="collar distance below the head center")
    ap.add_argument("--body-size", type=float, default=1.25, help="body size relative to the head")
    ap.add_argument("--no-headphones", action="store_true")
    ap.add_argument("--no-erase", action="store_true", help="don't remove you from the video")
    ap.add_argument("--lift", type=float, default=0.15, help="raise puppet, fraction of face height")
    ap.add_argument("--pez", type=float, default=1.0, help="head squareness 0 (oval) .. 1 (Pez block)")
    ap.add_argument("--mouth-gain", type=float, default=2.2, help="how wide the mouth opens per jaw movement")
    ap.add_argument("--brow-gain", type=float, default=1.8, help="eyebrow expressiveness")
    ap.add_argument("--long", type=float, default=1.0, help="head elongation 0 (round) .. 1.2 (long)")
    ap.add_argument("--no-body", action="store_true", help="head only")
    ap.add_argument("--shirt", choices=SHIRT, default="blue")
    ap.add_argument("--body-follow", type=float, default=1.0,
                    help="1 = body rigidly attached under the head (no neck stretch); <1 adds lag")
    ap.add_argument("--turn", type=float, default=1.0, help="head turn/nod gain")
    ap.add_argument("--blink", type=float, default=0.45, help="eyeBlink blendshape threshold")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--profile", action="store_true", help="print per-stage timings every 150 frames")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--model", default=os.path.join(here, "face_landmarker.task"))
    ap.add_argument("--seg-model", default=os.path.join(here, "selfie_segmenter.tflite"))
    ap.add_argument("--plate", default=os.path.join(here, "empty_room.png"),
                    help="saved empty-room background used to erase you")

    args = ap.parse_args()
    args.lid = float(np.clip(args.lid, 0, 0.9))

    cfg = {"skin": SKIN[args.skin], "hair": HAIR[args.hair], "beard": HAIR[args.beard],
           "beard_on": not args.no_beard, "glasses": not args.no_glasses,
           "long": float(np.clip(args.long, 0, 1.2)), "shirt": SHIRT[args.shirt],
           "pez": float(np.clip(args.pez, 0, 1)),
           "shoulders": float(np.clip(args.shoulders, 0.5, 1.2)),
           "headphones": not args.no_headphones}

    backend = {"win32": cv2.CAP_DSHOW, "darwin": cv2.CAP_AVFOUNDATION}.get(sys.platform, cv2.CAP_ANY)
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.cam, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit("camera read failed")
    h, w = frame.shape[:2]

    if not os.path.exists(args.model):
        print(f"downloading model -> {args.model}")
        urllib.request.urlretrieve(MODEL_URL, args.model)
    landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    segmenter, eraser = None, None
    if not args.no_erase:
        if not os.path.exists(args.seg_model):
            print(f"downloading segmentation model -> {args.seg_model}")
            urllib.request.urlretrieve(SEG_URL, args.seg_model)
        segmenter = ImageSegmenter.create_from_options(ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=args.seg_model), running_mode=RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False))
        eraser = Eraser(w, h, args.plate)
        if eraser.plate is None:
            print("to erase yourself: press b, step out of frame within 5 s, wait for the capture")
    grab_at = None
    t0 = time.monotonic()

    print("building fleece texture...")
    felt = build_felt()
    skin_tex = (np.array(cfg["skin"], np.float32) *
                (1.0 + 0.13 * tile(felt["pile"], w, h) + 0.07 * tile(felt["fiber"], w, h))[..., None])
    fiber_tex = tile(felt["fiber"], w, h)
    body = None            # felt body sprite + cached rescale + current position
    head_key, head_cache = None, None
    cal_fh = []
    s_pose, s_expr = Smooth(0.5), Smooth(0.6)
    last, lost = None, 0
    calib, neutral = [], (0.0, 0.0)   # first ~1 s of frames = your neutral pose
    prof = Prof(args.profile)

    with pyvirtualcam.Camera(width=w, height=h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        print(f"virtual camera: {vcam.device}")
        print("look straight at the camera for 1 s to calibrate")
        print("preview keys: b capture empty room (5 s timer), w/s lift, =/- scale, i/k body up/down, "
              "c recalibrate, q quit")
        try:
            while True:
                prof.start()
                ok, frame = cap.read()
                if not ok:
                    break
                prof.lap("read")
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                ts = int((time.monotonic() - t0) * 1000)
                res = landmarker.detect_for_video(img, ts)
                prof.lap("landmark")
                if grab_at is not None and time.monotonic() >= grab_at:
                    eraser.grab(frame)
                    grab_at = None
                if segmenter is not None and eraser.plate is not None and grab_at is None:
                    sr = segmenter.segment_for_video(img, ts)
                    if sr.confidence_masks:
                        conf = sr.confidence_masks[-1].numpy_view()
                        box = None
                        if res.face_landmarks:
                            xs = [q.x * w for q in res.face_landmarks[0]]
                            ys = [q.y * h for q in res.face_landmarks[0]]
                            box = (min(xs), min(ys), max(xs), max(ys))
                        frame = eraser.apply(frame, eraser.person_mask(conf, box))
                    prof.lap("erase")

                if res.face_landmarks:
                    bs = ({b.category_name: b.score for b in res.face_blendshapes[0]}
                          if res.face_blendshapes else {})
                    pose, expr = measure(res.face_landmarks[0], bs, w, h, args)
                    if calib is not None:
                        calib.append(pose[4:6])
                        cal_fh.append((pose[2], pose[1]))
                        if len(calib) >= 30:
                            neutral = tuple(np.median(np.array(calib), axis=0))
                            calib = None
                            cal_fh = []
                            print(f"calibrated: yaw0 {math.degrees(neutral[0]):+.1f} deg, "
                                  f"pitch0 {math.degrees(neutral[1]):+.1f} deg")
                    last, lost = (s_pose(pose), s_expr(expr)), 0
                else:
                    lost += 1
                prof.lap("measure")
                if last is not None and lost < 15:
                    (cx, cy, fh, roll, yaw, pitch, nx, ny), (mo, lid, gx, gy, bu_, bd_, cn_) = last
                    st = {"mouth": float(mo), "lid": float(lid), "gaze": (float(gx), float(gy)),
                          "brow": (float(bu_), float(bd_)), "corners": float(cn_),
                          "yaw": float(np.clip((yaw - neutral[0]) * args.turn, -MAX_YAW, MAX_YAW)),
                          "pitch": float(np.clip((pitch - neutral[1]) * args.turn, -MAX_PITCH, MAX_PITCH))}
                    # re-render the head only when some value moved a full step (1 deg, 0.05 mouth/lid,
                    # 0.1 gaze/brow/corners) from what was last drawn. A deadband, not rounding: landmark
                    # jitter across a bucket edge re-rendered 70-80 % of frames with rounding
                    key = np.array((st["mouth"] * 20, st["lid"] * 20, st["gaze"][0] * 10, st["gaze"][1] * 10,
                                    math.degrees(st["yaw"]), math.degrees(st["pitch"]),
                                    st["brow"][0] * 10, st["brow"][1] * 10, st["corners"] * 10))
                    if head_key is None or np.max(np.abs(key - head_key)) >= 1.0:
                        prof.lap("head_key")
                        head_key, head_cache = key, draw_puppet(st, cfg, felt)
                        prof.lap("draw_puppet")
                    canvas, alpha, nose_c = head_cache
                    M = head_matrix(cx, cy, fh, roll, args.scale, args.lift, nose_c, (nx, ny))
                    head_c = M @ np.array((ANCHOR[0] * S, ANCHOR[1] * S + PADT, 1.0))

                    # body rides under the head: follows with a little lag (sitting up, leaning,
                    # leaning in) but the neck is clamped to +/-12% so it can never stretch
                    if not args.no_body:
                        unit = 0.55 * fh * args.scale * args.body_size
                        if body is None or body["k"] != args.scale:
                            src = build_body(unit, cfg, felt)
                            body = {"k": args.scale, "unit0": unit, "src": src,
                                    "q": None, "pos": None}
                        q = round(unit / body["unit0"] / 0.02) * 0.02      # quantized rescale, cached
                        if body["q"] != q:
                            fc0, fa0, bc0, ba0, ox0, oy0 = body["src"]
                            sz = (max(1, int(fc0.shape[1] * q)), max(1, int(fc0.shape[0] * q)))
                            rs = lambda im: cv2.resize(im, sz, interpolation=cv2.INTER_AREA)
                            body["q"], body["spr"] = q, (rs(fc0), rs(fa0), rs(bc0), rs(ba0), int(ox0 * q), int(oy0 * q))
                        fc, fa, bc, ba, ox, oy = body["spr"]
                        neck = args.neck * fh * args.scale
                        want = np.array((head_c[0], head_c[1] + neck))
                        pos = want if body["pos"] is None else body["pos"] + args.body_follow * (want - body["pos"])
                        dy = float(np.clip(pos[1] - head_c[1], 0.97 * neck, 1.03 * neck))
                        dx = float(np.clip(pos[0] - head_c[0], -0.10 * neck, 0.10 * neck))
                        pos = np.array((head_c[0] + dx, head_c[1] + dy))
                        body["pos"] = pos
                        bu = unit
                        bx0, by0 = int(pos[0] - ox), int(pos[1] - oy)
                        blit(frame, bc, ba, bx0, by0)                       # collar back + shirt inside
                        draw_neck(frame, skin_tex, fiber_tex, (int(pos[0]), int(pos[1] + 0.26 * bu)),
                                  (int(head_c[0]), int(head_c[1])), int(0.17 * bu), int(0.15 * bu))
                        blit(frame, fc, fa, bx0, by0)                       # shirt front over the neck
                        cast_head_shadow(frame, alpha, M, fa, bx0, by0, fh * args.scale)
                    prof.lap("body")
                    frame = composite(frame, canvas, alpha, M)
                    prof.lap("composite")
                vcam.send(frame)
                prof.lap("send")
                if args.preview:
                    cv2.imshow("puppetcam", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("b") and eraser is not None:
                        grab_at = time.monotonic() + 5.0
                        print("capturing empty room in 5 s - step out of frame")
                    if key in (ord("i"), ord("k")):
                        args.neck += -.01 if key == ord("i") else .01
                        print(f"--neck {args.neck:.2f}")
                    if key == ord("c"):
                        calib, cal_fh = [], []
                        print("recalibrating - look straight at the camera")
                    if key in (ord("w"), ord("s"), ord("="), ord("-")):
                        args.lift += {ord("w"): .01, ord("s"): -.01}.get(key, 0)
                        args.scale += {ord("="): .05, ord("-"): -.05}.get(key, 0)
                        print(f"--lift {args.lift:.2f} --scale {args.scale:.2f}")
                    prof.lap("preview")
                vcam.sleep_until_next_frame()
                prof.lap("sleep")
                prof.end()
        except KeyboardInterrupt:
            pass
        prof.report()
    landmarker.close()
    if segmenter is not None:
        segmenter.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
