#!/usr/bin/env python3
"""Offline render checks for puppetcam (no camera, no virtual camera).

Run from the repo root:  python tests/render_check.py [--out /tmp/puppetcam_check]
Exit code 0 = all checks pass. Writes contact sheets to --out for eyeballing.

Checks (each one exists because the bug it guards against actually shipped once):
  holes     - head interior must be fully opaque (fuzz noise once punched see-through holes)
  strays    - no single-row alpha spikes (headphone cup slivers / skin slab at the mouth hinge)
  smooth    - 1-degree yaw steps change the image evenly (cups once popped in/out at a threshold)
  nose      - the CPU-projected nose point lands on the rendered nose (it pins the puppet to your face)
  felt      - no blown-out highlights on fabric: fleece is matte (the "plastic" failure)
  body      - torso interior is fully opaque
  timing    - GPU head/body render cost (report only)
"""
import argparse
import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import pyvirtualcam  # noqa: F401
except ImportError:  # harness never opens a camera; allow running without the native lib
    sys.modules["pyvirtualcam"] = types.ModuleType("pyvirtualcam")

import cv2
import numpy as np

import puppetcam as p
from puppet_gpu import GpuPuppet

CFG = {"skin": p.SKIN["medium"], "hair": p.HAIR["salt"], "beard": p.HAIR["gray"], "beard_on": True,
       "glasses": True, "long": 1.0, "shirt": p.SHIRT["blue"], "pez": 1.0, "shoulders": 0.8,
       "headphones": True}


def state(mouth=0.0, yaw=0.0, pitch=0.0, brow=(0.0, 0.0), corners=0.0, lid=0.35):
    return {"mouth": mouth, "lid": lid, "gaze": (0.0, 0.0), "yaw": math.radians(yaw),
            "pitch": math.radians(pitch), "brow": brow, "corners": corners}


def on_red(c, a):
    bg = np.zeros(c.shape, np.float32)
    bg[:] = (0, 0, 255)
    return (c * a[..., None] + bg * (1 - a[..., None])).astype(np.uint8)


def renderer(cfg):
    return GpuPuppet(cfg, p.S, p.PADT, p.OUT_H, p.MOUTH_OPEN)


def check_holes(r):
    worst = 0
    for mo in (0, 0.5, 1.0):
        for yaw in (-30, 0, 30):
            # headphones and glasses off here: the band arcs over the crown and the far lens frame
            # sticks past the cheek when turned; both enclose real background, which would read
            # as a false "hole" once the outline is filled
            c, a, _ = r["bare"].head(state(mo, yaw))
            solid = (a > 0.5).astype(np.uint8)
            # fill internal gaps (open mouth) so we only test the head's interior
            cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(solid)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            core = cv2.erode(filled, np.ones((21, 21), np.uint8)) > 0
            holes = int(((a < 0.95) & core).sum())
            worst = max(worst, holes)
    return worst == 0, f"max see-through pixels inside head: {worst}"


def check_strays(r):
    bad = []
    for mo in (0, 0.3, 0.6, 0.9):
        for pitch in (-20, -10, 0, 10, 20):
            for yaw in (-20, 0, 20):
                _, a, _ = r["g"].head(state(mo, yaw, pitch))
                w = (a > 0.05).sum(1).astype(int)
                spikes = [r for r in range(1, len(w) - 1) if w[r] - max(w[r - 1], w[r + 1]) > 12]
                if spikes:
                    bad.append((mo, pitch, yaw, spikes[:3]))
    return not bad, f"poses with stray rows: {len(bad)}/60 {bad[:3]}"


def check_smooth(r):
    prev, steps = None, []
    for yaw in range(-30, 31):
        c, a, _ = r["g"].head(state(yaw=yaw))
        img = on_red(c, a).astype(np.float32)
        if prev is not None:
            steps.append(float(np.abs(img - prev).mean()))
        prev = img
    ratio = max(steps) / float(np.median(steps))
    return ratio < 1.6, f"max/median per-degree change: {ratio:.2f} (max {max(steps):.2f})"


def check_nose(r):
    """The nose point drives horizontal placement; if the CPU projection drifts off the
    rendered nose, the puppet slides sideways against your face."""
    bad = []
    for mo in (0.0, 0.8):
        for yaw, pitch in ((0, 0), (-30, 0), (30, 0), (0, -20), (0, 20), (25, 15)):
            c, a, n = r["g"].head(state(mo, yaw, pitch))
            x, y = int(round(n[0])), int(round(n[1]))
            # nose tip: opaque and skin colored (not glasses/mustache/background)
            px = c[y, x].astype(int)
            skin = np.array(CFG["skin"], int)
            hue_ok = np.argmax(px) == np.argmax(skin) and np.argmin(px) == np.argmin(skin)
            if a[y, x] < 0.99 or not hue_ok:
                bad.append((mo, yaw, pitch, tuple(px), round(float(a[y, x]), 2)))
    return not bad, f"nose points off the nose: {len(bad)}/12 {bad[:3]}"


def check_felt(r):
    """Fleece is matte: lit skin felt must never clip. (Overexposed key + fill once clipped the
    red channel on 21k face pixels: a hot, plastic look.) Eye whites may be bright; they're not felt."""
    worst = 0
    for yaw, pitch in ((0, 0), (-30, 0), (30, 10), (0, -20)):
        c, a, _ = r["bare"].head(state(0.0, yaw, pitch))
        b, g, rr = (c[..., i].astype(int) for i in range(3))
        skin = (a > 0.99) & (rr > g) & (g > b) & (rr - b > 60)        # saturated warm = skin felt
        worst = max(worst, int((skin & (rr >= 254)).sum()))
    return worst == 0, f"clipped skin pixels: {worst}"


def check_body(r):
    worst = 0
    for yaw in (-0.25, 0.0, 0.25):
        c, a, ox, oy = r["g"].body(yaw, 1.25)
        solid = (a > 0.5).astype(np.uint8)
        core = cv2.erode(solid, np.ones((21, 21), np.uint8)) > 0
        worst = max(worst, int(((a < 0.95) & core).sum()))
    return worst == 0, f"max see-through pixels inside torso: {worst}"


def timing(r):
    st = state(0.5, 15, 5, (0.3, 0.0), 0.2)
    r["g"].head(st)  # warm-up
    t = time.perf_counter()
    for _ in range(10):
        r["g"].head(st)
    head = (time.perf_counter() - t) * 100
    t = time.perf_counter()
    r["g"].body(0.1, 1.25)
    body = (time.perf_counter() - t) * 1000
    return True, f"GPU head: {head:.1f} ms, body: {body:.1f} ms"


def sheets(r, out):
    os.makedirs(out, exist_ok=True)
    poses = [state(), state(0.9, 0, 0, (1, 0)), state(0.3, 0, 0, (0, 1), -1), state(0.2, -25, 0, (0, 0), 1),
             state(0.9, 0, -18), state(0.0, 30, 10)]
    tiles = []
    for st in poses:
        c, a, _ = r["g"].head(st)
        tiles.append(cv2.resize(on_red(c, a), (300, int(300 * a.shape[0] / a.shape[1]))))
    cv2.imwrite(os.path.join(out, "poses.png"), np.hstack(tiles))
    bodies = []
    for yaw in (-0.25, 0.0, 0.25):
        c, a, _, _ = r["g"].body(yaw, 1.25)
        bodies.append(cv2.resize(on_red(c, a), (400, int(400 * a.shape[0] / a.shape[1]))))
    cv2.imwrite(os.path.join(out, "body.png"), np.hstack(bodies))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/puppetcam_check")
    args = ap.parse_args()
    r = {"g": renderer(CFG),
         "bare": renderer(dict(CFG, headphones=False, glasses=False))}
    ok_all = True
    for name, fn in (("holes", check_holes), ("strays", check_strays), ("smooth", check_smooth),
                     ("nose", check_nose), ("felt", check_felt), ("body", check_body), ("timing", timing)):
        ok, msg = fn(r)
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:7s} {msg}")
    sheets(r, args.out)
    print(f"contact sheet: {os.path.join(args.out, 'poses.png')}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
