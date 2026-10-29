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
  erase     - after the camera moves, the plate re-aligns so the erased person leaves no ghost
              (a tilted monitor webcam once pasted a shifted room into the silhouette)
  style     - auto-style facial-hair and color decisions; command-line switches override them
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
import puppet_style
from puppet_gpu import GpuPuppet

CFG = {"skin": p.SKIN["medium"], "hair": p.HAIR["salt"], "beard": p.HAIR["gray"], "beard_on": True,
       "glasses": True, "long": 1.0, "shirt": p.SHIRT["blue"], "pez": 1.0, "shoulders": 0.8,
       "headphones": True, "mustache": False, "full_beard": False, "brow": p.HAIR["gray"]}


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


def check_erase(r):
    """Synthetic room; the camera then tilts/shifts (known homography) and a person stands in front.
    Erasing with the stale plate pastes the wrong patch of room; the aligned plate must match."""
    rng = np.random.default_rng(3)
    w, h = 1280, 720
    room = cv2.GaussianBlur(rng.integers(60, 200, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    for _ in range(140):                                   # furniture-like edges for the matcher
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        cv2.rectangle(room, (x, y), (x + int(rng.integers(20, 160)), y + int(rng.integers(20, 120))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    moved = np.array([[0.995, -0.03, 24.0], [0.02, 0.99, 70.0], [0.0, 0.0, 1.0]])
    truth = cv2.warpPerspective(room, moved, (w, h), borderMode=cv2.BORDER_REPLICATE)
    person = np.zeros((h, w), np.float32)
    cv2.ellipse(person, (640, 520), (230, 330), 0, 0, 360, 1.0, -1)
    frame = truth.copy()
    frame[person > 0.5] = (40, 50, 70)
    inside = cv2.erode((person > 0.5).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0

    e = p.Eraser(w, h, "/nonexistent/plate.png")
    e._set_plate(room)
    stale = np.median(cv2.absdiff(e.apply(frame.copy(), person), truth).max(-1)[inside])
    fit = e.align(frame, person)
    if fit is None:
        return False, f"align found no fit (stale residual {stale:.0f})"
    e.use_aligned(fit)
    fixed = np.median(cv2.absdiff(e.apply(frame.copy(), person), truth).max(-1)[inside])
    # no churn: once aligned, a static camera must not keep swapping in new (noisier) fits,
    # and a matching room must never be flagged as a mismatch
    churn = False
    for _ in range(3):
        churn |= e.update(e.align(frame, person)) is not None
    churn |= e.plate is None
    return fixed < 6 and not churn, (f"residual inside the silhouette: stale plate {stale:.0f}, aligned {fixed:.0f} "
                                     f"(limit 6); re-fit or drop on a static view: {churn}")


def check_plate_mismatch(r):
    """A plate from another desk can't be aligned; pasting it would put another room inside your
    silhouette. It must be dropped (erase off until b) within two checks, not used."""
    rng = np.random.default_rng(5)
    w, h = 1280, 720

    def room(seed):
        g = np.random.default_rng(seed)
        img = cv2.GaussianBlur(g.integers(60, 200, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
        for _ in range(140):
            x, y = int(g.integers(0, w - 60)), int(g.integers(0, h - 60))
            cv2.rectangle(img, (x, y), (x + int(g.integers(20, 160)), y + int(g.integers(20, 120))),
                          tuple(int(v) for v in g.integers(0, 255, 3)), -1)
        return img

    person = np.zeros((h, w), np.float32)
    cv2.ellipse(person, (640, 520), (230, 330), 0, 0, 360, 1.0, -1)
    frame = room(11)
    frame[person > 0.5] = tuple(int(v) for v in rng.integers(30, 90, 3))
    e = p.Eraser(w, h, "/nonexistent/plate.png")
    e._set_plate(room(3))                                  # yesterday's desk
    statuses = [e.update(e.align(frame, person)) for _ in range(2)]
    dropped = e.plate is None and statuses[-1] == "mismatch"
    return dropped, f"plate from another room dropped: {dropped} (statuses {statuses}, err {e.last_err})"


def check_style(r):
    """Auto-style decisions from measurements (thresholds from the owner's frames): shaved upper lip
    0.02 vs mustache 0.33+; goatee 0.39+; jaw without beard 0.12-0.21. Colors are relative to skin.
    Explicit switches must override the detected style."""
    import argparse

    def s(mu, go, jaw, hair, beard, brow, skin=(120.0, 20.0)):
        return {"mustache": mu, "goatee": go, "jaw": jaw, "skin": list(skin),
                "hair": list(hair), "beard": list(beard) if beard else None, "brow": list(brow) if brow else None}

    bad = []
    owner = puppet_style.decide([s(0.02, 0.48, 0.16, (91, 9), (87, 4.5), (57, 8.5), (121, 21))] * 3)
    if not (owner["goatee"] and not owner["mustache"] and not owner["full_beard"]
            and (owner["hair"], owner["beard"], owner["brow"]) == ("gray", "gray", "salt")):
        bad.append(("owner today", puppet_style.summary(owner)))
    full = puppet_style.decide([s(0.6, 0.9, 0.7, (30, 4), (28, 4), (25, 4))])
    if not (full["mustache"] and full["full_beard"] and full["beard"] == "black"):
        bad.append(("full dark beard", puppet_style.summary(full)))
    shaved = puppet_style.decide([s(0.03, 0.1, 0.1, (60, 18), None, (50, 16))])
    if shaved["goatee"] or shaved["mustache"] or shaved["full_beard"] or shaved["hair"] != "brown":
        bad.append(("clean-shaven brunette", puppet_style.summary(shaved)))
    if puppet_style.decide([]) is not None:
        bad.append(("no frames", "should be None"))
    flags = argparse.Namespace(hair="white", beard=None, brows=None, no_beard=True, full_beard=False,
                               mustache=True, no_mustache=False)
    lk = p.look(flags, owner)
    if lk["hair"] != p.HAIR["white"] or lk["beard_on"] or not lk["mustache"] or lk["beard"] != p.HAIR["gray"]:
        bad.append(("flags override", lk))
    return not bad, f"style decisions wrong: {len(bad)} {bad[:2]}"


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
                     ("nose", check_nose), ("felt", check_felt), ("body", check_body), ("erase", check_erase),
                     ("plate", check_plate_mismatch), ("style", check_style),
                     ("timing", timing)):
        ok, msg = fn(r)
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:7s} {msg}")
    sheets(r, args.out)
    print(f"contact sheet: {os.path.join(args.out, 'poses.png')}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
