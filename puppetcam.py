#!/usr/bin/env python3
"""puppetcam: real-time felt-puppet face filter -> virtual camera (Teams/Zoom/etc).

Block felt head, natural skin tone, glasses, goatee, heavy-lidded ping-pong eyes, headphones, shirt.
The puppet is a real 3-D model: raymarched signed-distance shapes on the GPU (puppet_gpu.py +
puppet.wgsl), shaded as fleece (matte wrap light, grazing sheen, fiber noise, AO, soft shadows,
fuzzy silhouette). This file does capture, tracking, erasing and compositing.
"""
import argparse
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions, ImageSegmenter,
                                           ImageSegmenterOptions, RunningMode)

import puppet_style
from puppet_gpu import BODY_UNIT, HAIRLINES, GpuPuppet

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

S = 512                 # local puppet canvas (square); on-screen head is usually smaller
ANCHOR = (0.50, 0.40)   # canvas point pinned to the real face center
PADT, PADB = int(0.19 * S), int(0.08 * S)   # headroom for the Pez head lift / jaw drop
OUT_H = S + PADT + PADB
MAX_YAW, MAX_PITCH = math.radians(38), math.radians(25)
MOUTH_OPEN = 0.06        # below this the mouth is a closed line, above it the head hinges
BODY_TURN = 0.35        # torso yaw as a fraction of head yaw
BODY_ROLL = 0.5         # torso tilt as a fraction of head roll (about the base of the neck)
PLATE_MISMATCH = 35.0   # plate match error (levels) above which, with no re-fit, it's another room
SEG_WAIT = 0.010       # s: longest the frame waits for its own person mask before using the last one

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


# ---------------------------------------------------------------- body

SHIRT = {"blue": (170, 100, 45), "navy": (95, 50, 25), "red": (45, 45, 170),
         "green": (60, 120, 50), "plaid-gray": (120, 115, 110)}


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


def roll_sprite(c, a, ox, oy, deg):
    """Tilt the torso sprite by `deg` (same sense as head_matrix's roll) about its collar point (ox, oy),
    on a canvas big enough to hold it. Returns (color, alpha, new collar x, new collar y)."""
    if abs(deg) < 1e-3:
        return c, a, ox, oy
    h, w = a.shape
    M = cv2.getRotationMatrix2D((float(ox), float(oy)), -deg, 1.0)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    M[:, 2] -= (x0, y0)
    size = (int(x1 - x0), int(y1 - y0))
    rc = cv2.warpAffine(c, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
    ra = cv2.warpAffine(a, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
    return rc, ra, int(round(ox - x0)), int(round(oy - y0))


def blit(frame, spr, alpha, x0, y0):
    H, W = frame.shape[:2]
    h, w = spr.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = np.ascontiguousarray(alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0])
    frame[fy0:fy1, fx0:fx1] = cv2.blendLinear(s, frame[fy0:fy1, fx0:fx1], a, 1.0 - a)


# ---------------------------------------------------------------- erase the real person

SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")


class PersonSegmenter:
    """Person confidence mask from MediaPipe's multiclass selfie model, on the GPU (Metal), on its own
    thread so it overlaps face tracking (13.6 ms vs 124 ms on CPU).
    Why multiclass: the single-class selfie model dropped the torso whenever the head turned (16 % torso
    coverage vs 89 %), leaving the real shirt un-erased around the puppet.
    Person = the strongest person class, not 1 - background: 1 - background floors at ~0.08 on an empty
    room, so nothing would pass the < 0.05 clear-background test that gain matching and plate alignment
    depend on. Single-class models (one mask) are used as-is."""

    def __init__(self, path):
        self.pool = ThreadPoolExecutor(1)               # MediaPipe's GPU context stays on this thread
        self.seg = self.pool.submit(self._create, path).result()

    @staticmethod
    def _create(path):
        return ImageSegmenter.create_from_options(ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=path, delegate=BaseOptions.Delegate.GPU),
            running_mode=RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))

    def submit(self, frame, ts, classes=False):
        """Start segmenting `frame` (it must not be modified until the result is taken). The result is
        (person mask, argmax class map if `classes` else None, ts) or None."""
        return self.pool.submit(self._run, frame, ts, classes)

    def _run(self, frame, ts, classes):
        img = mp.Image(image_format=mp.ImageFormat.SRGBA,      # the GPU path takes 4-channel input only
                       data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
        masks = self.seg.segment_for_video(img, ts).confidence_masks
        if not masks:
            return None
        views = [np.squeeze(m.numpy_view(), axis=-1) if m.numpy_view().ndim == 3 else m.numpy_view()
                 for m in masks]
        if len(views) == 1:
            return views[0].copy(), None, ts
        if classes:
            stack = np.stack(views)
            return np.max(stack[1:], axis=0), np.argmax(stack, axis=0).astype(np.uint8), ts
        return np.max(np.stack(views[1:]), axis=0), None, ts

    def close(self):
        self.pool.submit(self.seg.close).result()
        self.pool.shutdown()


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
        # plate0 = as captured; every alignment warps plate0 (never the warped plate), so fits don't compound
        self.plate0, self.H = img.copy(), np.eye(3)
        self.last_err, self.mismatches = None, 0
        self._use(self.plate0)

    def _use(self, img):
        self.plate = img                              # uint8; quarter-res copy for the gain match
        self.plate_q = cv2.resize(img, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)

    def align(self, frame, person):
        """Fit the captured plate to the live view (camera bumped or tilted since the capture): ORB features
        on background pixels only, at half res, RANSAC homography. Reads only plate0/H, so it is safe on a
        worker thread. Returns (warped_plate, H, plate0), or None when there's no trustworthy fit or it
        doesn't beat the current plate by 20 % and 3 levels on clear background (camera hasn't moved).
        Also records last_err, the current plate's error, which update() uses to spot a wrong room."""
        plate0, plate, s = self.plate0, self.plate, 0.5
        if plate is None:
            return None
        # match error on clear background, exposure-compensated like apply(): ~15 for this room's plate,
        # 54-100 for another room's
        qs = (self.w // 4, self.h // 4)
        fq = cv2.resize(frame, qs, interpolation=cv2.INTER_AREA).astype(np.float32)
        bg = cv2.resize(person, qs, interpolation=cv2.INTER_AREA) < 0.05
        if bg.sum() < 500:
            self.last_err = None
            return None

        def err(img):
            pq = cv2.resize(img, qs, interpolation=cv2.INTER_AREA).astype(np.float32)
            gain = np.clip(fq[bg].mean(0) / np.maximum(pq[bg].mean(0), 1), 0.7, 1.4)
            return float(np.median(np.abs(fq - pq * gain).max(-1)[bg]))

        cur = self.last_err = err(plate)
        g0 = cv2.cvtColor(cv2.resize(plate0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        clear = cv2.resize(person, (g1.shape[1], g1.shape[0]), interpolation=cv2.INTER_AREA) < 0.05
        clear = cv2.erode(clear.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))   # off the person's edge
        orb = cv2.ORB_create(1500)
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, clear)
        if d0 is None or d1 is None:
            return None
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
        if len(m) < 40:
            return None
        src = np.float32([k0[x.queryIdx].pt for x in m]) / s
        dst = np.float32([k1[x.trainIdx].pt for x in m]) / s
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inl.sum() < 40 or inl.mean() < 0.4:
            return None
        det = float(np.linalg.det(H[:2, :2]))
        if not 0.6 < det < 1.6:                       # a camera tilt/shift, not a wild fit
            return None
        warped = cv2.warpPerspective(plate0, H, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        # adopt only a clearly better match on clear background. Fits on a static camera scatter 1-7 px;
        # swapping each one in would make the erased patch jiggle once a second
        new = err(warped)
        if new > min(0.8 * cur, cur - 3.0):             # at least 20 % and 3 levels better
            return None
        self.last_err = new
        return warped, H, plate0

    def update(self, fit):
        """Take an align() result on the main thread. Returns "aligned" when a better fit was adopted,
        "mismatch" when the plate was dropped (no fit and still far off on two checks running: it's another
        room, e.g. a different desk; erase stays off until b), else None."""
        if fit is not None:
            return "aligned" if self.use_aligned(fit) else None
        if self.last_err is not None and self.last_err > PLATE_MISMATCH:
            self.mismatches += 1
            if self.mismatches >= 2:
                self.plate = None
                return "mismatch"
        else:
            self.mismatches = 0
        return None

    def use_aligned(self, fit):
        """Adopt a fit from align(); refused if the plate was recaptured (b) while it was computing."""
        warped, H, src = fit
        if src is not self.plate0:
            return False
        self.H = H
        self._use(warped)
        return True

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


# ---------------------------------------------------------------- background workers

class Grabber(threading.Thread):
    """Reads the camera continuously and keeps only the newest frame, so the main loop never
    blocks on the device and never processes a stale, buffered frame."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.cond = cap, threading.Condition()
        self.ok, self.frame, self.t, self.seq, self.stopped = True, None, 0.0, 0, False

    def run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok, self.frame, self.t, self.seq = ok, frame, time.monotonic(), self.seq + 1
                self.cond.notify_all()
            if not ok:
                return

    def next(self, seq):
        """Block until a frame newer than `seq` exists; returns (seq, ok, frame, capture_time)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq)
            return self.seq, self.ok, self.frame, self.t

    def stop(self):
        self.stopped = True
        self.join(timeout=2.0)


def look(args, style):
    """Hair and facial hair: built-in defaults <- detected/saved style <- explicit command-line switches."""
    hair, beard, brow = "salt", "gray", None
    goatee, mustache, full = True, False, False
    if style is not None:
        hair, beard, brow = style["hair"], style["beard"], style["brow"]
        goatee, mustache, full = style["goatee"], style["mustache"], style["full_beard"]
    if args.hair:
        hair = args.hair
    if args.beard:
        beard = args.beard
    if args.brows:
        brow = args.brows
    if args.no_beard:
        goatee, full = False, False
    if args.full_beard:
        full = True
    if args.mustache:
        mustache = True
    if args.no_mustache:
        mustache = False
    return {"hair": HAIR[hair], "beard": HAIR[beard], "brow": HAIR[brow or beard],
            "beard_on": goatee, "mustache": mustache, "full_beard": full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", help="read frames from a video file instead of the camera (benchmarking)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--skin", choices=SKIN, default="medium")
    # hair and facial hair come from auto-style (puppet_style.json); these switches override it
    ap.add_argument("--hair", choices=HAIR)
    ap.add_argument("--beard", choices=HAIR)
    ap.add_argument("--brows", choices=HAIR)
    ap.add_argument("--hairline", choices=HAIRLINES, default="widow",
                    help="widow's peak, straight, receding (bald front crown), bald (side/back fringe)")
    ap.add_argument("--forehead", type=float, default=0.0,
                    help="0..1: move the front hairline up and back (bigger forehead); ~0.35-0.5 keeps the peak visible")
    ap.add_argument("--no-beard", action="store_true", help="no goatee or full beard")
    ap.add_argument("--full-beard", action="store_true")
    ap.add_argument("--mustache", action="store_true")
    ap.add_argument("--no-mustache", action="store_true")
    ap.add_argument("--no-glasses", action="store_true")
    ap.add_argument("--lid", type=float, default=0.35, help="resting upper-lid coverage 0..0.9")
    ap.add_argument("--scale", type=float, default=2.4, help="puppet size vs face height")
    ap.add_argument("--shoulders", type=float, default=0.80, help="shoulder width (1.0 = old wide body)")
    ap.add_argument("--neck", type=float, default=0.45, help="collar distance below the head center")
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
    ap.add_argument("--seg-model", default=os.path.join(here, "selfie_multiclass_256x256.tflite"))
    ap.add_argument("--plate", default=os.path.join(here, "empty_room.png"),
                    help="saved empty-room background used to erase you")
    ap.add_argument("--style", default=os.path.join(here, "puppet_style.json"),
                    help="detected facial hair + hair colors; written by auto-style, read on later starts")
    ap.add_argument("--restyle", action="store_true", help="re-detect the style at calibration (or press c)")
    ap.add_argument("--no-auto-style", action="store_true", help="ignore/skip style detection")

    args = ap.parse_args()
    args.lid = float(np.clip(args.lid, 0, 0.9))

    style = None
    if not args.no_auto_style and not args.restyle:
        style = puppet_style.load(args.style)
        if style is not None:
            print(f"style ({args.style}): {puppet_style.summary(style)}")
    cfg = {"skin": SKIN[args.skin], "glasses": not args.no_glasses,
           "long": float(np.clip(args.long, 0, 1.2)), "shirt": SHIRT[args.shirt],
           "pez": float(np.clip(args.pez, 0, 1)),
           "shoulders": float(np.clip(args.shoulders, 0.5, 1.2)),
           "headphones": not args.no_headphones, "hairline": args.hairline,
           "forehead": float(np.clip(args.forehead, 0, 1))}
    cfg.update(look(args, style))

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
        segmenter = PersonSegmenter(args.seg_model)
        eraser = Eraser(w, h, args.plate)
        if eraser.plate is None:
            print("to erase yourself: press b, step out of frame within 5 s, wait for the capture")
    grab_at = None
    t0 = time.monotonic()

    body = None            # felt body sprite + cached rescale + current position
    head_key, head_cache = None, None
    cal_fh = []
    s_pose, s_expr = Smooth(0.5), Smooth(0.6)
    last, lost = None, 0
    calib, neutral = [], (0.0, 0.0)   # first ~1 s of frames = your neutral pose
    prof = Prof(args.profile)
    # camera frames come from a reader thread (newest frame wins); a --video file is read in-line
    # so benchmarks process every frame
    grabber, seq = None, 0
    if not args.video:
        grabber = Grabber(cap)
        grabber.start()
    # the GPU puppet lives on one worker thread: created there, and every head/body render runs there
    renderer, pending, body_job = ThreadPoolExecutor(1), None, None
    gpu = renderer.submit(GpuPuppet, cfg, S, PADT, OUT_H, MOUTH_OPEN).result()
    # the camera can be bumped or tilted after the plate is captured; re-fit the plate about once a
    # second on its own worker (~90 ms) or the erased silhouette shows a shifted room (a "ghost")
    aligner, align_job, frame_i = ThreadPoolExecutor(1), None, 0
    seg_job, seg_last = None, None
    # auto-style samples near-frontal frames during calibration (needs the segmenter's class map)
    can_style = not args.no_auto_style and segmenter is not None
    styling, style_samples = can_style and style is None, []
    if not args.no_auto_style and segmenter is None:
        print("auto-style needs the segmenter (it is off with --no-erase)")

    with pyvirtualcam.Camera(width=w, height=h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        print(f"virtual camera: {vcam.device}")
        print("look straight at the camera for 1 s to calibrate")
        print("preview keys: b capture empty room (5 s timer), w/s lift, =/- scale, i/k body up/down, "
              "c recalibrate, q quit")
        try:
            while True:
                prof.start()
                frame_i += 1
                if grabber is not None:
                    seq, ok, frame, t_cap = grabber.next(seq)
                else:
                    ok, frame = cap.read()
                    t_cap = time.monotonic()
                if not ok:
                    break
                prof.lap("read")
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                ts = int((t_cap - t0) * 1000)
                # person mask: segment this frame while we track the face, but never stall on it. It
                # shares the GPU with the puppet renders and sometimes runs long; then this frame uses the
                # previous mask (the 28 px dilation covers a frame of motion) and the job finishes later
                erasing = segmenter is not None and eraser.plate is not None and grab_at is None
                sampling = styling and calib is not None
                if seg_job is not None and seg_job.done():
                    seg_last, seg_job = seg_job.result(), None
                if (erasing or sampling) and seg_job is None:
                    # copy: the frame is erased in place
                    seg_job = segmenter.submit(frame.copy(), ts, classes=sampling)
                res = landmarker.detect_for_video(img, ts)
                prof.lap("landmark")
                if grab_at is not None and time.monotonic() >= grab_at:
                    eraser.grab(frame)
                    grab_at = None
                if erasing or sampling:
                    if seg_job is not None:
                        try:
                            # while style-sampling (calibration, ~1 s) wait for this frame's own class map:
                            # the class map makes the job slower than SEG_WAIT, so every result came a
                            # frame late and never matched its landmarks (0 samples in 30 frames)
                            seg_last, seg_job = seg_job.result(timeout=None if sampling else SEG_WAIT), None
                        except FutureTimeout:
                            prof.lap("seg_late")
                    prof.lap("seg_wait")
                # style sample: raw (not yet erased) frame, its own class map, facing the camera
                if (sampling and res.face_landmarks and seg_last is not None and seg_last[2] == ts
                        and seg_last[1] is not None and puppet_style.frontal(res.face_landmarks[0], w, h)):
                    style_samples.append(puppet_style.sample(frame, res.face_landmarks[0], seg_last[1]))
                if erasing:
                    conf = seg_last[0] if seg_last is not None else None
                    if conf is not None:
                        box = None
                        if res.face_landmarks:
                            xs = [q.x * w for q in res.face_landmarks[0]]
                            ys = [q.y * h for q in res.face_landmarks[0]]
                            box = (min(xs), min(ys), max(xs), max(ys))
                        person = eraser.person_mask(conf, box)
                        if align_job is not None and align_job.done():
                            fit, align_job = align_job.result(), None
                            status = eraser.update(fit)
                            if status == "aligned":
                                print("camera moved: empty-room plate re-aligned")
                            elif status == "mismatch":
                                print(f"empty-room plate doesn't match this room (error {eraser.last_err:.0f}): "
                                      "erase is off. Run with --preview, press b, step out of frame")
                        # first check ~5 frames in, so a plate from another desk is caught within ~1 s
                        if align_job is None and frame_i % 30 == 5:
                            align_job = aligner.submit(eraser.align, frame.copy(), person.copy())
                        frame = eraser.apply(frame, person)
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
                            if styling:
                                styling = False
                                found = puppet_style.decide(style_samples)
                                if found is None:
                                    print("style: no frames facing the camera; keeping the current look. "
                                          "Face the camera and press c (needs --preview) to retry")
                                else:
                                    puppet_style.save(args.style, found)
                                    cfg.update(look(args, found))
                                    renderer.submit(gpu.set_cfg, dict(cfg)).result()
                                    head_key = None                      # re-render with the new look
                                    print(f"style from {len(style_samples)} frames: "
                                          f"{puppet_style.summary(found)} (saved to {args.style})")
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
                    # the head renders on the worker; until a new one lands, reuse the last one
                    if pending is not None and pending.done():
                        head_cache, pending = pending.result(), None
                    if pending is None and (head_key is None or np.max(np.abs(key - head_key)) >= 1.0):
                        head_key, pending = key, renderer.submit(gpu.head, st)
                        prof.lap("submit")
                    if head_cache is None:                    # very first head: wait for it
                        head_cache, pending = pending.result(), None
                    canvas, alpha, nose_c = head_cache
                    M = head_matrix(cx, cy, fh, roll, args.scale, args.lift, nose_c, (nx, ny))
                    head_c = M @ np.array((ANCHOR[0] * S, ANCHOR[1] * S + PADT, 1.0))

                    # body rides under the head: follows with a little lag (sitting up, leaning,
                    # leaning in) but the neck is clamped to +/-12% so it can never stretch
                    if not args.no_body:
                        unit = 0.55 * fh * args.scale * args.body_size
                        # torso turns a third as far as the head; re-rendered per 1.5 deg on the worker,
                        # and when --neck changes (i/k keys) since the neck length follows it
                        bkey = (round(math.degrees(BODY_TURN * st["yaw"]) / 1.5), round(args.neck, 2))
                        if body_job is not None and body_job[1].done():
                            body = {"key": body_job[0], "src": body_job[1].result(), "q": None,
                                    "pos": body["pos"] if body else None}
                            body_job = None
                        if body_job is None and (body is None or body["key"] != bkey):
                            body_job = (bkey, renderer.submit(gpu.body, math.radians(bkey[0] * 1.5),
                                                              args.body_size, args.neck))
                        if body is None:                      # very first body: wait for it
                            body = {"key": body_job[0], "src": body_job[1].result(), "q": None, "pos": None}
                            body_job = None
                        q = round(unit / BODY_UNIT / 0.02) * 0.02          # quantized rescale, cached
                        if body["q"] != q:
                            c0, a0, ox0, oy0 = body["src"]
                            sz = (max(1, int(c0.shape[1] * q)), max(1, int(c0.shape[0] * q)))
                            body["q"], body["spr"] = q, (cv2.resize(c0, sz, interpolation=cv2.INTER_AREA),
                                                         cv2.resize(a0, sz, interpolation=cv2.INTER_AREA),
                                                         int(ox0 * q), int(oy0 * q))
                            body["rolled"] = None
                        # shoulders tilt with the head (part of its roll) about the base of the neck;
                        # 0.5 deg steps, cached with the rescale
                        rb = round(BODY_ROLL * roll / 0.5) * 0.5
                        if body.get("rolled") is None or body["rolled"][0] != rb:
                            body["rolled"] = (rb, roll_sprite(*body["spr"], rb))
                        fc, fa, ox, oy = body["rolled"][1]
                        neck = args.neck * fh * args.scale
                        want = np.array((head_c[0], head_c[1] + neck))
                        pos = want if body["pos"] is None else body["pos"] + args.body_follow * (want - body["pos"])
                        dy = float(np.clip(pos[1] - head_c[1], 0.97 * neck, 1.03 * neck))
                        dx = float(np.clip(pos[0] - head_c[0], -0.10 * neck, 0.10 * neck))
                        pos = np.array((head_c[0] + dx, head_c[1] + dy))
                        body["pos"] = pos
                        bx0, by0 = int(pos[0] - ox), int(pos[1] - oy)
                        blit(frame, fc, fa, bx0, by0)                       # torso + neck, one 3-D render
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
                        styling, style_samples = can_style, []
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
        finally:
            if grabber is not None:
                grabber.stop()
            renderer.shutdown(cancel_futures=True)
            aligner.shutdown(cancel_futures=True)
        prof.report()
    landmarker.close()
    if segmenter is not None:
        segmenter.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
