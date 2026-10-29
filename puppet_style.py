"""Auto-style: read the subject's facial hair and hair/beard/brow colors from calibration frames.

What is detected, and how (measured on the owner's frames, 2026-10-29):
  - Facial hair by COLOR, not segmenter class: the multiclass segmenter labels a goatee "body-skin" and
    brows "face-skin", so its classes can't find facial hair. Hair (gray, dark or brown) is far less
    saturated than skin, so each region's share of pixels with Lab chroma < 0.5 x the forehead's chroma
    says whether hair grows there. Chroma ratios survive lighting; plain color distance did not
    (brightness gradients made every region read as "different").
      upper lip: 0.02 shaved vs 0.33-0.47 with a mustache; chin: 0.39-1.0 with a goatee;
      jaw (no full beard): 0.12-0.21 on the side facing the camera. The far-side jaw picks up background
      when turned, so the smaller of the two sides is used.
  - Head-hair color from the segmenter's hair class (that one is reliable); beard and brow colors from the
    desaturated pixels in their regions. Colors are judged RELATIVE TO THE SUBJECT'S OWN SKIN (lightness
    ratio + chroma) so camera exposure cancels, then mapped to the nearest felt palette color.
Not detected, on purpose: skin tone (measured lightness swung 81-139 with lighting; it would flip
between tones) and head-hair coverage (the owner chose the full hairline with a widow's peak).
"""
import json
import os

import cv2
import numpy as np

# MediaPipe face-mesh indices
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
REGIONS = {
    "mustache": [98, 97, 2, 326, 327, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
    "goatee": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 152, 148, 176, 377, 400],
    "jaw_l": [93, 132, 58, 172, 136, 150, 149, 176, 214, 207, 205, 50, 123],
    "jaw_r": [323, 361, 288, 397, 365, 379, 378, 400, 434, 427, 425, 280, 352],
    "brow_l": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "brow_r": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    "forehead": [109, 10, 338, 297, 299, 9, 69, 67],
}
HAIR_CLASS = 1                 # selfie_multiclass_256x256: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other
DESAT = 0.5                    # a pixel is "hair-like" below this fraction of the skin's chroma
MUSTACHE_MIN, GOATEE_MIN, FULL_BEARD_MIN = 0.2, 0.25, 0.45
MIN_PIXELS = 30


FRONTAL_DEG = 15.0             # sample only near-frontal frames: turned, the regions land on neck shadow


def facing(lm, w, h):
    """(yaw, pitch) in degrees relative to the CAMERA (not the user's neutral), same formulas as
    puppetcam.measure: cheek 234->454 and forehead 10->chin 152, MediaPipe z in x-scale units."""
    def p3(i):
        return np.array((lm[i].x * w, lm[i].y * h, lm[i].z * w))

    v, u = p3(454) - p3(234), p3(152) - p3(10)
    return float(np.degrees(np.arctan2(v[2], v[0]))), float(np.degrees(np.arctan2(u[2], u[1])))


def frontal(lm, w, h):
    yaw, pitch = facing(lm, w, h)
    return abs(yaw) < FRONTAL_DEG and abs(pitch) < FRONTAL_DEG


def region_masks(lm, w, h):
    """Boolean masks per region (lips cut out), from one frame's landmarks."""
    def pts(idx):
        return cv2.convexHull(np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in idx], np.int32))

    lips = np.zeros((h, w), np.uint8)
    cv2.fillPoly(lips, [pts(LIPS)], 1)
    lips = cv2.dilate(lips, np.ones((7, 7), np.uint8))
    out = {}
    for name, idx in REGIONS.items():
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts(idx)], 1)
        out[name] = (m > 0) & (lips == 0)
    # a goatee hangs below the face mesh (chin = landmark 152); its lit lower half is the truest color
    fh = abs(lm[152].y - lm[10].y) * h
    chin = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in (172, 148, 152, 377, 397)], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(np.vstack([chin, chin + (0, int(0.3 * fh))]))], 1)
    out["below_chin"] = m > 0
    return out


def _lch(lab_px):
    """(lightness, chroma) of Lab pixels (OpenCV 8-bit Lab: a, b offset by 128)."""
    return lab_px[..., 0], np.hypot(lab_px[..., 1] - 128.0, lab_px[..., 2] - 128.0)


def sample(frame, lm, classes):
    """Measurements from one frame: hair-like fractions per facial-hair region and (lightness, chroma)
    of skin, head hair, beard and brows. `classes` is the segmenter's argmax class map or None."""
    h, w = frame.shape[:2]
    m = region_masks(lm, w, h)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    light, chroma = _lch(lab)
    if m["forehead"].sum() < MIN_PIXELS:
        return None
    skin_l, skin_c = float(np.median(light[m["forehead"]])), float(np.median(chroma[m["forehead"]]))
    hairy = chroma < DESAT * skin_c

    def frac(name):
        return float(hairy[m[name]].mean()) if m[name].sum() >= MIN_PIXELS else 0.0

    def color(mask):
        # lit strands (75th lightness percentile), not the median: shadow between hairs is also
        # desaturated and dragged a light-gray goatee down to "black"
        if mask is None or mask.sum() < MIN_PIXELS:
            return None
        return [float(np.percentile(light[mask], 75)), float(np.median(chroma[mask]))]

    fr = {"mustache": frac("mustache"), "goatee": frac("goatee"), "jaw": min(frac("jaw_l"), frac("jaw_r"))}
    # beard color only from regions where hair was found (the mustache region holds the nostrils)
    beard_m = np.zeros((h, w), bool)
    if fr["goatee"] > GOATEE_MIN:
        beard_m |= m["goatee"] | m["below_chin"]
    if fr["mustache"] > MUSTACHE_MIN:
        beard_m |= m["mustache"]
    if fr["jaw"] > FULL_BEARD_MIN:
        beard_m |= m["jaw_l"] | m["jaw_r"]
    head_px, head_color = None, None
    if classes is not None:
        # the hair band is thin and its edge blends into a bright background: erode, then median
        # (raw p75 called dark salt-and-pepper hair "white")
        head_px = cv2.erode((classes == HAIR_CLASS).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if head_px.sum() >= MIN_PIXELS:
            head_color = [float(np.median(light[head_px])), float(np.median(chroma[head_px]))]
        beard_m &= classes != 0
    # beard: no clothes-class filter (the segmenter labels a goatee "clothes": 982 of 1490 goatee pixels
    # on the owner's frame, so the filter left only neck shadow and called a gray goatee "black"). The
    # 90th lightness percentile picks lit strands; a dark shirt behind the goatee can't reach it
    beard_px = beard_m & hairy
    beard = None
    if beard_px.sum() >= MIN_PIXELS:
        beard = [float(np.percentile(light[beard_px], 90)), float(np.median(chroma[beard_px]))]
    return dict(fr, skin=[skin_l, skin_c], hair=head_color, beard=beard,
                brow=color((m["brow_l"] | m["brow_r"]) & hairy))


def hair_name(meas, skin):
    """Felt hair color from (lit lightness, chroma), judged relative to the subject's skin lightness so
    exposure cancels. Brown = clearly warm; otherwise gray tiers by lightness (webcam hair reads darker
    than it looks, so the tiers sit low)."""
    rel, c = meas[0] / max(skin[0], 1.0), meas[1]
    if c > 11.0 and rel < 0.8:
        return "brown"
    if rel < 0.3:
        return "black"
    if rel < 0.62:
        return "salt"
    if rel < 0.95:
        return "gray"
    return "white"


def decide(samples):
    """Median over frames -> style dict (facial-hair flags + felt color names + raw measurements)."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None

    def med(key):
        return float(np.median([s[key] for s in samples]))

    def med_color(key):
        vals = [s[key] for s in samples if s[key] is not None]
        return np.median(np.array(vals), axis=0).tolist() if vals else None

    skin = med_color("skin")
    meas = {"mustache": med("mustache"), "goatee": med("goatee"), "jaw": med("jaw"),
            "skin": skin, "hair": med_color("hair"), "beard": med_color("beard"), "brow": med_color("brow")}
    style = {"mustache": meas["mustache"] > MUSTACHE_MIN,
             "goatee": meas["goatee"] > GOATEE_MIN,
             "full_beard": meas["jaw"] > FULL_BEARD_MIN}
    for key in ("hair", "beard", "brow"):
        style[key] = hair_name(meas[key], skin) if meas[key] else None
    if style["brow"] is None:
        style["brow"] = style["hair"]
    if style["beard"] is None:
        style["beard"] = style["hair"]
    style["measured"] = meas
    return style


def summary(style):
    hair = [k for k in ("mustache", "goatee", "full_beard") if style[k]] or ["clean-shaven"]
    return (f"{', '.join(h.replace('_', ' ') for h in hair)}; hair {style['hair']}, beard {style['beard']}, "
            f"brows {style['brow']}")


def save(path, style):
    with open(path, "w") as f:
        json.dump(style, f, indent=2)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
