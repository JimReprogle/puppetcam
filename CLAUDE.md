# puppetcam — handoff for Claude Code

Real-time felt-puppet webcam filter. Reads the physical webcam, tracks the face, erases the
person using a saved empty-room photo, draws an original felt puppet (head + torso), and writes
frames to the OBS Virtual Camera via `pyvirtualcam` for Teams/Zoom. Single file, CPU only.

```
puppetcam.py            # everything (~1150 lines)
tests/render_check.py   # offline render regression harness (no camera needed)
face_landmarker.task    # auto-downloaded on first run
selfie_segmenter.tflite # auto-downloaded on first run
empty_room.png          # created by pressing `b` in the preview
```

## Run

```bash
python puppetcam.py --preview            # live, with preview window + tuning keys
python tests/render_check.py             # offline checks; exit 0 = pass; PNG sheet in /tmp/puppetcam_check
```

Environment (pinned, verified working):

```bash
pip install mediapipe==1.2.0 pyvirtualcam==0.15.0 opencv-python numpy
```

- Python 3.14 recommended. `pyvirtualcam` 0.15.0 wheels cover CPython 3.9–3.14 on macOS and Windows x64 only.
  On 3.15 it must be built from source (`git clone --recursive --branch v0.15.0 https://github.com/letmaik/pyvirtualcam`).
- `mediapipe` 1.2.0 wheels: macOS arm64, Windows x64/arm64. No Intel Mac.
- **Do not use `mp.solutions`** — removed in mediapipe ≥0.10.30. The code uses the Tasks API
  (`FaceLandmarker`, `ImageSegmenter`) with `RunningMode.VIDEO` and monotonic ms timestamps.
- OBS must be installed (provides the camera extension/driver); the OBS app need not run, and its own
  Virtual Camera must be stopped while puppetcam runs.
- Primary dev machine: Apple Silicon Mac. Windows path is written but has never been executed.

## Per-frame pipeline (`main`)

1. `Grabber` thread reads the camera and keeps only the newest frame (`--video` files are read in-line
   so benchmarks see every frame) → `mp.Image` (SRGB, contiguous); mediapipe timestamps = capture time.
2. `FaceLandmarker.detect_for_video` with `output_face_blendshapes=True`.
3. Empty-room capture timer (`b`, 5 s) runs before anything is drawn, so the plate is the raw frame.
4. `Eraser.apply`: `ImageSegmenter` confidence mask → quarter-res mask → dilate/feather → replace person
   pixels with `empty_room.png`, per-channel gain-matched to current exposure on clear background.
   Mask polarity is auto-detected once against the face box.
5. `measure()` → pose `(cx, cy, fh, roll, yaw, pitch, nose_x, nose_y)` and expression
   `(mouth, lid, gaze_x, gaze_y, brow_up, brow_down, corners)`; both EMA-smoothed (`Smooth`).
   First 30 detections = neutral yaw/pitch calibration (`c` to redo).
6. Head cache: `head_key` is a continuous state vector in step units (1°, 0.05 mouth/lid, 0.1 gaze/brow/
   corners); `draw_puppet` re-runs only when any value moves ≥1 step from the last render (deadband, not
   rounding: rounding re-rendered 75–83 % of frames on jitter). Renders run on a `ThreadPoolExecutor(1)`;
   the loop keeps compositing the previous head until the new one lands (first head is awaited).
7. `head_matrix()` → affine M (scale `fh*scale/S`, roll, lift, then horizontal nose alignment).
8. Body: back layer (`bc`) → `draw_neck` → front layer (`fc`) → `cast_head_shadow`.
9. `composite()` warps/blends the head canvas inside its ROI only (`warp_roi`).
10. `vcam.send(frame)`; preview + keys.

## Coordinate systems — read before touching geometry

- Head canvas is `S×S` (`S=512`); all head geometry is in **fractions of S** (`L(v)` → px).
  Output canvas from `draw_puppet` is `OUT_H × S` with `PADT` headroom on top (Pez lift) and `PADB` below.
  `ANCHOR=(0.5,0.40)` (+`PADT` in output coords) is the point pinned to the real face center.
- `make_proj(yaw, pitch)` returns `pr(x, y, z)`: orthographic 2.5D rotation; `z` = depth toward camera
  in canvas units, scaled by `DEPTH`. +yaw = nose toward image-right, +pitch = nose down.
  Every feature has a depth: nose .30–.32, glasses .27, mustache .25, eyes .20, mouth .22, head center .06.
- `fh` is the **3D** forehead(10)–chin(152) distance (landmark z is in x-scale units). Do not revert to the
  2D distance — it shrinks with pitch and made the neck/body stretch.
- Yaw = `atan2(dz, dx)` of cheek 234→454; pitch = `atan2(dz, dy)` of forehead→chin. MediaPipe z: smaller = closer.

## `draw_puppet` stages (in order)

1. Masks/state; head is a superellipse (`--pez` → exponent 2.0–3.6), widened by `|sin yaw|`.
2. Hair cap = superellipse ∩ above-hairline polygon (`hairline_y/z`); turns with the head.
3. Nose/brows/beard/mustache color shapes; `fur_m` collects raised fur masses; `cast` collects hard props.
4. Fleece: dilate-bleed color → `remap` with felt displacement → nostrils + closed mouth line.
5. **Sculpt**: height field `hf` at S/2 from `add_ellipsoid` primitives (head dome, nose, cheeks, sockets,
   brow ridge, lips, chin) + raised hair/fur. `sculpt_light(hf)` = key + fill + AO + height-field cast
   shadows (8-step march toward `LIGHT`) + rim sheen + spec. Multiplied with felt texture (`pile/fiber/mottle`).
6. Strands (hair clipped to cap, brows, beard), eyes (`draw_eye`), glasses (solid alpha).
7. Headphones: band on the head; each cup+arm on its own layer → `composite_cups` (opaque cups, occluded by
   a head-edge that sweeps inward as the cup turns away; `front` weight from `sin yaw`).
8. Contact shadows from `cast` (eyes, glasses, headphones), halo sheen, fuzzy alpha (solid inside: `soft>0.8 → 1`).
9. **Hinge split** at the mouth row `yh` (only if `mouth >= MOUTH_OPEN`): upper part lifts `0.17*mo` and
   foreshortens `1-0.22*mo`; jaw drops `0.05*mo`; gap = cheek web (skin × head shading × felt) + cavity
   (width from the head mask row, clamped inside the head). Cup pixels below `yh` ride with the upper part.
10. Returns `(bgr, alpha, nose_point_in_output_coords)`.

## Invariants — each one is a bug that already shipped once

| Rule | What broke without it |
|---|---|
| Interior alpha forced to 1 (`a[soft > 0.80] = 1.0`) | Fuzz noise made the face see-through |
| Cheek web sized from the **head felt mask** `m`, never final alpha `a` | Skin slab spanning cup-to-cup when hinge met the cups |
| Cup pixels below `yh` go to the upper part; jaw alpha hard-cut where `cup_m > 0.01` | Dark cup sliver line under the open mouth |
| Cups never alpha-blended front/back; occlusion is a moving edge | Ghost cups showing the face through them |
| `front` weight = 1 at yaw 0; drops over ~0.05–0.20 of `sin yaw` | Cups popping in/out at a threshold |
| No ears; no far-side headphone sliver at the silhouette | Ear sliding onto the face; seam lines |
| No smile-fold grooves in `hf` | Folds slid into the nose under pitch → lines through the nose |
| Soft-rim profile for non-dome ellipsoids (`n == 2` branch) | Light caught steep rims → crease rings |
| Mustache not in `cast` | Its shadow read as a second mustache |
| Closed mouth line XOR open cavity, single threshold `MOUTH_OPEN` | Two mouths at small openings |
| Web color not sampled from an image row | Mustache gray smeared across the gap |
| Head center `z=+0.06` (moves with features) | Glasses/nose slid off the head edge |
| Body rigidly attached under head (`--body-follow 1`), neck length from 3D `fh`, clamp ±3 % | Stretchy neck when sitting up |
| Neck drawn between body back and front layers | Neck outside the collar |
| Torso shading analytic barrel, not distance-transform | Diagonal ridge crease on the shirt |
| Composite/shadow warps limited to ROI | 36 ms full-frame warps |

Add a check to `tests/render_check.py` for any new visual bug before fixing it.

## Performance

- Apple Silicon, 1280×720, erase on (`--profile` on the recorded clip, 2026-10-29): frame **33.3 ms = 30 fps cap**,
  main-thread work ≈ 21 ms: landmark 7.1, erase 8.4 (segmenter ≈ 3.5 of it), body 4.0, composite 0.7, send 0.5.
  `draw_puppet` ≈ 47 ms on the render thread, ~25–35 % of frames. Before this work: 80 ms/frame (13 fps).
  Live camera with a real plate: 33.2–34.6 ms/frame (p95 36.6–40.9), ~20 ms of it main-thread work.
- Camera indices shift with what's attached; the OBS Virtual Camera has a number too. Never point
  `--cam` at it (feedback loop). Check with `system_profiler SPCameraDataType`.
- Blends are uint8 `cv2.blendLinear` (no float32 frame copies); `cast_head_shadow` buffers are window-sized
  and its wide blur runs at 1/4 res. Keep new per-frame work in that style.
- Benchmark: `python puppetcam.py --video clip.mp4 --profile --plate <png>` (a GIL build; a render process
  measured the same as the thread). Next wins: segmenter at lower input res; strands/`hf` at lower res.
- Owner priority: **speed over features** (hand/arm tracking was evaluated and declined for this reason).

## CLI

| Switch | Default | Notes |
|---|---|---|
| `--skin` light/medium/tan/deep | medium | |
| `--hair`, `--beard` gray/salt/white/brown/black | salt / gray | |
| `--no-beard`, `--no-glasses`, `--no-headphones`, `--no-body`, `--no-erase` | off | flags |
| `--shirt` blue/navy/red/green/plaid-gray | blue | plaid-gray is solid gray |
| `--lid` | 0.35 | 0–0.9 |
| `--scale` | 2.4 | head size vs face |
| `--pez` | 1.0 | 0 oval – 1 block |
| `--long` | 1.0 | 0–1.2 |
| `--lift` | 0.15 | fraction of face height |
| `--body-size` | 1.25 | |
| `--shoulders` | 0.80 | 0.5–1.2 |
| `--neck` | 0.30 | collar below head center |
| `--body-follow` | 1.0 | 1 = rigid |
| `--mouth-gain` | 2.2 | jawOpen multiplier |
| `--brow-gain` | 1.8 | |
| `--turn` | 1.0 | yaw/pitch gain |
| `--blink` | 0.45 | eyeBlink threshold |
| `--plate`, `--model`, `--seg-model` | beside script | paths |
| `--cam`, `--width`, `--height`, `--fps` | 0, 1280, 720, 30 | |
| `--video` | none | read a video file instead of the camera (benchmarks; not paced, every frame processed) |
| `--profile` | off | per-stage mean/p95 ms every 150 frames and on exit |
| `--preview` | off | keys: `w/s` lift, `=/-` scale, `i/k` neck, `b` plate, `c` calibrate, `q` quit |

## Backlog

- Roll gain / roll lock option (owner flagged head tilt as possibly exaggerated).
- First real run on Windows (camera index, OBS registration).
- `plaid-gray` should either become a real plaid texture or be renamed.
- Arms/hands via PoseLandmarker/HandLandmarker exist in mediapipe 1.2.0 — deferred for speed.

## Design guardrails

The puppet is an original design. Keep it that way: no recreations of specific puppet characters
(including Sesame Street / Muppets characters) and no copies of specific branded product designs
(the headphones are deliberately generic). Generic puppet mechanics (hinged mouth, felt, ping-pong eyes) are fine.

## Owner conventions

- Lead with the answer; code blocks and exact CLI commands; moderate length.
- Every claim fact-based: measure (harness numbers, timings) rather than assert.
- Verify against the harness and a contact-sheet PNG before reporting a visual fix as done.
