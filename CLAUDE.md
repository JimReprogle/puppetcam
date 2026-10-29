# puppetcam — handoff for Claude Code

Real-time felt-puppet webcam filter. Reads the physical webcam, tracks the face, erases the
person using a saved empty-room photo, renders an original 3-D felt puppet (head + torso) on the GPU,
and writes frames to the OBS Virtual Camera via `pyvirtualcam` for Teams/Zoom. macOS only (Metal).

```
puppetcam.py            # capture, tracking, erase, compositing, CLI (~590 lines)
puppet_gpu.py           # GpuPuppet: wgpu device, uniforms, render + readback, CPU nose projection
puppet.wgsl             # the puppet: raymarched signed-distance shapes + fleece shading
tests/render_check.py   # offline render regression harness (no camera needed; needs the GPU)
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
pip install mediapipe==1.2.0 pyvirtualcam==0.15.0 wgpu==0.32.0 opencv-python numpy
```

- `wgpu` 0.32.0 is a `py3-none` wheel (any CPython); it runs on Metal. Verified on an M2 Pro.

- Python 3.14 recommended. `pyvirtualcam` 0.15.0 wheels cover CPython 3.9–3.14 on macOS and Windows x64 only.
  On 3.15 it must be built from source (`git clone --recursive --branch v0.15.0 https://github.com/letmaik/pyvirtualcam`).
- `mediapipe` 1.2.0 wheels: macOS arm64, Windows x64/arm64. No Intel Mac.
- **Do not use `mp.solutions`** — removed in mediapipe ≥0.10.30. The code uses the Tasks API
  (`FaceLandmarker`, `ImageSegmenter`) with `RunningMode.VIDEO` and monotonic ms timestamps.
- OBS must be installed (provides the camera extension/driver); the OBS app need not run, and its own
  Virtual Camera must be stopped while puppetcam runs.
- Apple Silicon Mac only, by owner decision (2026-10-29). wgpu would also run on DX12/Vulkan, but
  Windows was never executed and is out of scope.

## Per-frame pipeline (`main`)

1. `Grabber` thread reads the camera and keeps only the newest frame (`--video` files are read in-line
   so benchmarks see every frame) → `mp.Image` (SRGB, contiguous); mediapipe timestamps = capture time.
2. `FaceLandmarker.detect_for_video` with `output_face_blendshapes=True`.
3. Empty-room capture timer (`b`, 5 s) runs before anything is drawn, so the plate is the raw frame.
4. `Eraser.apply`: `ImageSegmenter` confidence mask → quarter-res mask → dilate/feather → replace person
   pixels with `empty_room.png`, per-channel gain-matched to current exposure on clear background.
   Mask polarity is auto-detected once against the face box.
   **Plate alignment**: about once a second, `Eraser.align` (own worker, ~90 ms) fits the as-captured
   `plate0` to the live view with ORB on clear background + RANSAC homography, and the fit is adopted
   only if it beats the current plate by ≥20 % and ≥3 levels (median |frame−plate| on clear background).
5. `measure()` → pose `(cx, cy, fh, roll, yaw, pitch, nose_x, nose_y)` and expression
   `(mouth, lid, gaze_x, gaze_y, brow_up, brow_down, corners)`; both EMA-smoothed (`Smooth`).
   First 30 detections = neutral yaw/pitch calibration (`c` to redo).
6. Head cache: `head_key` is a continuous state vector in step units (1°, 0.05 mouth/lid, 0.1 gaze/brow/
   corners); `gpu.head` re-runs only when any value moves ≥1 step from the last render (deadband, not
   rounding: rounding re-rendered 75–83 % of frames on jitter). The `GpuPuppet` is created on, and every
   render runs on, one `ThreadPoolExecutor(1)` worker; the loop keeps compositing the previous head until
   the new one lands (first head is awaited).
7. `head_matrix()` → affine M (scale `fh*scale/S`, roll, lift, then horizontal nose alignment).
8. Body: `gpu.body(yaw)` sprite (torso + neck in one 3-D render; torso turns `BODY_TURN`=0.35 of head yaw,
   re-rendered per 1.5° on the worker), rescaled to the face size, `blit`, then `cast_head_shadow`.
9. `composite()` warps/blends the head canvas inside its ROI only (`warp_roi`).
10. `vcam.send(frame)`; preview + keys.

## Coordinate systems — read before touching geometry

- Head output canvas is `OUT_H × S` (`S=512`) with `PADT` headroom on top and `PADB` below.
  `ANCHOR=(0.5,0.40)` (+`PADT` in output coords) is the point pinned to the real face center.
- Shader world frame: x right, **y up**, z toward the camera; 1 unit = S px (head) or `BODY_UNIT` px (body).
  Perspective camera at `(0,0,F)` looking −z (`F_HEAD=3.0`); the z=0 plane maps 1:1 onto the canvas, so the
  2-D placement code didn't change. Head pivots at the origin (= ANCHOR). World→local = `rotX(rotY(p,−yaw),−pitch)`;
  +yaw = nose toward image-right, +pitch = nose down.
- Head local landmarks: front face z≈0.20, nose tip z≈0.325, glasses z 0.285, eyes centered z 0.175,
  hinge `(0, −0.165, −0.06)`; top ≈ +0.34, chin ≈ −0.41, goatee tip ≈ −0.55.
- `fh` is the **3D** forehead(10)–chin(152) distance (landmark z is in x-scale units). Do not revert to the
  2D distance — it shrinks with pitch and made the neck/body stretch.
- Yaw = `atan2(dz, dx)` of cheek 234→454; pitch = `atan2(dz, dy)` of forehead→chin. MediaPipe z: smaller = closer.

## GPU puppet (`puppet.wgsl` + `puppet_gpu.py`)

One full-screen fragment pass per render; `mode` uniform picks head (`headMap`) or torso (`bodyMap`).
Render target rgba8 (premultiplied), read back, downsampled as uint8, then unpremultiplied on the fringe only.
Head renders at 2× supersampling; body at 1× (`BODY_UNIT`=300 px/unit, main rescales).

1. **Head shape** `headBase`: mix(ellipsoid, rounded box) by `--pez`, height by `--long`, wider at the jaw.
   `skull` smooth-unions brow ridge, cheeks, chin, nose; subtracts nostrils; raises the hair cap (`hairAmt`:
   sides/back to the nape, bald dome over the front crown).
2. **Upper parts** (ride the upper head, never cut): ping-pong eyes, felt lid shells (edge height from
   `lid`), fur brows (`brow` up/down), mustache, glasses (rings, bridge, temples), headphones (band, sliders,
   cups, pads). **Lower parts**: goatee (tapered, fur), tongue.
3. **Hinge** (`headMap`): skull evaluated twice, upper rotated up `UP_ANG*mouth` about `HINGE`, jaw down
   `JAW_ANG*mouth`; each half cut by the hinge plane, cut faces = mouth material; a mouth sack (dark red
   centre, darker-skin cheek web at the sides) fills the gap.
4. **Fleece shading** `shade`: sRGB→linear albedo × fiber noise; bumped normal; wrap diffuse key with soft
   shadow (28-step SDF march), hemispheric fill × AO; grazing sheen instead of specular. Only eyes, glasses,
   headphones and buttons get a highlight. Closed mouth = stitched line along the hinge, bent by `corners`.
5. **Silhouette**: rays that miss but pass within `fuzz` of a felt surface get noisy partial alpha
   (fiber halo); hard props get none.
6. `GpuPuppet.nose_point` re-implements the hinge/rotation/projection in numpy to return the nose tip in
   output px. `HINGE`, `UP_ANG`, `NOSE`, `F_HEAD` must match the shader.

## Invariants — each one is a bug that already shipped once

| Rule | What broke without it |
|---|---|
| Closed mouth line XOR open hinge, single threshold `MOUTH_OPEN` (Python zeroes `mouth` below it) | Two mouths at small openings |
| Mouth sack x-radius ≥ head half-width (0.258) | Background visible through the side of the open mouth when turned |
| Exposure: front-lit face ≈ albedo (key 0.88, fill 0.3); `felt` check counts clipped skin | Skin red channel clipped on 21k px: hot plastic look |
| No specular on felt materials (`isFelt`) | Plastic look (owner's stated failure mode) |
| `nose_point` mirrors the shader's hinge/rotation/camera; `nose` check | Puppet slides sideways against the face |
| GPU device created on, and used only from, the render worker | (design rule: one thread owns the device) |
| Downsample premultiplied uint8, unpremultiply only `0<a<1` | 11.6 ms float conversion per head render |
| Body rigidly attached under head (`--body-follow 1`), neck length from 3D `fh`, clamp ±3 % | Stretchy neck when sitting up |
| `--neck` default 0.45; neck cylinder reaches 1.0 body units, sprite top `BODY_TOP` 1.06 | Collar hidden behind the beard (head parked on the shoulders) |
| Composite/shadow warps limited to ROI | 36 ms full-frame warps |
| Plate re-aligned to the live view; always warp `plate0`, never the warped plate | Camera tilted after `b`: erased silhouette showed a shifted room = semi-transparent "ghost" of the person |
| Adopt a new plate fit only on a real error improvement, not on "fit moved" | Static-camera fits scatter 1–7 px; swapping each in made the erased patch jiggle every second |
| `use_aligned` refuses fits computed from a different `plate0` | A fit in flight during `b` would warp the old plate over the new one |

Add a check to `tests/render_check.py` for any new visual bug before fixing it.

## Performance

- M2 Pro, 1280×720, erase on (`--profile`, 2026-10-29): frame **33.3 ms = 30 fps cap**, main-thread work
  ≈ 20 ms: landmark 7.4, erase 8.6 (segmenter ≈ 3.5 of it), body 1.6–2.3, composite 0.7, send 0.5.
  GPU head ≈ 22–24 ms (2× SS: ~20 ms draw+readback), body ≈ 15 ms, both on the worker.
  Live camera: 33.2–33.4 ms/frame. Before the perf work: 80 ms/frame (13 fps) with the old 2-D renderer.
- Camera indices shift with what's attached; the OBS Virtual Camera has a number too. Never point
  `--cam` at it (feedback loop). Check with `system_profiler SPCameraDataType`.
- Blends are uint8 `cv2.blendLinear` (no float32 frame copies); `cast_head_shadow` buffers are window-sized
  and its wide blur runs at 1/4 res. Keep new per-frame work in that style.
- Benchmark: `python puppetcam.py --video clip.mp4 --profile --plate <png>` (a GIL build; a render process
  measured the same as the thread). Next wins: segmenter at lower input res; cheaper head shading
  (soft-shadow/AO step counts) if head update rate ever matters.
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
| `--neck` | 0.45 | collar below head center (the neck reaches up to ~0.6) |
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
- `plaid-gray` should either become a real plaid texture or be renamed.
- Erase leaks seen live 2026-10-29 (after the alignment fix): real shirt at the frame edge where the
  segmenter misses it; faint outline on the frosted door glass where local brightness differs from the
  plate (global gain can't fix local lighting). Candidates: refresh plate pixels from clear background each
  frame; local (tiled) gain.
- Arms/hands via PoseLandmarker/HandLandmarker exist in mediapipe 1.2.0 — deferred for speed.

## Design guardrails

The puppet is an original design. Keep it that way: no recreations of specific puppet characters
(including Sesame Street / Muppets characters) and no copies of specific branded product designs
(the headphones are deliberately generic). Generic puppet mechanics (hinged mouth, felt, ping-pong eyes) are fine.

## Owner conventions

- Lead with the answer; code blocks and exact CLI commands; moderate length.
- Every claim fact-based: measure (harness numbers, timings) rather than assert.
- Verify against the harness and a contact-sheet PNG before reporting a visual fix as done.
