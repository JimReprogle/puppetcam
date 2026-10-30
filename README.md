# puppetcam

A real-time webcam filter that replaces you with a 3-D felt puppet on video calls.

It reads your webcam, tracks your face, erases you from the picture, and draws an original felt puppet
(head and torso) that copies your head movement, mouth, eyes, brows and expression. It sends the result
to the OBS Virtual Camera, so Teams, Zoom or any other video app can use it as a normal camera.

- Runs at 30 fps at 1280×720 on Apple Silicon (measured on an M2 Pro).
- The puppet is a real 3-D model rendered on the GPU (Metal on macOS, Vulkan on Linux, DirectX 12 or
  Vulkan on Windows), shaded to look like fleece, not plastic.
- On first calibration it detects your facial hair and hair colours and styles the puppet to match.

**Platforms:** macOS on Apple Silicon (tested). Windows x64 and Linux x86-64 (written and
dependency-checked, not yet run on those platforms; see [Windows](#windows) and [Linux](#linux)).

## Contents

- [Setup](#setup)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Command-line reference](#command-line-reference)
- [Preview keys](#preview-keys)
- [Presets](#presets)
- [Files](#files)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Windows](#windows)
- [Linux](#linux)

## Setup

The commands below are for macOS. For Windows, see [Windows](#windows).

1. Install OBS Studio. It provides the virtual-camera driver. The OBS app does not have to run, but
   its own Virtual Camera must be stopped while puppetcam runs.
2. Create the Python environment:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install mediapipe==1.2.0 pyvirtualcam==0.15.0 wgpu==0.32.0 opencv-python numpy
   ```
   - Use Python 3.14 if you can. `pyvirtualcam` 0.15.0 has wheels for CPython 3.9–3.14. On 3.15 you
     must build it from source:
     `git clone --recursive --branch v0.15.0 https://github.com/letmaik/pyvirtualcam`.
   - `mediapipe` 1.2.0 has no Intel-Mac wheels.
3. Run it once. The face and person models download automatically on first start
   (`face_landmarker.task`, `selfie_multiclass_256x256.tflite`, and on Windows and Linux
   `selfie_segmenter.tflite`).
4. If macOS asks for camera access for the terminal, allow it.

## Quick start

1. Find your camera index:
   ```bash
   .venv/bin/python puppetcam.py --list-cams
   ```
   This prints every index that opens. Device names aren't available, and the OBS Virtual Camera is one
   of the indices too: never pick it, or puppetcam reads its own output. On macOS,
   `system_profiler SPCameraDataType` lists the names in index order.
2. Start puppetcam with the preview window:
   ```bash
   .venv/bin/python puppetcam.py --cam 0 --preview
   ```
3. Look straight at the camera for the first second. This calibrates your neutral head pose and detects
   your style.
4. Press `b`, step out of the frame, and wait 5 seconds. This captures `empty_room.png`, the empty-room
   photo used to erase you.
5. In your video app, select **OBS Virtual Camera** as the camera.

Later runs reuse the saved room photo and style. You only need `--preview` again to recapture or tune.

## How it works

Each frame goes through this pipeline:

1. **Capture.** A reader thread keeps only the newest camera frame, so the program never processes a
   stale, queued frame.
2. **Face tracking.** MediaPipe Face Landmarker finds 478 face points and expression scores
   ("blendshapes"). From these, puppetcam measures:
   - head position, size and tilt (roll)
   - turn (yaw) and nod (pitch)
   - mouth opening, eyelids, gaze, brow raise and frown, smile and frown.

   All values are smoothed. Size uses the 3-D forehead-to-chin distance, so leaning in or nodding
   does not stretch the puppet.
3. **Erase.** A person segmenter finds every pixel that is you (hair, skin and clothes):
   - **macOS (`gpu` mode):** the MediaPipe multiclass selfie model runs on the GPU every frame.
   - **Windows and Linux (`hybrid` mode):** MediaPipe's GPU delegate is not usable here (the Windows build
     disables it; the Linux wheel's delegate fails at EGL display creation off a desktop session), and the
     multiclass model takes about 124 ms per frame on a CPU. A fast single-class model (about 4 ms) runs
     every frame for your head and moving edges. The multiclass model refreshes the mask for your torso on
     a second thread whenever it is free. Coverage on a recorded test clip was the same as in `gpu` mode.
     A Linux desktop with a working GPU/EGL context can force `--seg-mode gpu`.

   Those pixels are replaced with the empty-room photo, matched to the
   current exposure. Two safeguards protect it:
   - **Alignment.** About once a second the room photo is re-aligned to the live view, so a bumped
     camera does not leave a ghost of you.
   - **Wrong room.** If the photo is from a different room, puppetcam turns erasing off and tells you
     to press `b`.
4. **Render.** The puppet head and torso are 3-D shapes built from smooth signed-distance primitives,
   ray-marched on the GPU through `wgpu`:
   - The **head** has a hinged mouth: the upper head lifts, the jaw drops, and the opening tapers to
     closed corners.
   - The **torso** turns a third as far as your head and tilts half as far as your head roll. Its neck
     sits inside the collar.
   - **Lighting** is felt-like: matte light, a soft sheen at grazing angles, fibre texture, soft
     shadows, ambient occlusion and a fuzzy outline.
   - The **eyebrows** are rendered as hair locks.
   - **Smiling and frowning** bend the mouth, open or closed. Smiling makes an open-mouthed grin
     and raises the cheeks.

   Renders run on a worker thread. The head re-renders only when your pose or expression changes by a
   full step.
5. **Composite.** The torso and the head's shadow are blended into the frame, then the head is warped to
   your face position and blended on top.
6. **Output.** The frame goes to the OBS Virtual Camera.

**Auto-style.** During calibration, puppetcam samples frames where you face the camera:
- **Facial hair** (mustache, goatee, full beard) is detected by colour saturation in each face region.
  Hair is much less saturated than skin.
- **Hair, beard and brow colours** are judged relative to your own skin tone, so camera exposure
  cancels out.

The result is saved to `puppet_style.json`. Command-line switches always override it.

## Command-line reference

Run `.venv/bin/python puppetcam.py --help` for the short form. Values outside a clamped range are
silently set to the nearest limit.

### Camera and output

| Switch | Default | Values | Description |
|---|---|---|---|
| `--cam N` | `0` | integer | Camera index. Never the OBS Virtual Camera. |
| `--width W` | `1280` | pixels | Requested capture width. |
| `--height H` | `720` | pixels | Requested capture height. |
| `--fps F` | `30` | integer | Requested capture and output frame rate. |
| `--video FILE` | none | path | Read frames from a video file instead of the camera. For benchmarking: not paced, every frame processed. |
| `--preview` | off | flag | Show a preview window and enable the [preview keys](#preview-keys). |
| `--profile` | off | flag | Print per-stage timings (mean, 95th percentile) every 150 frames and on exit. |
| `--list-cams` | off | flag | Print the camera indices that open, then exit. |

### Erasing you

| Switch | Default | Values | Description |
|---|---|---|---|
| `--no-erase` | off | flag | Do not erase you. The puppet draws over the live picture. Also disables auto-style. |
| `--plate FILE` | `empty_room.png` | path | Empty-room photo. Written when you press `b`. |
| `--seg-model FILE` | `selfie_multiclass_256x256.tflite` | path | Person segmentation model. Single-class models also work but lose the torso when you turn. |
| `--seg-mode` | `auto` | `auto`, `gpu`, `hybrid` | Person segmentation. `auto` picks `gpu` on macOS (falls back to `hybrid` if the GPU path fails) and `hybrid` on Windows and Linux. Force `gpu` on a Linux desktop with a working EGL/GPU context. |
| `--fast-seg-model FILE` | `selfie_segmenter.tflite` | path | Fast single-class model used every frame in `hybrid` mode. |
| `--model FILE` | `face_landmarker.task` | path | Face landmark model. |

### Look: skin, hair and facial hair

| Switch | Default | Values | Description |
|---|---|---|---|
| `--female` | off | flag | Feminine puppet preset: long face-framing hair, eyelashes, lips, and a bust, plus a narrower jaw and finer brows. Turns off facial hair and the headphones (the cups clip the hair curtains). Every underlying switch (`--hair`, `--shirt`, `--hairline`, …) still applies alongside it. |
| `--skin` | `medium` | `light`, `medium`, `tan`, `deep` | Felt skin colour. Not auto-detected, because lighting changes it too much. |
| `--hair` | auto, else `salt` | `gray`, `salt`, `white`, `brown`, `black` | Head-hair colour. |
| `--beard` | auto, else `gray` | same as `--hair` | Beard and mustache colour. |
| `--brows` | auto, else beard colour | same as `--hair` | Eyebrow colour. |
| `--hairline` | `widow` | `widow`, `straight`, `receding`, `bald` | Front hairline. `widow` is a widow's peak, `straight` a square front, `receding` a bald front crown, `bald` a fringe at the sides and back only. |
| `--forehead` | `0.0` | `0`–`1` (clamped) | Moves the front hairline up and back for a bigger forehead. `0.35`–`0.5` keeps the widow's peak visible. Has no effect with `bald`. |
| `--mustache` | auto | flag | Add a mustache. |
| `--no-mustache` | auto | flag | Remove the mustache. Wins over `--mustache`. |
| `--full-beard` | auto | flag | Full beard over the jaw and jowls. |
| `--no-beard` | auto | flag | No goatee and no full beard. |
| `--no-glasses` | off | flag | Remove the glasses. |
| `--no-headphones` | off | flag | Remove the headphones. |

### Auto-style

| Switch | Default | Values | Description |
|---|---|---|---|
| `--style FILE` | `puppet_style.json` | path | Saved detected style. Written by detection and loaded on later starts. |
| `--restyle` | off | flag | Ignore the saved style and detect again at calibration. |
| `--no-auto-style` | off | flag | No detection and no saved style. Uses only the defaults and your switches. |

The precedence is built-in defaults, then the saved or detected style, then explicit switches. Detection
needs you facing the camera, within about 15° in turn and nod. If no frame qualifies, it keeps the
current look and says so.

### Head shape and placement

| Switch | Default | Values | Description |
|---|---|---|---|
| `--scale` | `2.4` | > 0 | Puppet head size relative to your face. |
| `--pez` | `1.0` | `0`–`1` (clamped) | Head squareness: `0` oval, `1` block. |
| `--long` | `1.0` | `0`–`1.2` (clamped) | Head length: `0` round, `1.2` long. |
| `--lift` | `0.15` | fraction of face height | Raises the puppet above your face position. |

### Body

| Switch | Default | Values | Description |
|---|---|---|---|
| `--no-body` | off | flag | Head only. |
| `--shirt` | `blue` | `blue`, `navy`, `red`, `green`, `plaid-gray` | Shirt colour. `plaid-gray` is currently plain gray. |
| `--body-size` | `1.25` | > 0 | Torso size relative to the head. |
| `--shoulders` | `0.80` | `0.5`–`1.2` (clamped) | Shoulder width. |
| `--neck` | `0.45` | about `0.2`–`0.62` | Collar distance below the head centre, in head units. Lower values tuck the head onto the collar. The neck length follows it. |
| `--body-follow` | `1.0` | `0`–`1` | `1` attaches the body rigidly under the head. Lower values add lag. The neck length is clamped either way. |

### Expression and motion

| Switch | Default | Values | Description |
|---|---|---|---|
| `--mouth-gain` | `2.2` | > 0 | How far the mouth opens per unit of your jaw opening. |
| `--brow-gain` | `1.8` | > 0 | How far the brows move per unit of your brow movement. |
| `--lid` | `0.35` | `0`–`0.9` (clamped) | Resting upper-lid coverage. Higher looks sleepier. |
| `--blink` | `0.45` | `0`–`1` | Blink score above which the eyes close fully. |
| `--turn` | `1.0` | > 0 | Gain on head turn and nod. Limited to ±38° turn and ±25° nod. |

## Preview keys

These work only with `--preview`, while the preview window has focus.

| Key | Action |
|---|---|
| `b` | Capture the empty-room photo after a 5-second timer. Step out of frame. |
| `c` | Recalibrate your neutral pose and re-detect your style. Look straight at the camera. |
| `w` / `s` | Raise or lower the puppet (`--lift` ±0.01). |
| `=` / `-` | Grow or shrink the puppet (`--scale` ±0.05). |
| `i` / `k` | Move the body up or down (`--neck` ∓0.01). |
| `q` | Quit. |

Tuning keys print the matching switch value, so you can copy it into your command line.

## Presets

Use `--no-auto-style` with presets so your detected facial hair does not mix in.

**Grumpy grandpa:** large block head, bald dome, white beard and mustache, wild white brows.
```bash
.venv/bin/python puppetcam.py --cam 0 --preview --no-auto-style \
  --scale 3.0 --long 1.2 --pez 1 --lift 0.15 --body-size 0.8 --shoulders 0.6 \
  --hairline bald --hair white --beard white --brows white --full-beard --mustache \
  --lid 0.55 --mouth-gain 3.5 --brow-gain 3.0 --turn 1.4 --shirt red
```

**Bobblehead:** large rounder head on a small body, brown mustache, black brows.
```bash
.venv/bin/python puppetcam.py --cam 0 --preview --no-auto-style \
  --scale 3.2 --long 0.6 --pez 0.4 --lift 0.0 --body-size 0.7 --shoulders 0.55 --neck 0.35 \
  --hairline straight --hair brown --brows black --beard brown --mustache --no-beard \
  --lid 0.1 --mouth-gain 3.5 --brow-gain 3.0 --turn 1.5 --shirt green --skin light
```

## Files

| File | Purpose |
|---|---|
| `puppetcam.py` | Capture, tracking, erasing, compositing, command line, main loop. |
| `puppet_gpu.py` | `GpuPuppet`: the GPU device, shader parameters, render and read-back. |
| `puppet.wgsl` | The puppet itself: signed-distance geometry and fleece shading. |
| `puppet_style.py` | Auto-style: facial-hair detection and colour mapping. |
| `tests/render_check.py` | Offline render regression checks. Needs no camera. |
| `CLAUDE.md` | Detailed design notes, invariants and performance numbers for maintainers. |

These local files are not tracked in git:

| File | Purpose |
|---|---|
| `empty_room.png` | Your empty-room photo. |
| `puppet_style.json` | Your detected style. |
| `*.task`, `*.tflite` | Downloaded models. |

## Development

Run the regression checks after any change:
```bash
.venv/bin/python tests/render_check.py
```
- Exit code 0 means every check passes.
- Contact sheets are written to `/tmp/puppetcam_check/` (`poses.png`, `body.png`).
- Each check guards a bug that already shipped once. Add a check for a new visual bug before fixing
  it.

Benchmark the full pipeline on a recorded clip:
```bash
.venv/bin/python puppetcam.py --video clip.mp4 --profile --no-auto-style --plate empty_room.png
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `camera read failed` | Wrong `--cam` index or no camera permission. Run `--list-cams`. Allow camera access for your terminal (macOS) or apps (Windows privacy settings). On Linux check the device permission (your user must be in the `video` group). |
| `virtual camera unavailable` (Linux) | The v4l2loopback kernel module is not loaded. See [Linux](#linux). |
| A ghost of you behind the puppet | The room photo no longer matches. Small camera moves re-align automatically. Otherwise press `b` again. |
| `empty-room plate doesn't match this room` | You are at a different desk. Run with `--preview` and press `b`. |
| `style: no frames facing the camera` | Your camera is off to the side. Face it and press `c`. |
| Puppet too big or off-centre | Use the preview keys `w`/`s`/`=`/`-`, then put the printed values in your command. |
| Video app shows a black or OBS image | Stop the OBS app's own Virtual Camera, then restart puppetcam. |

## Windows

Every dependency has Windows x64 wheels for Python 3.13 and 3.14 (checked against PyPI). The code path is
the same except for person segmentation, which uses `hybrid` mode (see [How it works](#how-it-works)).
The `hybrid` path was benchmarked on macOS with `--seg-mode hybrid`: 30 fps live, the same torso coverage
as `gpu` mode. It has not yet run on a Windows machine.

1. Install OBS Studio. It registers the virtual camera.
2. Install Python 3.14 (x64) from python.org.
3. Create the environment in PowerShell:
   ```powershell
   py -3.14 -m venv .venv
   .venv\Scripts\pip install mediapipe==1.2.0 pyvirtualcam==0.15.0 wgpu==0.32.0 opencv-python numpy
   ```
4. Find your camera:
   ```powershell
   .venv\Scripts\python puppetcam.py --list-cams
   ```
5. Run it with profiling the first time:
   ```powershell
   .venv\Scripts\python puppetcam.py --cam 0 --preview --profile
   ```
   Check that the `frame` line reads about 33 ms. Check that `person segmentation: hybrid` is printed.

Things only a Windows run can confirm:
- **GPU speed.** On integrated graphics the puppet may render 2–4× slower than on an M2 Pro. Renders run in
  the background, so the frame rate holds but the head updates less often. If it looks steppy, try a
  smaller `--scale` or `--no-body`.
- **Calibration.** Auto-style runs the multiclass model on the CPU for each sampled frame. The first
  second after start (or `c`) runs at a low frame rate.
- **Camera.** Capture uses DirectShow. If the camera won't open at 1280×720, try another `--cam` index or
  `--width 1920 --height 1080`.

## Linux

Every dependency runs on Linux x86-64: MediaPipe and OpenCV have `manylinux` wheels, `wgpu` is the same
`py3-none` wheel (it renders on Vulkan), and `pyvirtualcam` writes to a **v4l2loopback** virtual camera.
Person segmentation uses `hybrid` mode: the Linux wheel's GPU delegate fails at EGL display creation off a
desktop session, so the CPU path is the default. The code path is written and dependency-checked but has
not yet run on a Linux machine.

The virtual camera is the one Linux-specific setup step. OBS's own virtual camera on Linux is a
v4l2loopback device too, so if you can "Start Virtual Camera" in OBS you already have the module; otherwise
load it yourself.

1. Install the virtual-camera kernel module and the Vulkan loader:
   ```bash
   sudo apt install v4l2loopback-dkms libvulkan1      # Debian/Ubuntu; use your distro's equivalents
   ```
   You also need your GPU's Vulkan driver (for example `mesa-vulkan-drivers`).
2. Load a loopback device (do this after each reboot, or configure it to load at boot):
   ```bash
   sudo modprobe v4l2loopback devices=1 card_label=puppetcam exclusive_caps=1
   ```
   `exclusive_caps=1` makes apps like Chrome and Teams list it as a normal webcam.
3. Create the environment:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install mediapipe==1.2.0 pyvirtualcam==0.15.0 wgpu==0.32.0 opencv-python numpy
   ```
   `pyvirtualcam` has no Linux wheel for some Python versions; if the install builds from source it needs a
   C compiler (`build-essential`). Its Linux backend has no other dependencies.
4. Add yourself to the `video` group if the camera won't open: `sudo usermod -aG video $USER`, then log out
   and back in.
5. Run it with profiling the first time:
   ```bash
   .venv/bin/python puppetcam.py --cam 0 --preview --profile
   ```
   Check that `person segmentation: hybrid` is printed and that `virtual camera:` names the loopback
   device.

Things only a Linux run can confirm: Vulkan render speed on your GPU, that capture opens over V4L2 at
1280×720, and that `--preview` finds a GUI (the OpenCV window needs a display; it is not needed for the
virtual-camera output itself).

## Design note

The puppet is an original character. It uses generic puppet construction (felt, a hinged mouth,
ping-pong-ball eyes), not a copy of any existing puppet character or branded product.
