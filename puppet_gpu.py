"""GPU felt puppet: raymarches puppet.wgsl with wgpu (Metal on macOS).

GpuPuppet.head(st)  -> (bgr uint8 OUT_H x S, alpha float32, nose point in output px), the same
                       contract the old 2-D draw_puppet had, so head_matrix/composite are unchanged.
GpuPuppet.body(yaw) -> (bgr, alpha, collar_x, collar_y) torso sprite at BODY_UNIT px per body unit,
                       neck included (it rises out of the collar up under the head).

Create and call it from ONE thread (the render worker); the device lives there.
"""
import math
import os

import cv2
import numpy as np
import wgpu

SHADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "puppet.wgsl")
F_HEAD = 3.0                          # camera distance in head units (mild perspective)
HINGE = np.array((0.0, -0.165, -0.06))
UP_ANG = 0.48                         # must match puppet.wgsl
NOSE = np.array((0.0, -0.025, 0.325))  # nose tip, head local
BODY_UNIT = 300                       # body sprite px per body unit (main rescales)
BODY_W, BODY_TOP, BODY_BOT = 3.2, 1.06, 2.5   # sprite: width, above / below the collar (units)


def _lin(bgr):
    return [(c / 255.0) ** 2.2 for c in bgr[::-1]]


def _rot_x(p, a):
    c, s = math.cos(a), math.sin(a)
    return np.array((p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c))


def _rot_y(p, a):
    c, s = math.cos(a), math.sin(a)
    return np.array((p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c))


class GpuPuppet:
    # ss 1.5: looks the same as 2x at output size, and its GPU load slows the person segmenter
    # (sharing the GPU) far less: segmenter 24.5 ms at 2x vs 17.0 ms at 1.5x under contention
    def __init__(self, cfg, S, padt, out_h, mouth_open, ss=1.5):
        self.cfg, self.S, self.padt, self.out_h, self.mouth_open, self.ss = cfg, S, padt, out_h, mouth_open, ss
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.dev = adapter.request_device_sync()
        with open(SHADER) as f:
            mod = self.dev.create_shader_module(code=f.read())
        self.ubuf = self.dev.create_buffer(size=16 * 11, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.pipe = self.dev.create_render_pipeline(
            layout="auto",
            vertex={"module": mod, "entry_point": "vs"},
            fragment={"module": mod, "entry_point": "fs",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list})
        self.bind = self.dev.create_bind_group(layout=self.pipe.get_bind_group_layout(0),
                                               entries=[{"binding": 0, "resource": {"buffer": self.ubuf}}])
        self.targets = {}

    def _params(self, st, mode, extra):
        cfg = self.cfg
        mouth = st.get("mouth", 0.0)
        closed = mouth < self.mouth_open
        v = np.zeros((11, 4), np.float32)
        v[0] = (st["yaw"], st["pitch"], 0.0 if closed else mouth, st.get("lid", 0.35))
        v[1] = (*st.get("gaze", (0.0, 0.0)), *st.get("brow", (0.0, 0.0)))
        v[2] = (st.get("corners", 0.0), cfg["pez"], cfg["long"], mode)
        v[3, :3] = _lin(cfg["skin"])
        v[4] = (*_lin(cfg["hair"]), 1.0 if cfg["beard_on"] else 0.0)
        v[5] = (*_lin(cfg["beard"]), 1.0 if cfg["glasses"] else 0.0)
        v[6] = (*_lin(cfg["shirt"]), 1.0 if cfg["headphones"] else 0.0)
        v[7, :2] = (cfg["shoulders"], 1.0 if closed else 0.0)
        v[7, 2:] = extra[:2]
        v[8] = extra[2:]
        v[9] = (1.0 if cfg["mustache"] else 0.0, *_lin(cfg["brow"]))
        v[10, 0] = 1.0 if cfg["full_beard"] else 0.0
        return v

    def set_cfg(self, cfg):
        """Swap the look (auto-style result); call on the render worker like every other render."""
        self.cfg = cfg

    def _draw(self, w, h, params):
        if (w, h) not in self.targets:
            self.targets[(w, h)] = self.dev.create_texture(
                size=(w, h, 1), format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        tex = self.targets[(w, h)]
        self.dev.queue.write_buffer(self.ubuf, 0, params.tobytes())
        enc = self.dev.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{
            "view": tex.create_view(), "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)}])
        rp.set_pipeline(self.pipe)
        rp.set_bind_group(0, self.bind)
        rp.draw(3)
        rp.end()
        self.dev.queue.submit([enc.finish()])
        data = self.dev.queue.read_texture({"texture": tex, "origin": (0, 0, 0), "mip_level": 0},
                                           {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
                                           (w, h, 1))
        return np.frombuffer(data, np.uint8).reshape(h, w, 4)

    @staticmethod
    def _unpremultiply(rgba, out_size):
        """Shader output is premultiplied: downsample it as uint8, then return straight BGR + alpha."""
        if (rgba.shape[1], rgba.shape[0]) != out_size:
            rgba = cv2.resize(rgba, out_size, interpolation=cv2.INTER_AREA)
        a = rgba[..., 3].astype(np.float32) * (1.0 / 255.0)
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        # interior (a == 1) is already straight; only the fuzzy fringe needs dividing
        fringe = (a > 0.0) & (a < 1.0)
        bgr[fringe] = np.clip(bgr[fringe] / a[fringe][:, None], 0, 255).astype(np.uint8)
        return bgr, a

    def nose_point(self, st):
        """Projected nose tip in output px (mirrors the shader's hinge, rotation and camera)."""
        n = NOSE.copy()
        if st.get("mouth", 0.0) >= self.mouth_open:
            n = HINGE + _rot_x(n - HINGE, -UP_ANG * st["mouth"])
        n = _rot_y(_rot_x(n, st["pitch"]), st["yaw"])
        k = F_HEAD / (F_HEAD - n[2])
        return np.array(((0.5 + n[0] * k) * self.S, (0.40 - n[1] * k) * self.S + self.padt), np.float32)

    def head(self, st):
        ss, S = self.ss, self.S
        w, h = int(S * ss), int(self.out_h * ss)      # w stays a multiple of 64 for readback rows
        fuzz = 0.007
        extra = (F_HEAD, fuzz, S * ss, 0.5 * w, (0.40 * S + self.padt) * ss, 0.78)
        rgba = self._draw(w, h, self._params(st, 0.0, extra))
        bgr, a = self._unpremultiply(rgba, (S, self.out_h))
        return bgr, a, self.nose_point(st)

    def body(self, yaw, body_size):
        u = BODY_UNIT
        w = int(math.ceil(BODY_W * u / 64.0)) * 64
        h = int((BODY_TOP + BODY_BOT) * u)
        ox, oy = w // 2, int(BODY_TOP * u)
        # camera distance in body units: same physical camera as the head (1 body unit = 0.55 * size head units)
        f = F_HEAD / (0.55 * body_size)
        st = {"yaw": yaw, "pitch": 0.0}
        extra = (f, 0.012, u, ox, oy, 3.6)
        bgr, a = self._unpremultiply(self._draw(w, h, self._params(st, 1.0, extra)), (w, h))
        return bgr, a, ox, oy
