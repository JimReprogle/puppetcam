// puppetcam GPU puppet: raymarched signed-distance felt puppet (head or torso per draw).
//
// World frame: x right, y UP, z toward the camera, 1 unit = 1 head-canvas width (S px) for the
// head and 1 body unit (u px) for the torso. Perspective camera at (0, 0, F) looking down -z; the
// z = 0 plane maps 1:1 onto the output canvas, so 2-D placement code (head_matrix) is unchanged.
// Felt, not plastic: no specular on fabric, wrap diffuse, grazing sheen, fiber albedo/normal noise,
// AO, soft shadows, and a fuzzy fiber halo on the silhouette instead of a hard edge.

struct Params { v: array<vec4f, 10> };
@group(0) @binding(0) var<uniform> P: Params;

// decoded parameters (set once per fragment in fs)
var<private> g_yaw: f32;
var<private> g_pitch: f32;
var<private> g_mouth: f32;      // 0 when below the MOUTH_OPEN threshold (closed line instead)
var<private> g_lid: f32;
var<private> g_gaze: vec2f;
var<private> g_brow: vec2f;     // (up, down)
var<private> g_corners: f32;
var<private> g_pez: f32;
var<private> g_long: f32;
var<private> g_mode: f32;       // 0 head, 1 body
var<private> g_beard: f32;
var<private> g_glasses: f32;
var<private> g_phones: f32;
var<private> g_shoulders: f32;
var<private> g_closed: f32;     // 1 when the mouth is a closed line

const M_SKIN = 0.0;
const M_HAIR = 1.0;
const M_BEARD = 2.0;
const M_EYE = 3.0;
const M_LID = 4.0;
const M_GLASS = 5.0;
const M_PHONE = 6.0;
const M_PAD = 7.0;
const M_MOUTH = 8.0;
const M_TONGUE = 9.0;
const M_BROW = 10.0;
const M_SHIRT = 11.0;
const M_BUTTON = 12.0;
const M_WEB = 13.0;
const M_NECK = 14.0;

const HINGE = vec3f(0.0, -0.165, -0.06);
const UP_ANG = 0.38;
const JAW_ANG = 0.14;
const EYE_R = 0.056;
const EYE_C = vec3f(0.082, 0.07, 0.175);

// ------------------------------------------------------------------ helpers

fn rotX(p: vec3f, a: f32) -> vec3f {
    let c = cos(a);
    let s = sin(a);
    return vec3f(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

fn rotY(p: vec3f, a: f32) -> vec3f {
    let c = cos(a);
    let s = sin(a);
    return vec3f(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

fn rotZ(p: vec3f, a: f32) -> vec3f {
    let c = cos(a);
    let s = sin(a);
    return vec3f(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn sdSphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

fn sdRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
    let q = abs(p) - b + vec3f(r);
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn sdRoundRect(p: vec2f, b: vec2f, r: f32) -> f32 {
    let q = abs(p) - b + vec2f(r);
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// rounded cylinder along x: radius ra, half length hl, edge rounding rb
fn sdCylX(p: vec3f, ra: f32, hl: f32, rb: f32) -> f32 {
    let d = vec2f(length(p.yz) - ra + rb, abs(p.x) - hl + rb);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0))) - rb;
}

// capped cylinder along y between y0 and y1
fn sdCylY(p: vec3f, r: f32, y0: f32, y1: f32) -> f32 {
    let c = 0.5 * (y0 + y1);
    let h = 0.5 * (y1 - y0);
    let d = vec2f(length(p.xz) - r, abs(p.y - c) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn U(a: vec2f, b: vec2f) -> vec2f {
    return select(b, a, a.x < b.x);
}

fn hash3(p: vec3f) -> f32 {
    let q = fract(p * 0.3183099 + vec3f(0.11, 0.17, 0.13)) * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn vnoise(x: vec3f) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let w = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i + vec3f(0.0, 0.0, 0.0)), hash3(i + vec3f(1.0, 0.0, 0.0)), w.x),
                   mix(hash3(i + vec3f(0.0, 1.0, 0.0)), hash3(i + vec3f(1.0, 1.0, 0.0)), w.x), w.y),
               mix(mix(hash3(i + vec3f(0.0, 0.0, 1.0)), hash3(i + vec3f(1.0, 0.0, 1.0)), w.x),
                   mix(hash3(i + vec3f(0.0, 1.0, 1.0)), hash3(i + vec3f(1.0, 1.0, 1.0)), w.x), w.y), w.z);
}

// ------------------------------------------------------------------ head (local frame, mouth closed)

fn headBase(q: vec3f) -> f32 {
    let hy = 0.31 + 0.06 * g_long;
    var c = q - vec3f(0.0, -0.03, 0.0);
    let t = clamp(-c.y / hy, -1.0, 1.0);
    c.x = c.x / (1.0 + 0.07 * t);                 // slightly wider at the jaw
    let e = sdEllipsoid(c, vec3f(0.25, hy, 0.21));
    let b = sdRoundBox(c, vec3f(0.245, hy, 0.205), 0.13);
    return mix(e, b, g_pez) * 0.93;
}

fn hairAmt(q: vec3f) -> f32 {
    // full head of hair: sides and back down to the nape, front hairline where the forehead turns
    // into the crown, with only slight recession at the temples
    let front = smoothstep(-0.02, 0.17, q.z);
    let hl = mix(-0.14, 0.245 - 1.1 * q.x * q.x, front);
    return smoothstep(-0.008, 0.008, q.y - hl);
}

// sculpted felt head: dome, brow ridge, cheeks, chin, nose, nostrils, raised hair cap
fn skull(q: vec3f) -> vec2f {
    var d = headBase(q);
    d = smin(d, sdEllipsoid(q - vec3f(0.0, 0.135, 0.165), vec3f(0.20, 0.045, 0.06)), 0.04);
    let ax = vec3f(abs(q.x), q.y, q.z);
    d = smin(d, sdSphere(ax - vec3f(0.13, -0.10, 0.14), 0.085), 0.05);
    d = smin(d, sdEllipsoid(q - vec3f(0.0, -0.31, 0.12), vec3f(0.11, 0.08, 0.09)), 0.05);
    d = smin(d, sdEllipsoid(q - vec3f(0.0, -0.025, 0.24), vec3f(0.08, 0.09, 0.085)), 0.03);
    d = max(d, -sdSphere(ax - vec3f(0.032, -0.1, 0.275), 0.015));
    let h = hairAmt(q);
    if (h > 0.0) {
        d -= h * (0.014 + 0.006 * vnoise(q * vec3f(70.0, 22.0, 70.0)));
    }
    return vec2f(d, select(M_SKIN, M_HAIR, h > 0.5));
}

// parts that ride on the upper head (never cut by the hinge)
fn upperParts(q: vec3f) -> vec2f {
    let ax = vec3f(abs(q.x), q.y, q.z);
    // ping-pong eyes
    var r = vec2f(sdSphere(ax - EYE_C, EYE_R), M_EYE);
    // heavy upper lids: a felt shell over the top of each eye, edge height set by lid
    let edge = EYE_C.y + (1.0 - 2.0 * g_lid) * EYE_R;
    let shell = abs(sdSphere(ax - EYE_C, EYE_R + 0.004)) - 0.0035;
    r = U(r, vec2f(max(shell, edge - q.y), M_LID));
    // brows: fur capsules, raised by brow-up, inner ends pulled down by brow-down
    let by = 0.158 + 0.03 * g_brow.x;
    let bi = by - 0.035 * g_brow.y;
    var brow = sdCapsule(ax, vec3f(0.035, bi, 0.232), vec3f(0.145, by + 0.004, 0.203), 0.019);
    brow -= 0.006 * vnoise(q * vec3f(25.0, 90.0, 60.0));
    r = U(r, vec2f(brow, M_BROW));
    // mustache: two drooping lobes under the nose
    var mu = smin(sdCapsule(ax, vec3f(0.008, -0.122, 0.262), vec3f(0.112, -0.162, 0.205), 0.028),
                  sdSphere(ax - vec3f(0.0, -0.118, 0.255), 0.03), 0.02);
    mu -= 0.008 * vnoise(q * vec3f(90.0, 20.0, 90.0));
    r = U(r, vec2f(mu, M_BEARD));
    if (g_glasses > 0.5) {
        let lc = ax - vec3f(0.086, 0.07, 0.285);
        let ring = abs(sdRoundRect(lc.xy, vec2f(0.068, 0.047), 0.016)) - 0.0075;
        let w = vec2f(ring, abs(lc.z) - 0.0065);
        var gl = min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0)));
        gl = min(gl, sdCapsule(ax, vec3f(0.0, 0.088, 0.29), vec3f(0.02, 0.088, 0.287), 0.0065));
        gl = min(gl, sdCapsule(ax, vec3f(0.152, 0.082, 0.284), vec3f(0.262, 0.085, 0.04), 0.0062));
        r = U(r, vec2f(gl, M_GLASS));
    }
    if (g_phones > 0.5) {
        let c = q - vec3f(0.0, -0.03, -0.01);
        var band = sdRoundRect(vec2f(length(c.xy) - 0.305, c.z), vec2f(0.013, 0.036), 0.011);
        band = max(band, 0.07 - c.y);
        band = min(band, sdCapsule(ax, vec3f(0.297, 0.04, -0.01), vec3f(0.30, -0.02, -0.01), 0.015));
        let cup = sdCylX(ax - vec3f(0.30, -0.075, -0.005), 0.102, 0.042, 0.022);
        r = U(r, vec2f(min(band, cup), M_PHONE));
        r = U(r, vec2f(sdCylX(ax - vec3f(0.26, -0.075, -0.005), 0.09, 0.022, 0.018), M_PAD));
    }
    return r;
}

fn lowerParts(q: vec3f) -> vec2f {
    var r = vec2f(1e3, M_BEARD);
    if (g_beard > 0.5) {
        var c = q - vec3f(0.0, -0.36, 0.172);
        let t = clamp((-0.25 - q.y) / 0.32, 0.0, 1.0);
        let k = 1.0 - 0.7 * t;                      // tapers to a point
        c.x = c.x / k;
        var gt = sdEllipsoid(c, vec3f(0.078, 0.195, 0.056)) * k;
        gt = smin(gt, sdEllipsoid(q - vec3f(0.0, -0.215, 0.215), vec3f(0.03, 0.03, 0.02)), 0.02);
        gt -= 0.009 * vnoise(q * vec3f(85.0, 9.0, 85.0));
        r = vec2f(gt, M_BEARD);
    }
    // tongue on the jaw floor (hidden inside the head while closed)
    r = U(r, vec2f(sdEllipsoid(q - vec3f(0.0, HINGE.y + 0.012, 0.05), vec3f(0.10, 0.03, 0.11)), M_TONGUE));
    return r;
}

fn headMap(pw: vec3f) -> vec2f {
    // world -> head local (inverse of yaw then pitch)
    let q = rotX(rotY(pw, -g_yaw), -g_pitch);
    // upper part: rotated up about the hinge; jaw: rotated down
    let qu = HINGE + rotX(q - HINGE, UP_ANG * g_mouth);
    let ql = HINGE + rotX(q - HINGE, -JAW_ANG * g_mouth);
    let su = skull(qu);
    let cutU = HINGE.y - qu.y;
    var r = vec2f(max(su.x, cutU), select(su.y, M_MOUTH, cutU > su.x));
    r = U(r, upperParts(qu));
    let sl = skull(ql);
    let cutL = ql.y - HINGE.y;
    r = U(r, vec2f(max(sl.x, cutL), select(sl.y, M_MOUTH, cutL > sl.x)));
    r = U(r, lowerParts(ql));
    if (g_mouth > 0.0) {
        // mouth sack between palate and jaw; its outer sides are the cheek web
        let sk = vec3f(0.0, HINGE.y + 0.03 * g_mouth, -0.02);
        // as wide as the head: narrower and the open mouth is see-through from an angle
        let sack = sdEllipsoid(q - sk, vec3f(0.258, 0.012 + 0.075 * g_mouth, 0.19));
        r = U(r, vec2f(sack, select(M_MOUTH, M_WEB, abs(q.x) > 0.17)));
    }
    return r;
}

// ------------------------------------------------------------------ torso (body units, collar at origin)

fn bodyMap(pw: vec3f) -> vec2f {
    let p = rotY(pw, -g_yaw);
    let sh = g_shoulders / 0.8;
    let ax = vec3f(abs(p.x), p.y, p.z);
    var c = p;
    c.y += 0.2 * smoothstep(0.2, 1.0, ax.x);       // sloping shoulders
    var d = sdRoundBox(c - vec3f(0.0, -1.32, -0.02), vec3f(0.92 * sh, 1.25, 0.30), 0.30);
    // rounded chest so the front isn't a flat slab (kept below the shoulder line)
    d = smin(d, sdEllipsoid(p - vec3f(0.0, -1.0, 0.0), vec3f(0.8 * sh, 0.8, 0.40)), 0.12);
    d = smin(d, sdCapsule(ax, vec3f(0.84 * sh, -0.48, -0.02), vec3f(0.97 * sh, -2.6, 0.04), 0.2), 0.05);
    d = max(d, -sdCylY(p - vec3f(0.0, 0.0, 0.03), 0.2, -0.5, 1.0));    // neck opening
    var r = vec2f(d, M_SHIRT);
    // collar band standing up around the back of the neck
    let cb = p - vec3f(0.0, 0.0, 0.02);
    var band = sdRoundRect(vec2f(length(cb.xz) - 0.225, cb.y - 0.02), vec2f(0.024, 0.075), 0.016);
    band = max(band, cb.z - 0.1);
    r = U(r, vec2f(band, M_SHIRT));
    // collar points lying on the chest
    for (var s = -1.0; s <= 1.0; s += 2.0) {
        var f = p - vec3f(0.15 * s, -0.09, 0.27);
        f = rotX(rotZ(f, -0.6 * s), 0.45);
        r = U(r, vec2f(sdRoundBox(f, vec3f(0.15, 0.095, 0.014), 0.012), M_SHIRT));
    }
    // placket + buttons
    r = U(r, vec2f(sdRoundBox(p - vec3f(0.0, -0.4, 0.375), vec3f(0.06, 0.24, 0.014), 0.008), M_SHIRT));
    for (var i = 0; i < 3; i++) {
        let y = -0.26 - 0.15 * f32(i);
        r = U(r, vec2f(sdEllipsoid(p - vec3f(0.0, y, 0.392), vec3f(0.026, 0.026, 0.012)), M_BUTTON));
    }
    // neck, rising out of the collar up under the head
    r = U(r, vec2f(sdCylY(p - vec3f(0.0, 0.0, 0.03), 0.16, -0.4, 1.0) - 0.01, M_NECK));
    return r;
}

fn map(p: vec3f) -> vec2f {
    if (g_mode < 0.5) {
        return headMap(p);
    }
    return bodyMap(p);
}

fn normalAt(p: vec3f) -> vec3f {
    let e = vec2f(1.0, -1.0) * 0.0012;
    return normalize(e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x +
                     e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x);
}

fn softShadow(ro: vec3f, rd: vec3f) -> f32 {
    var res = 1.0;
    var t = 0.006;
    for (var i = 0; i < 28; i++) {
        let h = map(ro + rd * t).x;
        res = min(res, 10.0 * h / t);
        t += clamp(h, 0.006, 0.06);
        if (res < 0.02 || t > 0.8) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

fn ambientOcc(p: vec3f, n: vec3f) -> f32 {
    var occ = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 5; i++) {
        let h = 0.012 * f32(i);
        occ += (h - map(p + n * h).x) * w;
        w *= 0.7;
    }
    return clamp(1.0 - 4.5 * occ, 0.0, 1.0);
}

fn srgb2lin(c: vec3f) -> vec3f {
    return pow(c, vec3f(2.2));
}

fn albedo(mat: f32) -> vec3f {
    if (mat == M_SKIN || mat == M_LID || mat == M_NECK) {
        return P.v[3].xyz;
    }
    if (mat == M_HAIR) {
        return P.v[4].xyz;
    }
    if (mat == M_BEARD || mat == M_BROW) {
        return P.v[5].xyz;
    }
    if (mat == M_EYE) {
        return vec3f(0.85, 0.85, 0.83);
    }
    if (mat == M_GLASS) {
        return vec3f(0.035, 0.035, 0.04);
    }
    if (mat == M_PHONE) {
        return vec3f(0.03, 0.03, 0.033);
    }
    if (mat == M_PAD) {
        return vec3f(0.05, 0.05, 0.055);
    }
    if (mat == M_MOUTH) {
        return vec3f(0.16, 0.012, 0.02);
    }
    if (mat == M_TONGUE) {
        return vec3f(0.62, 0.17, 0.2);
    }
    if (mat == M_WEB) {
        return P.v[3].xyz * 0.55;
    }
    if (mat == M_BUTTON) {
        return vec3f(0.8, 0.8, 0.78);
    }
    return P.v[6].xyz;                                // shirt
}

fn isFelt(mat: f32) -> bool {
    return !(mat == M_EYE || mat == M_GLASS || mat == M_PHONE || mat == M_BUTTON);
}

fn shade(p: vec3f, rd: vec3f, mat: f32) -> vec3f {
    var n = normalAt(p);
    let v = -rd;
    var q = p;                                        // pattern space rides with the object
    if (g_mode < 0.5) {
        q = rotX(rotY(p, -g_yaw), -g_pitch);
    } else {
        q = rotY(p, -g_yaw) * 0.6;
    }
    var alb = albedo(mat);
    let felt = isFelt(mat);
    if (felt) {
        // fleece: fine pile + coarse mottle in color, fiber bumps in the normal
        let fq = q * 150.0;
        alb *= 0.82 + 0.22 * vnoise(fq) + 0.12 * vnoise(q * 38.0);
        let b = vec3f(vnoise(fq + vec3f(7.1, 0.0, 0.0)), vnoise(fq + vec3f(0.0, 3.7, 0.0)),
                      vnoise(fq + vec3f(0.0, 0.0, 5.3))) - vec3f(0.5);
        n = normalize(n + 0.45 * b);
        if (mat == M_HAIR || mat == M_BEARD || mat == M_BROW) {
            // strands: streaky along the growth direction
            alb *= 0.65 + 0.6 * vnoise(q * vec3f(260.0, 40.0, 260.0));
        }
    }
    if (mat == M_EYE) {
        let side = select(-1.0, 1.0, q.x > 0.0);
        let qu = HINGE + rotX(q - HINGE, UP_ANG * g_mouth);
        let dir = normalize(qu - vec3f(EYE_C.x * side, EYE_C.y, EYE_C.z));
        let gz = normalize(vec3f(g_gaze.x * 0.45, -g_gaze.y * 0.35, 1.0));
        let k = dot(dir, gz);
        alb = mix(alb, vec3f(0.012), smoothstep(0.935, 0.945, k));
    }
    if (mat == M_SKIN && g_closed > 0.5 && g_mode < 0.5) {
        // closed mouth: a stitched line along the hinge, corners bent by smile/frown
        let line = HINGE.y + g_corners * 0.035 * (q.x * q.x) / (0.12 * 0.12);
        let lm = (1.0 - smoothstep(0.0025, 0.0055, abs(q.y - line))) *
                 (1.0 - smoothstep(0.10, 0.12, abs(q.x))) * smoothstep(0.12, 0.17, q.z);
        alb *= 1.0 - 0.8 * lm;
    }
    let L = normalize(vec3f(-0.35, 0.55, 0.76));
    let diff = max((dot(n, L) + 0.3) / 1.3, 0.0);
    let sh = softShadow(p + n * 0.004, L);
    let ao = ambientOcc(p, n);
    let sky = 0.55 + 0.45 * n.y;
    // exposure: a front-lit face lands near its albedo (brighter clips skin red = "hot plastic")
    var col = alb * (0.88 * diff * mix(0.35, 1.0, sh) * vec3f(1.0, 0.97, 0.92) +
                     0.3 * ao * sky * vec3f(0.85, 0.9, 1.0) + 0.05 * ao);
    let ndv = max(dot(n, v), 0.0);
    if (felt) {
        // velvet sheen: fibers catch light at grazing angles; no mirror highlight
        col += pow(1.0 - ndv, 2.5) * 0.4 * (alb * 0.7 + vec3f(0.05)) * ao;
    } else {
        let sp = pow(max(dot(reflect(-L, n), v), 0.0), 36.0);
        let ks = select(0.18, 0.55, mat == M_EYE);
        col += ks * sp * sh;
    }
    return col;
}

// ------------------------------------------------------------------ entry points

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
    g_yaw = P.v[0].x;
    g_pitch = P.v[0].y;
    g_mouth = P.v[0].z;
    g_lid = P.v[0].w;
    g_gaze = P.v[1].xy;
    g_brow = P.v[1].zw;
    g_corners = P.v[2].x;
    g_pez = P.v[2].y;
    g_long = P.v[2].z;
    g_mode = P.v[2].w;
    g_beard = P.v[4].w;
    g_glasses = P.v[5].w;
    g_phones = P.v[6].w;
    g_shoulders = P.v[7].x;
    g_closed = P.v[7].y;
    let F = P.v[7].z;
    let fuzz = P.v[7].w;
    // pixel -> z=0 plane: v[8] = (px per unit, origin x px, origin y px, bound radius)
    let ppu = P.v[8].x;
    let xy = vec2f((fc.x - P.v[8].y) / ppu, -(fc.y - P.v[8].z) / ppu);
    let ro = vec3f(0.0, 0.0, F);
    let rd = normalize(vec3f(xy, -F));
    // bounding sphere around the origin
    let br = P.v[8].w;
    let bb = dot(ro, rd);
    let disc = bb * bb - dot(ro, ro) + br * br;
    if (disc < 0.0) {
        return vec4f(0.0);
    }
    var t = max(-bb - sqrt(disc), 0.0);
    let tEnd = -bb + sqrt(disc);
    var minD = 1e9;
    var tMin = t;
    var hit = false;
    for (var i = 0; i < 160; i++) {
        let h = map(ro + rd * t).x;
        if (h < minD) {
            minD = h;
            tMin = t;
        }
        if (h < 0.0003) {
            hit = true;
            break;
        }
        t += h * 0.75;
        if (t > tEnd) {
            break;
        }
    }
    if (hit) {
        let p = ro + rd * t;
        let col = shade(p, rd, map(p).y);
        return vec4f(pow(clamp(col, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)), 1.0);
    }
    // near miss: fleece fibers standing off the surface make a soft, broken halo
    if (minD < fuzz) {
        let p = ro + rd * tMin;
        let mat = map(p).y;
        if (!isFelt(mat)) {
            return vec4f(0.0);
        }
        let n = vnoise(p * 420.0) * 0.7 + vnoise(p * 140.0) * 0.5;
        let a = clamp((1.0 - minD / fuzz) * n * 1.2, 0.0, 1.0);
        let col = shade(p, rd, mat);
        return vec4f(pow(clamp(col, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)) * a, a);
    }
    return vec4f(0.0);
}
