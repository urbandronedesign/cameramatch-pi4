"""Pi4 inference server for the CameraMatch Blender add-on.

API (localhost only):
  GET  /health -> {"status": "starting|loading|ready|error", ...}
  POST /solve  {"job": str, "input": str (relative to /work/<job>), "stride": int,
                "max_frames": int, "max_size": int, "track": bool,
                "conf_threshold": float, "num_points": int}
       -> writes /work/<job>/result.npz (+ masks/) and returns a JSON summary.

Output conventions (result.npz):
  c2w          [T,4,4] camera-to-world, OpenCV camera axes (x right, y down, z forward),
               translations multiplied by Pi4's metric scale (approx. metres).
  fx, fy       [T] focal length in model pixels, principal point assumed at image centre.
  model_size   [W,H] resolution the model ran at.
  frame_indices[T] 0-based source frame index of each output frame.
  points/colors  sampled world point cloud (same frame as c2w) and uint8 RGB.
  track_grid_* coarse 2D tracks from frame 0 (for a later bundle-adjustment stage).
"""
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from progress import PROGRESS, Cancelled

WORK = os.path.realpath(os.environ.get("WORK_DIR", "/work"))
PORT = int(os.environ.get("PORT", "8765"))
MODEL_ID = os.environ.get("PI4_MODEL", "yyfz233/Pi4")
TRACK_GRID = 16  # keep one track every N pixels

_model = None
_state = {"status": "starting", "error": None}
_gpu_lock = threading.Lock()


def _load_model():
    global _model
    try:
        from pi4 import Pi4

        _state["status"] = "loading"
        t0 = time.time()
        _model = Pi4.from_pretrained(MODEL_ID).cuda()
        _state["status"] = "ready"
        print(f"[pi4] model ready in {time.time() - t0:.1f}s", flush=True)
    except Exception:
        _state["status"] = "error"
        _state["error"] = traceback.format_exc()
        print(_state["error"], flush=True)


def _job_path(job, rel=""):
    path = os.path.realpath(os.path.join(WORK, job, rel))
    if not path.startswith(WORK + os.sep):
        raise ValueError("path escapes work dir")
    return path


@torch.no_grad()
def _estimate_focal(points, conf):
    """Per-frame least-squares focal (pixels) from camera-frame pointmaps.

    Model: u - cx = fx * X/Z,  v - cy = fy * Y/Z, principal point at centre.
    Uses the most confident half of the pixels of each frame.
    """
    T, H, W, _ = points.shape
    dev = points.device
    v, u = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    u = (u.float() + 0.5 - W / 2)[None]
    v = (v.float() + 0.5 - H / 2)[None]
    Z = points[..., 2]
    valid = Z > 1e-6
    Zs = torch.where(valid, Z, torch.ones_like(Z))
    x, y = points[..., 0] / Zs, points[..., 1] / Zs
    thr = torch.quantile(conf.reshape(T, -1).float(), 0.5, dim=1).view(T, 1, 1)
    w = ((conf >= thr) & valid).float()
    fx = (w * u * x).sum((1, 2)) / (w * x * x).sum((1, 2)).clamp_min(1e-12)
    fy = (w * v * y).sum((1, 2)) / (w * y * y).sum((1, 2)).clamp_min(1e-12)
    # Residual of the pinhole fit = how "pinhole-like" the predicted rays are (pixels).
    res = torch.sqrt((u - fx.view(T, 1, 1) * x) ** 2 + (v - fy.view(T, 1, 1) * y) ** 2)
    res_med = torch.stack([res[i][w[i] > 0].median() for i in range(T)])
    return fx, fy, res_med


def solve(req):
    from pi4.io import load_images

    job = req["job"]
    src = _job_path(job, req["input"])
    out_dir = _job_path(job)
    stride = int(req.get("stride", 1))
    max_frames = int(req.get("max_frames", 300)) or None
    max_size = int(req.get("max_size", 512))
    track = bool(req.get("track", True))
    conf_thr = float(req.get("conf_threshold", 0.05))
    num_points = int(req.get("num_points", 200_000))

    t0 = time.time()
    stages = [("decode", 4), ("pi4", 40), ("post", 4)]
    if req.get("refine"):
        stages += [("klt", 22), ("ba", 26), ("write", 4)]
    PROGRESS.start(job, stages)
    plog = lambda s: (print(s, flush=True), PROGRESS.log(job, s))  # noqa: E731
    plog(f"decoding frames (stride {stride}, max {max_frames}, model size {max_size})")
    PROGRESS.update(job, "decode", 0.1, "decoding frames")
    frames, meta = load_images(src, stride=stride, max_frames=max_frames, max_size=max_size)
    T, H, W, _ = frames.shape
    plog(f"decoded {T} frames at {W}x{H}")
    images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).cuda()
    query = 0 if (track and T >= 2) else None

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    # Pi4 is one forward pass with no internal progress: estimate from the measured rate (~0.27 s/frame
    # at 512 px with tracking on RTX 5090), capped at 95% until it really finishes.
    expected = max(5.0, T * (0.27 if query is not None else 0.15) * (max_size / 512) ** 2)
    ticking = threading.Event()

    def tick():
        while not ticking.wait(0.5):
            el = time.time() - t1
            try:
                PROGRESS.update(job, "pi4", min(0.95, el / expected), f"Pi4 inference {el:.0f}s / ~{expected:.0f}s")
            except Cancelled:      # the forward pass cannot be interrupted: the job stops right after it
                return

    plog(f"Pi4 inference on {T} frames (tracking {'on' if query is not None else 'off'}), ~{expected:.0f}s")
    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        pred = _model(images, query_frame=query)
    finally:
        ticking.set()
    t_infer = time.time() - t1
    plog(f"Pi4 done in {t_infer:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    PROGRESS.update(job, "post", 0.1, "focal fit, point cloud, masks")

    P = pred["local_points"][0].float()          # T,H,W,3 camera frame, native scale
    conf = pred["confidence"][0].float()         # T,H,W
    c2w = pred["camera_poses"][0].float()        # T,4,4 native scale
    m = float(pred["metric_scale"][0])
    dyn = pred["dynamic_mask"][0].float() if "dynamic_mask" in pred else None

    fx, fy, res = _estimate_focal(P, conf)

    # World point cloud (metric): static pixels passing upstream's geometry confidence
    # (zeroes invalid pixels and depth edges), sampled evenly per frame so short-lived
    # parts of the shot keep their geometry. Same formula as upstream: m * (R P + t).
    from pi4.postprocess import geometry_confidence

    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    world = (torch.einsum("tij,thwj->thwi", R, P) + t[:, None, None, :]) * m
    gconf = torch.stack([geometry_confidence(P[i], conf[i]) for i in range(T)])
    keep = gconf >= conf_thr
    if dyn is not None:
        keep &= dyn < 0.5
    per_frame = max(1, num_points // T)
    gen = torch.Generator(device=P.device).manual_seed(0)
    chunks = []
    for i in range(T):
        fi = keep[i].reshape(-1).nonzero().squeeze(1)
        if fi.numel() > per_frame:
            fi = fi[torch.randperm(fi.numel(), device=fi.device, generator=gen)[:per_frame]]
        chunks.append(fi + i * H * W)
    idx = torch.cat(chunks)
    points = world.reshape(-1, 3)[idx].cpu().numpy().astype(np.float32)
    colors = frames.reshape(-1, 3)[idx.cpu().numpy()]
    # Radius = footprint of one model pixel at that depth (metres).
    depth_m = (P[..., 2] * m).reshape(-1)[idx]
    fpx = fx.clamp_min(1.0).repeat_interleave(H * W)[idx]
    radii = (depth_m / fpx).cpu().numpy().astype(np.float32)

    c2w_m = c2w.clone()
    c2w_m[:, :3, 3] *= m

    out = dict(
        c2w=c2w_m.cpu().numpy().astype(np.float64),
        fx=fx.cpu().numpy(), fy=fy.cpu().numpy(),
        pinhole_residual_px=res.cpu().numpy(),
        model_size=np.array([W, H]),
        source_size=np.array(meta.get("image_size", [W, H])),
        frame_indices=np.array(meta.get("source_frame_indices", list(range(0, T * stride, stride)))),
        metric_scale=np.array(m),
        points=points, colors=colors, radii=radii,
    )
    if "tracks_2d" in pred:
        g = TRACK_GRID
        out["track_grid_uv"] = pred["tracks_2d"][0, :, g // 2::g, g // 2::g].cpu().numpy().astype(np.float32)
        out["track_grid_vis"] = pred["visibility"][0, :, g // 2::g, g // 2::g].cpu().numpy().astype(np.float16)
        out["track_query_frame"] = np.array(query)
    if dyn is not None:
        mask_dir = os.path.join(out_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        dyn_u8 = (dyn.clamp(0, 1) * 255).byte().cpu().numpy()
        for i, fi in enumerate(out["frame_indices"]):
            Image.fromarray(dyn_u8[i]).save(os.path.join(mask_dir, f"mask_{int(fi) + 1:05d}.png"))

    np.savez(os.path.join(out_dir, "result.npz"), **out)
    # Cache for plate-resolution refinement (re-runnable via /refine without Pi4).
    dyn_cache = (dyn.clamp(0, 1) * 255).byte().cpu().numpy() if dyn is not None else np.zeros((T, H, W), np.uint8)
    np.savez(os.path.join(out_dir, "pi4_cache.npz"), depth=(P[..., 2] * m).half().cpu().numpy(), dyn=dyn_cache,
             src=np.array(req["input"]))
    del pred, P, conf, gconf, world, images
    torch.cuda.empty_cache()

    summary = dict(
        ok=True, frames=int(T), model_size=[int(W), int(H)], metric_scale=m,
        fx_median=float(fx.median()), fx_std=float(fx.std()) if T > 1 else 0.0,
        pinhole_residual_px=float(res.median()),
        points=int(points.shape[0]), tracking=query is not None,
        inference_s=round(t_infer, 2), total_s=round(time.time() - t0, 2),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
    )
    plog(f"Pi4 stage written: f={summary['fx_median']:.1f}px (model res), {summary['points']} points")
    if req.get("refine"):
        summary["refine"] = run_refine(req, started=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _ignore_classes(req):
    """Request field ignore_classes: list of ADE20K names, or true / "default" for semantic.DEFAULT_CLASSES."""
    v = req.get("ignore_classes")
    if not v:
        return None
    if v is True or v == "default":
        import semantic
        return list(semantic.DEFAULT_CLASSES)
    return list(v)


# Pi4's per-frame focal varies < 4 % (smoothed 5-95 % range) on every fixed-lens test shot and 46 % on the
# synthetic 1.6x zoom: 8 % separates them with margin.
ZOOM_AUTO_RANGE = 0.08


def detect_zoom(fx):
    """Zoom shot? From Pi4's per-frame focal (model pixels). Returns (is_zoom, relative smoothed range)."""
    fx = np.asarray(fx, np.float64).reshape(-1)
    if len(fx) < 12:
        return False, 0.0
    s = np.convolve(fx, np.ones(9) / 9, "valid")
    rng = float((np.percentile(s, 95) - np.percentile(s, 5)) / np.median(fx))
    return rng > ZOOM_AUTO_RANGE, rng


def run_refine(req, started=False):
    """Plate-resolution KLT + bundle adjustment on top of an existing Pi4 result."""
    import refine as rf

    job = req["job"]
    if not started:
        PROGRESS.start(job, [("klt", 45), ("ba", 50), ("write", 5)])
    plog = lambda s: (print(s, flush=True), PROGRESS.log(job, s))  # noqa: E731
    out_dir = _job_path(job)
    res_path = os.path.join(out_dir, "result.npz")
    result = dict(np.load(res_path))
    with np.load(os.path.join(out_dir, "pi4_cache.npz")) as z:
        cache = {k: z[k] for k in z.files}
    src = _job_path(job, str(cache["src"]))
    t0 = time.time()
    zoom = req.get("zoom", False)
    zoom_auto = None
    if zoom == "auto":
        zoom, zr = detect_zoom(result["fx"])
        zoom_auto = dict(zoom=bool(zoom), focal_range=round(zr, 4))
        plog(f"zoom auto-detect: Pi4 focal varies {zr * 100:.1f}% over the shot -> "
             f"{'zoom lens (per-frame focal)' if zoom else 'fixed focal'}")
    out, stats = rf.refine(src, result, cache, log=plog,
                           progress=lambda st, fr, msg=None: PROGRESS.update(job, st, fr, msg),
                           max_active=int(req.get("max_tracks", 600)),
                           prior_frac=float(req.get("prior_frac", 0.1)),
                           quality=float(req.get("klt_quality", 0.001)),
                           focal_sigma=float(req.get("focal_sigma", 0.3)),
                           solve_distortion=bool(req.get("solve_distortion", True)),
                           solver=str(req.get("solver", "lm")), zoom=bool(zoom),
                           zoom_smooth=float(req.get("zoom_smooth", 0.01)),
                           rolling_shutter=bool(req.get("rolling_shutter", False)),
                           rs_axis=str(req.get("rs_axis", "auto")),
                           ignore_classes=_ignore_classes(req), job_dir=out_dir)
    PROGRESS.update(job, "write", 0.5, "writing result")
    result.update(out)
    np.savez(res_path, **result)
    stats["refine_s"] = round(time.time() - t0, 2)
    if zoom_auto is not None:
        stats["zoom_auto"] = zoom_auto
    stats["plate_portrait"] = bool(stats.get("plate", [1, 0])[1] > stats.get("plate", [1, 0])[0])
    with open(os.path.join(out_dir, "refine_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    plog(f"refinement done in {stats['refine_s']}s")
    return stats


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/progress"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            return self._send(200, PROGRESS.get(q.get("job", [""])[0], int(q.get("since", ["0"])[0])))
        if self.path.startswith("/cancel"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            return self._send(200, {"ok": PROGRESS.cancel(q.get("job", [""])[0])})
        if self.path == "/health":
            info = dict(_state)
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["torch"] = torch.__version__
            info["busy"] = _gpu_lock.locked()
            self._send(200, info)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/solve", "/refine"):
            return self._send(404, {"error": "not found"})
        if self.path == "/solve" and _state["status"] != "ready":
            return self._send(503, {"ok": False, "error": f"model {_state['status']}", "detail": _state["error"]})
        req = {}
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with _gpu_lock:
                result = solve(req) if self.path == "/solve" else {"ok": True, "refine": run_refine(req)}
            PROGRESS.finish(req.get("job", ""), ok=True)
            self._send(200, result)
        except Cancelled:
            torch.cuda.empty_cache()
            PROGRESS.log(req.get("job", ""), "cancelled by the user")
            PROGRESS.finish(req.get("job", ""), ok=False, error="cancelled")
            self._send(200, {"ok": False, "cancelled": True, "error": "cancelled"})
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            msg = "GPU out of memory: raise stride, lower max frames/size, or disable motion"
            PROGRESS.log(req.get("job", ""), "ERROR " + msg)
            PROGRESS.finish(req.get("job", ""), ok=False, error=msg)
            self._send(500, {"ok": False, "error": msg})
        except Exception as e:
            traceback.print_exc()
            PROGRESS.log(req.get("job", ""), f"ERROR {type(e).__name__}: {e}")
            PROGRESS.finish(req.get("job", ""), ok=False, error=f"{type(e).__name__}: {e}")
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}", "detail": traceback.format_exc()})

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args, flush=True)


if __name__ == "__main__":
    threading.Thread(target=_load_model, daemon=True).start()
    print(f"[pi4] listening on :{PORT}, work dir {WORK}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
