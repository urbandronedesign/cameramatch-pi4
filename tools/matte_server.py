"""Matte server for the CameraMatch add-on (SAM 2.1 + MatAnyone 2), localhost only.

GET  /health -> {"status": "ready", "busy": bool, "gpu": ...}
POST /matte  {"job": str, "anchor": 30, "chunk": 30}
     -> writes /work/<job>/matte/fused/alpha_####.png and /work/<job>/occlusion/{invdepth,fgmask}_####.png
        (numbered source frame + 1), returns the fused matte's metrics.
POST /set    {"job": str, "tris": 60000, "voxel": 0.04, "rebuild": false}
     -> low-poly set mesh (lowpoly.npz) + plate environment (plate_env.png) for the add-on's Build Set.
POST /comp   {"job": str, "render_dir": "render", "out_dir": "comp", "soft": 0.6, "wrap": 0.25, "grain": "auto"}
     -> plate x shadow-catcher pass + matched CG (comp_plate.py), comp_####.png + <clip>_comp.mp4 + plate_vs_comp.mp4.
MatAnyone 2: NTU S-Lab License 1.0 (non-commercial). SAM 2.1: Apache-2.0.
"""
import glob
import json
import os
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import comp_plate
import matte_pipeline as mp
from progress import PROGRESS

WORK = os.path.realpath(os.environ.get("WORK_DIR", "/work"))
PORT = int(os.environ.get("PORT", "8766"))
_lock = threading.Lock()


def _job_dir(job):
    p = os.path.realpath(os.path.join(WORK, job))
    if not p.startswith(WORK + os.sep) or not os.path.isfile(os.path.join(p, "pi4_cache.npz")):
        raise ValueError(f"unknown job or missing pi4_cache.npz: {job}")
    return p


def _plate_input(job):
    """The job's source plate: input/<movie> or input/seq/%06d.<ext> (as the add-on copied it)."""
    ind = os.path.join(job, "input")
    seq = os.path.join(ind, "seq")
    if os.path.isdir(seq):
        f = sorted(os.listdir(seq))[0]
        return os.path.join(seq, "%06d" + os.path.splitext(f)[1]), os.path.splitext(f)[0]
    movies = [f for f in sorted(os.listdir(ind)) if os.path.isfile(os.path.join(ind, f))]
    if not movies:
        raise ValueError(f"no plate in {ind}")
    return os.path.join(ind, movies[0]), os.path.splitext(movies[0])[0]


def _run_tool(name, args, stage, log):
    """Run a /tools script as a subprocess, streaming its stdout into the job log."""
    proc = subprocess.Popen(["python", "-u", os.path.join(os.path.dirname(__file__), name)] + args,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    for line in proc.stdout:
        line = line.rstrip()
        if line and "WARN" not in line:
            log(line)
    if proc.wait() != 0:
        raise RuntimeError(f"{name} failed (exit {proc.returncode}), see log")
    log(f"{name} done in {time.time() - t0:.0f}s")


def _do_set(name, req):
    job = _job_dir(name)
    log = lambda s: PROGRESS.log(name, s)  # noqa: E731
    rebuild = bool(req.get("rebuild", False))
    if rebuild or not os.path.isfile(os.path.join(job, "lowpoly.npz")):
        PROGRESS.update(name, "mesh", 0.05, "fusing Pi4 depth, Poisson surface, decimation")
        _run_tool("build_lowpoly_mesh.py", [job, "--tris", str(int(req.get("tris", 60000))),
                                            "--voxel", str(float(req.get("voxel", 0.04))), "--max_depth", "35"],
                  "mesh", log)
    else:
        log("lowpoly.npz exists, reusing it (tick Rebuild to recompute)")
    PROGRESS.update(name, "mesh", 1.0)
    if rebuild or not os.path.isfile(os.path.join(job, "plate_env.png")):
        PROGRESS.update(name, "envmap", 0.1, "plate environment map")
        _run_tool("build_plate_envmap.py", [job], "envmap", log)
    else:
        log("plate_env.png exists, reusing it")
    PROGRESS.update(name, "envmap", 1.0)
    return dict(lowpoly="lowpoly.npz", env="plate_env.png")


def _do_comp(name, req):
    job = _job_dir(name)
    log = lambda s: PROGRESS.log(name, s)  # noqa: E731
    plate, stem = _plate_input(job)
    rdir = os.path.join(job, req.get("render_dir", "render"))
    odir = os.path.join(job, req.get("out_dir", "comp"))
    comp_plate.LOG_HOOK = log
    comp_plate.PROGRESS_HOOK = lambda fr, msg: PROGRESS.update(name, "comp", fr, msg)
    try:
        done = comp_plate.run(plate, rdir, odir, frames=req.get("frames"), grain=req.get("grain", "auto"),
                              soft=float(req.get("soft", 0.6)), wrap=float(req.get("wrap", 0.25)))
    finally:
        comp_plate.LOG_HOOK = comp_plate.PROGRESS_HOOK = None
    fps = float(req.get("fps", 25.0))
    out_mp4 = os.path.join(odir, f"{stem}_comp.mp4")
    sbs_mp4 = os.path.join(odir, f"{stem}_plate_vs_comp.mp4")
    PROGRESS.update(name, "encode", 0.1, "encoding H.264")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-start_number", str(done[0]),
                    "-i", os.path.join(odir, "comp_%04d.png"), "-frames:v", str(len(done)), "-c:v", "libx264",
                    "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", out_mp4], check=True)
    PROGRESS.update(name, "encode", 0.6, "encoding plate-vs-comp")
    seek = (done[0] - 1) / fps
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.4f}", "-i", plate, "-i", out_mp4,
                    "-filter_complex", "[0]scale=960:-2[a];[1]scale=960:-2[b];[a][b]hstack", "-frames:v",
                    str(len(done)), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", sbs_mp4], check=True)
    PROGRESS.update(name, "encode", 1.0)
    log(f"wrote {os.path.basename(out_mp4)} and {os.path.basename(sbs_mp4)} ({len(done)} frames)")
    rel = lambda p: os.path.relpath(p, job).replace(os.sep, "/")  # noqa: E731
    return dict(frames=len(done), video=rel(out_mp4), side_by_side=rel(sbs_mp4))


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
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, dict(status="ready", busy=_lock.locked(),
                             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                             torch=torch.__version__))

    def do_POST(self):
        if self.path in ("/set", "/comp"):
            return self._post_job(self.path[1:])
        if self.path != "/matte":
            return self._send(404, {"error": "not found"})
        name = ""
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            name = req.get("job", "")
            PROGRESS.start(name, [("load", 4), ("sam_seeds", 10), ("sam_video", 32), ("matanyone", 34),
                                  ("fuse", 8), ("export", 12)])
            mp.LOG_HOOK = lambda s: PROGRESS.log(name, s)
            mp.PROGRESS_HOOK = lambda st, fr, msg=None: PROGRESS.update(name, st, fr, msg)
            job = _job_dir(name)
            t0 = time.time()
            with _lock:
                rep = mp.run(job, "fused", anchor=int(req.get("anchor", 30)), chunk=int(req.get("chunk", 30)),
                             write=["fused"], crops=False, occlusion=True, mode=req.get("mode", "auto"))
                torch.cuda.empty_cache()
            PROGRESS.log(name, f"matte done in {time.time() - t0:.1f}s")
            PROGRESS.finish(name, ok=True)
            self._send(200, dict(ok=True, metrics=rep.get("fused", {}), seconds=round(time.time() - t0, 1),
                                 alpha_dir="matte/fused", occlusion_dir="occlusion"))
        except Exception as e:
            traceback.print_exc()
            PROGRESS.log(name, f"ERROR {type(e).__name__}: {e}")
            PROGRESS.finish(name, ok=False, error=f"{type(e).__name__}: {e}")
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            mp.LOG_HOOK = mp.PROGRESS_HOOK = None

    def _post_job(self, kind):
        name = ""
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            name = req.get("job", "")
            stages = [("mesh", 70), ("envmap", 30)] if kind == "set" else [("comp", 85), ("encode", 15)]
            PROGRESS.start(name, stages)
            t0 = time.time()
            with _lock:
                out = _do_set(name, req) if kind == "set" else _do_comp(name, req)
            PROGRESS.finish(name, ok=True)
            self._send(200, dict(ok=True, seconds=round(time.time() - t0, 1), **out))
        except Exception as e:
            traceback.print_exc()
            PROGRESS.log(name, f"ERROR {type(e).__name__}: {e}")
            PROGRESS.finish(name, ok=False, error=f"{type(e).__name__}: {e}")
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args, flush=True)


if __name__ == "__main__":
    print(f"[matte] listening on :{PORT}, work dir {WORK}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
