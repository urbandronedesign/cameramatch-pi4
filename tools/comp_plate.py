"""Composite Cycles CG renders (multilayer EXR: Combined + Shadow Catcher) over the plate, matched to it.

python /tools/comp_plate.py <plate.mp4|seq_dir/%06d.png> <exr_dir> <out_dir> [--frames 30,140] [--grain auto|<std>]
Also used by matte_server.py (POST /comp) through run().
- plate x Shadow Catcher pass (physical CG shadows / occlusion of the real light)
- CG black level + softness matched to the plate (measured), light wrap from the plate
- grain measured from the plate's high-frequency noise (flat regions), added to the CG only
Frames: r_####.exr, #### = 1-based plate frame number. Writes comp_####.png (sRGB 8-bit).
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import OpenEXR


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def read_exr(p):
    f = OpenEXR.File(p)
    out = {}
    for part in f.parts:
        for k, v in part.channels.items():
            out[k.replace("ViewLayer.", "")] = v.pixels.astype(np.float32)
    return out


def plate_stats(frames):
    """Black level (display, per channel) and grain std (display, per channel) of the plate."""
    blk, gr = [], []
    for fr in frames:
        d = fr.astype(np.float32) / 255.0
        blk.append(np.percentile(d.reshape(-1, 3), 0.3, axis=0))
        hp = d - cv2.GaussianBlur(d, (0, 0), 1.2)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F, 0, 1)
        flat = np.hypot(gx, gy) < np.percentile(np.hypot(gx, gy), 25)
        gr.append(1.4826 * np.median(np.abs(hp[flat]), axis=0))
    return np.median(blk, axis=0), np.median(gr, axis=0)


LOG_HOOK = None
PROGRESS_HOOK = None


def _log(msg):
    print(msg, flush=True)
    if LOG_HOOK:
        LOG_HOOK(msg)


def run(plate, exr_dir, out, frames=None, grain="auto", soft=0.6, wrap=0.25, lift="auto"):
    """Composite every r_####.exr in exr_dir (or `frames`) -> out/comp_####.png. Returns written frame numbers."""
    os.makedirs(out, exist_ok=True)
    exrs = {int(re.findall(r"(\d+)\.exr$", p)[0]): p for p in glob.glob(os.path.join(exr_dir, "r_*.exr"))}
    want = sorted(exrs) if not frames else sorted(int(x) for x in frames)
    if not want:
        raise RuntimeError(f"no r_####.exr renders in {exr_dir}")
    cap = cv2.VideoCapture(plate)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise RuntimeError(f"cannot read plate {plate}")
    samp = []
    for i in np.linspace(0, n - 1, 6).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            samp.append(fr)
    blk, grn = plate_stats(samp)            # BGR order
    if lift != "auto":
        blk = np.full(3, float(lift))
    if grain != "auto":
        grn = np.full(3, float(grain))
    _log(f"plate black level {np.round(blk[::-1], 4)} (RGB, display), grain std {np.round(grn[::-1], 4)}")
    rng = np.random.default_rng(7)
    cap = cv2.VideoCapture(plate)
    cur, done = 0, []
    for k, fnum in enumerate(want):
        idx = fnum - 1
        if idx < cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cur = idx
        plate_fr = None
        while cur <= idx:
            ok, plate_fr = cap.read()
            cur += 1
            if not ok:
                break
        if plate_fr is None or fnum not in exrs:
            continue
        E = read_exr(exrs[fnum])
        cg = E["Combined"][..., [2, 1, 0, 3]]            # RGBA -> BGRA, premultiplied linear
        if cg.shape[:2] != plate_fr.shape[:2]:
            raise RuntimeError(f"render {cg.shape[1]}x{cg.shape[0]} != plate {plate_fr.shape[1]}x{plate_fr.shape[0]}")
        shad = np.clip(E["Shadow Catcher"][..., ::-1], 0, 1.5) if "Shadow Catcher" in E else 1.0
        pl = srgb2lin(plate_fr.astype(np.float32) / 255.0)
        bg = pl * shad
        if soft > 0:
            cg = cv2.GaussianBlur(cg, (0, 0), soft)
        al = np.clip(cg[..., 3:4], 0, 1)
        # black-level match in display space on the un-premultiplied CG
        col = np.where(al > 1e-4, cg[..., :3] / np.maximum(al, 1e-4), 0)
        col = srgb2lin(blk + (1 - blk) * lin2srgb(col))
        # light wrap: blurred plate bleeding onto the CG edges
        wr = 0
        if wrap > 0:
            inv = cv2.GaussianBlur((1 - al[..., 0]), (0, 0), 4)[..., None]
            wr = cv2.GaussianBlur(bg, (0, 0), 5) * inv * al * wrap
        res = lin2srgb(col * al + bg * (1 - al) + wr)
        # grain on the CG area (luma-correlated, ~1 px grain), matched to the plate's measured std
        g_m = cv2.GaussianBlur(rng.standard_normal(res.shape[:2]).astype(np.float32), (0, 0), 0.55)
        g_c = cv2.GaussianBlur(rng.standard_normal(res.shape).astype(np.float32), (0, 0), 0.55)
        g = 0.75 * g_m[..., None] + 0.45 * g_c
        g /= g.std() + 1e-6
        res = res + g * grn * al
        cv2.imwrite(os.path.join(out, f"comp_{fnum:04d}.png"), np.clip(res * 255 + 0.5, 0, 255).astype(np.uint8))
        done.append(fnum)
        if PROGRESS_HOOK:
            PROGRESS_HOOK((k + 1) / len(want), f"composited frame {fnum} ({k + 1}/{len(want)})")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plate")
    ap.add_argument("exr_dir")
    ap.add_argument("out")
    ap.add_argument("--frames", default="")
    ap.add_argument("--grain", default="auto")
    ap.add_argument("--soft", type=float, default=0.6)
    ap.add_argument("--wrap", type=float, default=0.25)
    ap.add_argument("--lift", default="auto")
    a = ap.parse_args()
    fr = [int(x) for x in a.frames.split(",")] if a.frames else None
    done = run(a.plate, a.exr_dir, a.out, fr, a.grain, a.soft, a.wrap, a.lift)
    print("COMPOSITED", len(done), "frames")


if __name__ == "__main__":
    main()
