"""Plate, masks and depth of a solved job as image sequences for other software (runs in the matte image).

python /tools/export_maps.py <job_dir> <out_dir> --first 1 --scale 1.0 [--plate] [--masks] [--depth]
Numbered by scene frame (#### = source frame index + first). Everything at plate resolution:
  plate/plate_####.png         the footage, every frame of the solved range (8-bit, as decoded)
  masks/moving_####.png        Pi4 moving objects (soft, 0..255), solved frames
  masks/trees_water_sky_####.png  semantic ignore masks (when the solve computed them)
  masks/matte_####.png         MatAnyone 2 foreground matte (when Refine Matte was run)
  masks/ignore_####.png        the user's Ignore Mask (when one was drawn)
  depth/depth_####.exr         Pi4 depth, 32-bit float channel "Z", scene units, distance along the optical axis,
                               scaled to agree with the solved camera and 3D track points
Masks / depth are upscaled from the model resolution (bilinear). They follow the plate's lens (distorted).
Prints '@progress <0..1> <msg>' lines and a final JSON summary.
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
import OpenEXR

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def plate_src(job):
    ind = os.path.join(job, "input")
    seq = os.path.join(ind, "seq")
    if os.path.isdir(seq):
        return seq
    return os.path.join(ind, sorted(f for f in os.listdir(ind) if os.path.isfile(os.path.join(ind, f)))[0])


def frames(src, wanted):
    wanted = sorted(set(wanted))
    if os.path.isdir(src):
        files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT)
        for i in wanted:
            yield i, cv2.imread(os.path.join(src, files[i]), cv2.IMREAD_COLOR)
        return
    cap = cv2.VideoCapture(src)
    i, last, ws = 0, wanted[-1], set(wanted)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in ws:
            yield i, fr
        i += 1


def solve_alignment(r, dep, W, H):
    """One factor that makes Pi4's depth agree with the bundle-adjusted solve: the median over all inlier track
    observations of (solved distance along the optical axis / Pi4 depth at the observed pixel). A per-frame factor
    was tried and rejected: frames see different track subsets and the ratio follows the subset (Pi4's depth
    error depends on depth), not a real drift. Returns (factors [T], stats)."""
    if "track_obs" not in r or "c2w_refined" not in r or not len(r["track_obs"]):
        return np.ones(len(dep)), None
    obs, X, c2w = r["track_obs"], r["track_points"].astype(np.float64), r["c2w_refined"]
    T, Hm, Wm = dep.shape
    ratios = []
    for k in range(T):
        o = obs[obs[:, 0].astype(int) == k]
        if len(o) < 10:
            continue
        w2c = np.linalg.inv(c2w[k])
        Z = (X[o[:, 1].astype(int)] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]           # OpenCV camera: +z forward
        um = np.clip(((o[:, 2] + 0.5) * Wm / W).astype(int), 0, Wm - 1)
        vm = np.clip(((o[:, 3] + 0.5) * Hm / H).astype(int), 0, Hm - 1)
        d = dep[k][vm, um].astype(np.float64)
        ok = (Z > 0) & (d > 0)
        ratios.append(Z[ok] / d[ok])
    if not ratios:
        return np.ones(T), None
    q = np.concatenate(ratios)
    if len(q) < 50:
        return np.ones(T), None
    g = float(np.median(q))
    return np.full(T, g), dict(factor=g, observations=int(len(q)),
                               point_error_pct=float(100 * np.median(np.abs(np.log(q / g)))))


def write_depth(path, z):
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {"Z": np.ascontiguousarray(z.astype(np.float32))}).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--first", type=int, default=1, help="scene frame of source frame 0")
    ap.add_argument("--scale", type=float, default=1.0, help="scene scale (Set Scale) applied to depth")
    ap.add_argument("--plate", action="store_true")
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--depth", action="store_true")
    a = ap.parse_args()
    job = a.job
    r = np.load(os.path.join(job, "result.npz"))
    fidx = r["frame_indices"].astype(int)
    W, H = (int(v) for v in r["plate_size"]) if "plate_size" in r else (None, None)
    src = plate_src(job)
    if W is None:
        fr0 = next(frames(src, [int(fidx[0])]))[1]
        H, W = fr0.shape[:2]
    num = lambda i: int(i) + a.first  # noqa: E731
    summary = dict(width=W, height=H, first_frame=num(fidx[0]), last_frame=num(fidx[-1]), written={})
    steps = int(a.plate) + int(a.masks) + int(a.depth)
    done = 0

    def prog(frac, msg):
        print(f"@progress {(done + frac) / max(steps, 1):.3f} {msg}", flush=True)

    if a.plate:
        d = os.path.join(a.out, "plate")
        os.makedirs(d, exist_ok=True)
        rng = list(range(int(fidx[0]), int(fidx[-1]) + 1))
        n = 0
        for i, img in frames(src, rng):
            cv2.imwrite(os.path.join(d, f"plate_{num(i):04d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            n += 1
            if n % 20 == 0:
                prog(n / len(rng), f"plate {n}/{len(rng)}")
        summary["written"]["plate"] = n
        done += 1
    if a.masks:
        d = os.path.join(a.out, "masks")
        os.makedirs(d, exist_ok=True)
        cnt = {}
        sources = [("moving", os.path.join(job, "masks"), r"mask_(\d+)\.png", 1),       # named source index + 1
                   ("trees_water_sky", os.path.join(job, "semantic"), r"sem_(\d+)\.png", 1),
                   ("matte", os.path.join(job, "matte", "fused"), r"alpha_(\d+)\.png", 1)]
        for k, (name, folder, pat, base) in enumerate(sources):
            if not os.path.isdir(folder):
                continue
            files = [(int(m.group(1)), f) for f in sorted(os.listdir(folder)) if (m := re.match(pat, f))]
            for j, (n1, f) in enumerate(files):
                m_ = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if m_.shape[:2] != (H, W):
                    m_ = cv2.resize(m_, (W, H), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(d, f"{name}_{num(n1 - base):04d}.png"), m_)
                if j % 25 == 0:
                    prog((k + j / len(files)) / 4, f"masks: {name} {j}/{len(files)}")
            cnt[name] = len(files)
        mj = os.path.join(job, "ignore_mask.json")
        if os.path.isfile(mj):                          # already in scene-frame numbering, plate resolution
            meta = json.load(open(mj))
            n = 0
            for i in fidx:
                p = os.path.join(job, meta["pattern"].format(int(i) + int(meta["first_scene_frame"])))
                m_ = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m_ is not None:
                    if m_.shape[:2] != (H, W):
                        m_ = cv2.resize(m_, (W, H), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(d, f"ignore_{num(i):04d}.png"), m_)
                    n += 1
            cnt["ignore"] = n
        summary["written"]["masks"] = cnt
        done += 1
    if a.depth:
        d = os.path.join(a.out, "depth")
        os.makedirs(d, exist_ok=True)
        with np.load(os.path.join(job, "pi4_cache.npz")) as z:
            dep = z["depth"]                            # (T, Hm, Wm) half, metres along the optical axis
        fac, st = solve_alignment(r, dep, W, H)
        if st:
            print(f"depth: scaled x{st['factor']:.3f} to agree with the solve ({st['observations']} track observations, "
                  f"{st['point_error_pct']:.1f}% median per-point difference)", flush=True)
            summary["depth_alignment"] = st
        for k, i in enumerate(fidx):
            zz = cv2.resize(dep[k].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * a.scale * fac[k]
            write_depth(os.path.join(d, f"depth_{num(i):04d}.exr"), zz)
            if k % 20 == 0:
                prog(k / len(fidx), f"depth {k}/{len(fidx)}")
        summary["written"]["depth"] = len(fidx)
        done += 1
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
