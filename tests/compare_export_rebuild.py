"""Plugin render vs the same shot rebuilt from its export folder (tests/rebuild_from_export.py). Runs in the matte image.

python /tools/../tests/compare_export_rebuild.py <job_dir> <export_dir> <cmp_dir>
<cmp_dir>/plugin_render/r_####.exr: the plugin scene rendered with set_builder.setup_render (same frames).
<cmp_dir>/import_render/r_####.exr: rebuild_from_export.py output.
Plugin comp: comp_plate (with the solved lens). Export comp: plate PNG x shadow + CG, both through
stmap_redistort.exr (what Nuke's STMap does). Prints per-frame silhouette IoU / centroid offset (pinhole renders),
the CG offset between the two comps, colour differences, and writes compare_sheet.jpg.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import OpenEXR

sys.path.insert(0, "/tools")
import comp_plate  # noqa: E402

JOB, EXP, C = sys.argv[1:4]           # job dir (plugin), export folder, comparison dir (plugin_render/, import_render/)
plate_mov = glob.glob(JOB + "/input/*.mp4")[0]
frames = sorted(int(os.path.basename(f)[2:6]) for f in glob.glob(C + "/plugin_render/r_*.exr"))

# plugin pipeline: its own comp
_r = np.load(JOB + "/result.npz")
LENS = dict(k1=float(_r["k1"]), k2=float(_r["k2"]), f_px=float(np.median(_r["f_px_frames"])))
print("lens", LENS)
comp_plate.run(plate_mov, C + "/plugin_render", C + "/plugin_comp", frames=frames, **LENS)


def exr(p):
    out = {}
    for part in OpenEXR.File(p).parts:
        for k, v in part.channels.items():
            out[k.replace("ViewLayer.", "")] = v.pixels.astype(np.float32)
    return out


def lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


st = exr(EXP + "/stmap_redistort.exr")["RGBA"]
H, W = st.shape[:2]
mx = (st[..., 0] * W - 0.5).astype(np.float32)              # source pixel (x right) in the undistorted render
my = ((1 - st[..., 1]) * H - 0.5).astype(np.float32)        # ST maps: origin bottom-left


def redistort(img):
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


rows, tiles = [], []
os.makedirs(C + "/import_comp", exist_ok=True)
for f in frames:
    P, I = exr(f"{C}/plugin_render/r_{f:04d}.exr"), exr(f"{C}/import_render/r_{f:04d}.exr")
    ap, ai = P["Combined"][..., 3], I["Combined"][..., 3]
    # 1. silhouettes, undistorted space (camera + geometry + animation transfer)
    mp, mi = ap > 0.5, ai > 0.5
    iou = (mp & mi).sum() / max((mp | mi).sum(), 1)
    cen = lambda m: np.array(np.nonzero(m)[::-1]).mean(1) if m.any() else np.array([np.nan, np.nan])  # noqa: E731
    dc = float(np.linalg.norm(cen(mp) - cen(mi)))
    # 2. the export-only comp (Nuke-style): plate x shadow (redistorted) + CG (redistorted), premultiplied over
    pl = cv2.imread(f"{EXP}/plate/plate_{f:04d}.png")[..., ::-1].astype(np.float32) / 255
    cg = redistort(np.dstack([I["Combined"][..., :3], ai]))
    sh = redistort(np.clip(I["Shadow Catcher"], 0, 1.5)) if "Shadow Catcher" in I else 1.0
    if not np.isscalar(sh):
        sh[cg[..., 3] == 0] = np.where(sh[cg[..., 3] == 0] == 0, 1.0, sh[cg[..., 3] == 0])
    comp = srgb(lin(pl) * sh * (1 - cg[..., 3:]) + cg[..., :3])
    cv2.imwrite(f"{C}/import_comp/comp_{f:04d}.png", (comp[..., ::-1] * 255 + 0.5).astype(np.uint8))
    pc = cv2.imread(f"{C}/plugin_comp/comp_{f:04d}.png")[..., ::-1].astype(np.float32) / 255
    # CG placement in the final (plate-lens) images: plugin comp puts the pinhole render as is
    md = redistort(ai) > 0.5
    mpd = cv2.remap(ap, *comp_plate.lens_maps(W, H, LENS["f_px"], LENS["k1"], LENS["k2"]), cv2.INTER_LINEAR) > 0.5
    shift = float(np.linalg.norm(cen(mpd) - cen(md)))          # plugin comp's CG (now through its lens) vs ST map
    mp = mpd
    region = mp | md
    diff = np.abs(pc - comp).mean(2)
    rows.append(dict(frame=f, iou_undistorted=round(float(iou), 4), centroid_px=round(dc, 2),
                     cg_px=int(mp.sum()), plugin_vs_lens_shift_px=round(shift, 2),
                     mean_abs_diff_cg_region=round(float(diff[region].mean()), 4) if region.any() else None,
                     mean_abs_diff_outside=round(float(diff[~region].mean()), 4)))
    if f in (15, 55, 95, 115):
        ov = (pc * 0.5).copy()
        ov[mp] = ov[mp] * 0.4 + np.array([1, 0.15, 0.15]) * 0.6      # red: plugin comp CG
        ov[md] = ov[md] * 0.4 + np.array([0.15, 1, 0.15]) * 0.6      # green: through the lens (export)
        ov[mp & md] = np.array([1, 1, 0.3])                          # yellow: both
        t = np.hstack([pc, comp, ov])
        t = (np.clip(cv2.resize(t, (t.shape[1] // 3, t.shape[0] // 3)), 0, 1) * 255).astype(np.uint8)
        cv2.putText(t, f"frame {f}: plugin comp | rebuilt from export | CG red=plugin green=export yellow=both",
                    (8, 22), 0, 0.55, (255, 255, 255), 1)
        tiles.append(t)
for r in rows:
    print(json.dumps(r))
a = lambda k: np.array([r[k] for r in rows if r[k] is not None], float)  # noqa: E731
print("SUMMARY", json.dumps(dict(
    iou_min=float(a("iou_undistorted").min()), iou_median=float(np.median(a("iou_undistorted"))),
    centroid_px_max=float(a("centroid_px").max()),
    plugin_lens_shift_px_median=float(np.median(a("plugin_vs_lens_shift_px"))),
    plugin_lens_shift_px_max=float(a("plugin_vs_lens_shift_px").max()),
    diff_cg_median=float(np.median(a("mean_abs_diff_cg_region"))),
    diff_outside_median=float(np.median(a("mean_abs_diff_outside"))))))
cv2.imwrite(C + "/compare_sheet.jpg", np.vstack(tiles)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
