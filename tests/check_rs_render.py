"""Does Cycles' rolling shutter (as configured by set_builder.setup_render) reproduce the solved readout?

blender -b --factory-startup --python tests/check_rs_render.py -- <job_dir> <clip.mp4> [n_points]
Imports the job (solved WITH rolling shutter), picks fast-moving frames and tracked points near the top /
bottom of the image, renders a tiny emissive sphere at each solved 3D point (border render around it) with
the Cycles rolling shutter ON and OFF, and compares the rendered centroid to the KLT observation in the plate.
"""
import importlib
import os
import sys

import addon_utils
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
job, video = argv[0], argv[1]
NP = int(argv[2]) if len(argv) > 2 else 12
addon_utils.enable("bl_ext.user_default.cameramatch_pi4", default_set=True)
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sb = cm.set_builder
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
clip = bpy.data.movieclips.load(video)
clip.tracking.camera.sensor_width = 36.0
sc.cameramatch.last_job = job
cam, lens, _ = cm.import_result(sc, clip, job, sc.cameramatch)
for o in sc.objects:
    if o.name.endswith(("_points", "_tracks")):
        o.hide_render = True
d = np.load(os.path.join(job, "result.npz"))
rs = float(d["rolling_shutter"])
W, H = clip.size
print(f"[rs] job readout {rs:+.3f} ({sc['cm_rolling_shutter_axis']})", flush=True)

# fastest-rotating frames
c2w = d["c2w_refined"]
Rw = np.transpose(c2w[:, :3, :3], (0, 2, 1))
ang = [np.degrees(np.arccos(np.clip((np.trace(Rw[i + 1] @ Rw[i].T) - 1) / 2, -1, 1))) for i in range(len(Rw) - 1)]
fast = np.argsort(ang)[::-1][:40]
ob = d["track_obs_full"]
X = d["track_points"].astype(np.float64)
anchor = np.diag([1.0, 1, 1, 1]) @ cm.CV_TO_BL_WORLD @ np.linalg.inv(c2w[0])
fidx = d["frame_indices"].astype(int)
pick = []
for i in fast:
    m = (ob[:, 0] == i) & ((ob[:, 3] < 0.18 * H) | (ob[:, 3] > 0.82 * H))
    for r in ob[m][:2]:
        pick.append(r)
    if len(pick) >= NP:
        break
print(f"[rs] {len(pick)} samples on frames rotating {np.mean([ang[int(p[0])] for p in pick]):.2f} deg/frame", flush=True)

# render setup: emission sphere, transparent film, border render around the expected spot
mat = bpy.data.materials.new("emit")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Strength"].default_value = 20.0
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0], out.inputs[0])
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
sph = bpy.context.object
sph.data.materials.append(mat)
f_px = float(d["f_px"])
res = {"on": [], "off": []}
for mode in ("on", "off"):
    sc["cm_rolling_shutter"] = rs if mode == "on" else 0.0
    sb.setup_render(sc, bpy.app.tempdir, samples=16, motion_blur=False)
    sc.cycles.use_denoising = False
    sc.render.film_transparent = True
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.color = (0, 0, 0)
    im = sc.render.image_settings
    if hasattr(im, "media_type"):
        im.media_type = "IMAGE"
    im.file_format = "OPEN_EXR"
    im.color_depth = "32"
    sc.render.use_border = True
    sc.render.use_crop_to_border = False
    for p in pick:
        fi, t, u, v = int(p[0]), int(p[1]), float(p[2]), float(p[3])
        Pw = anchor @ np.r_[X[t], 1.0]
        sph.location = Pw[:3]
        dep = np.linalg.norm(Pw[:3] - np.array(cam.matrix_world.translation))
        sc.frame_set(int(fidx[fi]) + clip.frame_start)
        dep = (cam.matrix_world.inverted() @ sph.location).length
        sph.scale = (2.0 * dep / f_px,) * 3                      # ~2 px radius
        x0, x1 = max(u - 40, 0) / W, min(u + 40, W) / W
        y0, y1 = 1 - min(v + 40, H) / H, 1 - max(v - 40, 0) / H
        sc.render.border_min_x, sc.render.border_max_x, sc.render.border_min_y, sc.render.border_max_y = x0, x1, y0, y1
        path = os.path.join(bpy.app.tempdir, f"rs_{mode}.exr")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(path, check_existing=False)
        a = np.array(img.pixels[:], np.float32).reshape(H, W, 4)[::-1, :, 3]
        bpy.data.images.remove(img)
        ys, xs = np.nonzero(a > 0.05)
        if len(xs) == 0:
            continue
        wgt = a[ys, xs]
        cxr, cyr = (xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()
        res[mode].append(float(np.hypot(cxr - u, cyr - v)))
for mode in res:
    e = np.array(res[mode])
    print(f"[rs] Cycles rolling shutter {mode.upper():3s}: rendered sphere vs tracked feature: median {np.median(e):.2f}px, "
          f"max {e.max():.2f}px over {len(e)} samples", flush=True)

# the solve's own prediction at the same samples (rolling-shutter projection, numpy) for comparison
from mathutils import Matrix, Vector  # noqa: E402


def rotvec(Rm):
    ax, an = Matrix(Rm.tolist()).to_quaternion().to_axis_angle()
    return np.array(ax) * an


def rot(v):
    an = float(np.linalg.norm(v))
    return np.eye(3) if an < 1e-12 else np.array(Matrix.Rotation(an, 3, Vector(v / an)))


T = len(c2w)
Cw = c2w[:, :3, 3]
nxt, prv = np.minimum(np.arange(T) + 1, T - 1), np.maximum(np.arange(T) - 1, 0)
dt = np.maximum(nxt - prv, 1)[:, None]
om = np.array([rotvec(Rw[nxt[i]] @ Rw[prv[i]].T) for i in range(T)]) / dt
V = (Cw[nxt] - Cw[prv]) / dt
k1, k2 = float(d["k1"]), float(d["k2"])
for label, rr in (("model RS", rs), ("model global", 0.0)):
    es = []
    for p in pick:
        fi, t, u, v = int(p[0]), int(p[1]), float(p[2]), float(p[3])
        tt = (v - (H - 1) / 2) / H * rr
        R = rot(tt * om[fi]) @ Rw[fi]
        Xc = R @ (X[t] - (Cw[fi] + tt * V[fi]))
        x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
        r2 = x * x + y * y
        dd = 1 + k1 * r2 + k2 * r2 * r2
        es.append(np.hypot(f_px * x * dd + (W - 1) / 2 - u, f_px * y * dd + (H - 1) / 2 - v))
    print(f"[rs] {label:12s} (solver's projection, with k1/k2): median {np.median(es):.2f}px, max {np.max(es):.2f}px",
          flush=True)
