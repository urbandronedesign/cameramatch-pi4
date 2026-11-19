"""Export round trip: the exported camera must reproject the solve's 3D track points where the original does.

blender -b <solved scene>.blend --python tests/check_export.py -- <job_dir|auto> <out_dir>
1. export everything (export.export_scene), then the same with an animated focal (zoom 30 -> 45 mm)
2. Alembic / FBX / USD: re-import each into an empty scene, compare the camera's projection of the track points
   with the original's on sampled frames (pixels), and the focal per frame
3. camera.json + stmap_undistort.exr alone: project the 3D track points, distort through the ST map, compare with
   the tracked 2D positions (should equal the solve's own reprojection error)
4. Nuke: rebuild the camera from the .nk curves (Y up, rot_order XYZ, focal / haperture) and compare
"""
import importlib
import json
import math
import os
import re
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
job, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
E = cm.export
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[export] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


sc = bpy.context.scene
if argv[0] == "auto":
    job = bpy.path.abspath(sc.cameramatch.last_job)
print("[export] job", job, flush=True)
cam = sc.camera
W, H = sc.render.resolution_x, sc.render.resolution_y
fps = sc.render.fps / sc.render.fps_base
f0, f1 = sc.frame_start, sc.frame_end
d = np.load(os.path.join(job, "result.npz"))
tracks = next(o for o in sc.objects if o.name.endswith("_tracks") and o.type == "MESH")
P = np.array([(tracks.matrix_world @ v.co)[:] for v in tracks.data.vertices])
rng = np.random.default_rng(0)
sub = rng.choice(len(P), min(400, len(P)), replace=False)
FR = list(range(f0, f1 + 1, 12))
meta = dict(result=dict(k1=float(d["k1"]), k2=float(d["k2"])), track_obs=d["track_obs_full"],
            plate=bpy.path.abspath(cm._main_clip(bpy.context).filepath), generator="test")


def project(scene, camo, pts, frames):
    out_ = []
    for f in frames:
        scene.frame_set(f)
        uv = [world_to_camera_view(scene, camo, Vector(p)) for p in pts]
        out_.append([(c.x * W, (1 - c.y) * H, c.z) for c in uv])
    return np.array(out_)


def lens_curve(scene, camo, frames):
    v = []
    for f in frames:
        scene.frame_set(f)
        v.append(camo.data.lens)
    return np.array(v)


ref = project(sc, cam, P[sub], FR)
vis = (ref[..., 2] > 0.1) & (ref[..., 0] > 0) & (ref[..., 0] < W) & (ref[..., 1] > 0) & (ref[..., 1] < H)
ref_lens = lens_curve(sc, cam, FR)
w1 = E.export_scene(bpy.context, os.path.join(out, "shot"), "shot", include_cg=True, meta=meta)
check("all files written", all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in w1.values()),
      ", ".join(f"{os.path.basename(p)} {os.path.getsize(p) // 1024} KB" for p in w1.values()))

# zoom variant: animate the focal
lens0 = cam.data.lens
for f, ln in ((f0, 30.0), (f1, 45.0)):
    cam.data.lens = ln
    cam.data.keyframe_insert("lens", frame=f)
ref_zoom = project(sc, cam, P[sub], FR)
zoom_lens = lens_curve(sc, cam, FR)
w2 = E.export_scene(bpy.context, os.path.join(out, "zoom"), "zoom", formats=("abc", "fbx", "usd", "nk", "json"),
                    include_set=False, n_locators=0, meta=meta)
nk_ref = {"shot": ref, "zoom": ref_zoom}

# 3. camera.json + ST map vs the tracked 2D positions (before the scene is replaced)
J = json.load(open(w1["json"]))
im = bpy.data.images.load(w1["stmap_undistort"])
um = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)[::-1]   # rows top-down
obs = d["track_obs_full"]
fidx = d["frame_indices"].astype(int)
cs = cm._main_clip(bpy.context).frame_start
byframe = {fr["frame"]: fr for fr in J["frames"]}
sel = rng.choice(len(obs), min(30000, len(obs)), replace=False)
err_json, err_map = [], []
fpx_all = {}
for o in obs[sel]:
    k, pid, u, v = int(o[0]), int(o[1]), float(o[2]) + 0.5, float(o[3]) + 0.5      # KLT: centres at integers
    fr = int(fidx[k]) + cs
    if fr not in byframe or pid >= len(P):
        continue
    M = Matrix([byframe[fr]["matrix_world_zup"][i * 4:i * 4 + 4] for i in range(4)])
    pc = M.inverted() @ Vector(P[pid])
    if pc.z > -0.1:
        continue
    f_px = byframe[fr]["focal_px"]
    x, y = pc.x / -pc.z, -pc.y / -pc.z                                              # v down
    xd, yd = E.distort(np.array(x), np.array(y), J["distortion"]["k1"], J["distortion"]["k2"])
    ud, vd = float(xd) * f_px + W / 2, float(yd) * f_px + H / 2
    err_json.append(math.hypot(ud - u, vd - v))
    pu, pv = x * f_px + W / 2, y * f_px + H / 2                                     # undistorted pixel
    if 1 <= pu < W - 1 and 1 <= pv < H - 1:                                          # ST map lookup (bilinear)
        i0, j0 = int(pv - 0.5), int(pu - 0.5)
        a, b = pv - 0.5 - i0, pu - 0.5 - j0
        s = (um[i0, j0] * (1 - a) * (1 - b) + um[i0, j0 + 1] * (1 - a) * b + um[i0 + 1, j0] * a * (1 - b)
             + um[i0 + 1, j0 + 1] * a * b)
        err_map.append(math.hypot(s[0] * W - u, (1 - s[1]) * H - v))
med_solve = json.load(open(os.path.join(job, "result.json")))["refine"]["final"]["median"]
check("camera.json + distortion reproject the tracks like the solve", np.median(err_json) < med_solve * 1.3 + 0.05,
      f"median {np.median(err_json):.3f} px (solve {med_solve:.3f} px) over {len(err_json)} observations")
check("ST map (undistort) agrees", abs(np.median(err_map) - np.median(err_json)) < 0.05,
      f"median {np.median(err_map):.3f} px over {len(err_map)} observations")
rm = bpy.data.images.load(w1["stmap_redistort"])
rmap = np.array(rm.pixels[:], np.float32).reshape(H, W, 4)[::-1]
# round trip: redistort(undistort(p)) = p on a grid
ys, xs = np.mgrid[20:H - 20:37, 20:W - 20:41]
rt = []
for yy, xx in zip(ys.ravel(), xs.ravel()):
    s = um[yy, xx]                                     # undistorted pixel (xx, yy) reads the plate at s
    sx, sy = s[0] * W - 0.5, (1 - s[1]) * H - 0.5
    if 0 <= sx < W - 1 and 0 <= sy < H - 1:
        r_ = rmap[int(round(sy)), int(round(sx))]
        rt.append(math.hypot(r_[0] * W - 0.5 - xx, (1 - r_[1]) * H - 0.5 - yy))
check("ST maps invert each other", np.percentile(rt, 95) < 0.6, f"p95 {np.percentile(rt, 95):.3f} px (nearest lookup)")


# 4. Nuke: rebuild the camera from the script
def nuke_camera(path):
    txt = open(path).read()
    blk = txt[txt.index("Camera3 {"):]

    def curves(knob):
        m = re.search(knob + r" \{(\{curve x\d+ [^}]*\}) (\{curve x\d+ [^}]*\}) (\{curve x\d+ [^}]*\})\}", blk)
        return [np.array([float(v) for v in c.strip("{}").split()[2:]]) for c in m.groups()]

    T, R = curves("translate"), curves("rotate")
    fm = re.search(r" focal (\{curve x\d+ [^}]*\}|[\d.eE+-]+)", blk).group(1)
    F = np.array([float(v) for v in fm.strip("{}").split()[2:]]) if fm.startswith("{") else None
    ha = float(re.search(r" haperture ([\d.eE+-]+)", blk).group(1))
    rows = []
    for i in range(len(T[0])):
        Ry = Matrix.Rotation(math.radians(R[0][i]), 4, "X")
        Ry = Matrix.Rotation(math.radians(R[1][i]), 4, "Y") @ Ry
        Ry = Matrix.Rotation(math.radians(R[2][i]), 4, "Z") @ Ry              # rot_order XYZ: X first
        My = Matrix.Translation((T[0][i], T[1][i], T[2][i])) @ Ry
        rows.append((E.ZUP_TO_YUP.inverted() @ My, F[i] if F is not None else float(fm), ha))
    return rows


for tag, wr in (("shot", w1), ("zoom", w2)):
    rows = nuke_camera(wr["nk"])
    e = []
    for kf, f in enumerate(FR):
        Mz, foc, ha = rows[f - f0]
        f_px = foc / ha * W
        for j, p in enumerate(P[sub]):
            if not vis[kf, j]:
                continue
            pc = Mz.inverted() @ Vector(p)
            e.append(math.hypot(pc.x / -pc.z * f_px + W / 2 - nk_ref[tag][kf, j, 0],
                                -pc.y / -pc.z * f_px + H / 2 - nk_ref[tag][kf, j, 1]))
    check(f"Nuke camera ({tag}) reprojects like the original", np.max(e) < 0.05, f"max {np.max(e):.4f} px")

# 2. round trips (each in an empty scene)
for tag, wr, refp, refl in (("shot", w1, ref, ref_lens), ("zoom", w2, ref_zoom, zoom_lens)):
    for fmt in ("abc", "fbx", "usd"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        s2 = bpy.context.scene
        s2.render.resolution_x, s2.render.resolution_y = W, H
        s2.render.fps, s2.render.fps_base = round(fps), round(fps) / fps
        s2.frame_start, s2.frame_end = f0, f1
        try:
            if fmt == "abc":
                bpy.ops.wm.alembic_import(filepath=wr["abc"], as_background_job=False)
            elif fmt == "fbx":
                bpy.ops.import_scene.fbx(filepath=wr["fbx"], anim_offset=0.0)   # Blender default shifts keys 1 frame
            else:
                bpy.ops.wm.usd_import(filepath=wr["usd"])
        except Exception as ex:
            check(f"{tag} {fmt} import", False, str(ex)[:160])
            continue
        cams = [o for o in s2.objects if o.type == "CAMERA"]
        if not cams:
            check(f"{tag} {fmt}: camera present", False)
            continue
        c2 = cams[0]
        got = project(s2, c2, P[sub], FR)
        e = np.hypot(got[..., 0] - refp[..., 0], got[..., 1] - refp[..., 1])[vis]
        lens = lens_curve(s2, c2, FR)
        n_obj = len([o for o in s2.objects if o.type in {"MESH", "EMPTY", "ARMATURE"}])
        check(f"{tag} {fmt}: camera reprojects like the original", np.max(e) < 0.1,
              f"median {np.median(e):.4f} / max {np.max(e):.4f} px, focal max diff "
              f"{np.max(np.abs(lens - refl)):.4f} mm, {n_obj} other objects")
        if tag == "zoom":
            check(f"zoom {fmt}: animated focal kept", np.max(np.abs(lens - refl)) < 0.01,
                  f"{lens[0]:.2f} -> {lens[-1]:.2f} mm")
print("[export]", "ALL PASS" if ok else "SOME FAILED", flush=True)
