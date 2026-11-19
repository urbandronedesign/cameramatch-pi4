"""Headless checks for Set Scale from Distance, Export Tracks to Clip and zoom import.

blender -b <solved.blend> --python tests/check_robust_features.py -- [zoom_job_dir zoom_plate_first_png zoom_gt.json]
"""
import json
import os
import sys

import bpy
import importlib

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
job = sc.cameramatch.last_job
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[check] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


def state():
    cam = sc.camera
    sc.frame_set(sc.frame_end)
    end = cam.matrix_world.translation.copy()
    sc.frame_set(sc.frame_start)
    start = cam.matrix_world.translation.copy()
    p = next(o for o in sc.objects if o.name.endswith("_points"))
    return (end - start).length, p.dimensions.length


path0, ext0 = state()
# two empties on the camera's first and last positions: their distance is the camera path chord
e1 = bpy.data.objects.new("m1", None)
e2 = bpy.data.objects.new("m2", None)
sc.collection.objects.link(e1)
sc.collection.objects.link(e2)
sc.frame_set(sc.frame_start)
e1.location = sc.camera.matrix_world.translation
sc.frame_set(sc.frame_end)
e2.location = sc.camera.matrix_world.translation
bpy.context.view_layer.update()
for o in sc.objects:
    o.select_set(o in (e1, e2))
bpy.context.view_layer.objects.active = e1
r = bpy.ops.cameramatch.set_scale("EXEC_DEFAULT", real_distance=2 * path0)
path1, ext1 = state()
check("set_scale ran", r == {"FINISHED"}, str(r))
check("camera path x2", abs(path1 / path0 - 2) < 1e-3, f"{path0:.4f} -> {path1:.4f}")
check("point cloud x2", abs(ext1 / ext0 - 2) < 1e-3, f"{ext0:.3f} -> {ext1:.3f}")
check("scene_scale.json", abs(cm.job_scale(job) - 2) < 1e-3, f"{cm.job_scale(job):.4f}")
check("no duplicate import", sum(c.name.startswith("Pi4_") for c in bpy.data.collections) == 1)
os.remove(os.path.join(job, "scene_scale.json"))          # leave the job at its original scale

clip = cm._main_clip(bpy.context)
cam = sc.camera
r = bpy.ops.cameramatch.export_tracks("EXEC_DEFAULT", max_tracks=1500)
tr = [t for t in clip.tracking.tracks if t.name.startswith("Pi4_")]
nm = sum(len([m for m in t.markers if not m.mute]) for t in tr)
check("export_tracks", r == {"FINISHED"} and len(tr) > 100, f"{len(tr)} tracks, {nm} markers, "
      f"{nm / max(len(tr), 1):.0f} markers/track")
# a marker must sit where the solved camera projects the track's 3D point
import numpy as np  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402

d = np.load(os.path.join(job, "result.npz"))
tp = d["track_points"].astype(np.float64)
c2w = d["c2w_refined"]
anchor = cm.CV_TO_BL_WORLD @ np.linalg.inv(c2w[0])
errs = []
for t in tr[:200]:
    k = int(t.name[4:])
    P = anchor @ np.r_[tp[k], 1.0]
    m = next(m for m in t.markers if not m.mute)
    sc.frame_set(int(m.frame) + clip.frame_start - 1)
    co = world_to_camera_view(sc, cam, __import__("mathutils").Vector(P[:3]))
    W, H = clip.size
    errs.append(np.hypot((co.x - m.co[0]) * W, (co.y - m.co[1]) * H))
check("markers match solve", np.median(errs) < 2.0, f"median {np.median(errs):.2f}px over {len(errs)} tracks "
      "(lens distortion ignored)")

if len(argv) >= 3:
    zjob, zfirst, zgt = argv[0], argv[1], argv[2]
    gt = json.load(open(zgt))
    zclip = bpy.data.movieclips.load(zfirst)
    zclip.tracking.camera.sensor_width = 36.0
    zclip.name = "gt_zoom"
    s = sc.cameramatch
    s.animate_focal = True
    zcam, lens_med, lens_std = cm.import_result(sc, zclip, zjob, s)
    lens = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        lens.append(zcam.data.lens)
    f_gt = np.array(gt["f_px_frames"]) / 1920 * 36.0
    rel = np.abs(np.array(lens) / f_gt[: len(lens)] - 1)
    check("zoom lens keyframed", lens[-1] / lens[0] > 1.5, f"lens {lens[0]:.1f} -> {lens[-1]:.1f} mm "
          f"(GT {f_gt[0]:.1f} -> {f_gt[-1]:.1f})")
    check("zoom lens vs GT", np.median(rel) < 0.02, f"median error {np.median(rel) * 100:.2f}%")
print("ALL PASS" if ok else "SOME FAILED", flush=True)
