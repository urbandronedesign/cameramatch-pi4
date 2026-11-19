"""End-to-end test against the running Pi4 container (add-on code path minus the modal UI).

Run: blender -b --factory-startup --python tests/blender_e2e.py -- <video> [stride] [max_frames]
"""
import os
import sys
import time

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "blender_addon"))
import cameramatch_pi4 as cm  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
video = os.path.abspath(argv[0])
stride = int(argv[1]) if len(argv) > 1 else 1
max_frames = int(argv[2]) if len(argv) > 2 else 300

cm.register()
for o in list(bpy.data.objects):  # factory startup cube/light/camera would sit on the solved path
    bpy.data.objects.remove(o)
clip = bpy.data.movieclips.load(video)
work = os.environ.get("CAMERAMATCH_WORK") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")
job = "e2e_" + time.strftime("%Y%m%d_%H%M%S")
job_dir = os.path.join(work, job)
os.makedirs(job_dir)
rel = cm._copy_clip_input(video, clip.source, job_dir)
payload = dict(job=job, input=rel, stride=stride, max_frames=max_frames, max_size=512, track=True,
               conf_threshold=0.05, num_points=200_000, refine=True, max_tracks=600)
t0 = time.time()
r = cm._http("http://127.0.0.1:8765/solve", payload)
print("SERVER", r, f"roundtrip {time.time() - t0:.1f}s")
if not r.get("ok"):
    sys.exit(1)
s = bpy.context.scene.cameramatch
cam, lens, lens_std = cm.import_result(bpy.context.scene, clip, job_dir, s)
fc = cam.animation_data.action
tc = clip.tracking.camera
print("REFINE", r.get("refine", {}).get("final"), "k1", tc.k1, "k2", tc.k2, "model", tc.distortion_model)
print(f"IMPORT camera={cam.name} lens={lens:.2f}mm std={lens_std:.2f} frames={bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}")
out = os.path.join(job_dir, "e2e.blend")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out)


