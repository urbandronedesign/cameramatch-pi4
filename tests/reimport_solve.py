"""Replace the imported solve in a .blend with the job's current result.npz (e.g. after a /refine re-run).

blender -b <solved.blend> --python tests/reimport_solve.py -- <out.blend>
"""
import importlib
import os
import sys

import bpy

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
clip = next(c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte")))
for coll in [c for c in bpy.data.collections if c.name.startswith("Pi4_")]:
    for ob in list(coll.objects):
        bpy.data.objects.remove(ob)
    bpy.data.collections.remove(coll)
cam, lens, std = cm.import_result(sc, clip, sc.cameramatch.last_job, sc.cameramatch)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("REIMPORTED", cam.name, f"lens {lens:.2f}mm", "k1", clip.tracking.camera.k1, "->", out)
