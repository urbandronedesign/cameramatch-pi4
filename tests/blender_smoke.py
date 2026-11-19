"""Headless smoke test for the add-on's import path (no server needed).

Run: blender -b --factory-startup --python tests/blender_smoke.py
Builds a fake clip + a synthetic result.npz with known motion and checks the conversion:
  OpenCV camera moving +x (to its right) and looking +z  ->  Blender camera moving +X,
  looking +Y with +Z up, 1 unit per frame.
"""
import os
import sys
import tempfile

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "blender_addon"))
import cameramatch_pi4 as cm  # noqa: E402

cm.register()
tmp = tempfile.mkdtemp()

# Fake 5-frame PNG sequence -> movie clip.
for i in range(5):
    img = bpy.data.images.new(f"f{i}", 64, 36)
    img.filepath_raw = os.path.join(tmp, f"frame_{i + 1:04d}.png")
    img.file_format = "PNG"
    img.save()
clip = bpy.data.movieclips.load(os.path.join(tmp, "frame_0001.png"))
clip.tracking.camera.sensor_width = 36.0

# Synthetic solve: camera translates along its own +x, 1 m per frame, fx = 0.5 * W (=> 36mm lens).
T, W, H = 5, 512, 288
c2w = np.tile(np.eye(4), (T, 1, 1))
c2w[:, 0, 3] = np.arange(T)
pts = np.array([[0, 0, 5.0], [1, -1, 5.0]], np.float32)  # in front; 2nd one is 1m above (cv y down)
np.savez(os.path.join(tmp, "result.npz"), c2w=c2w, fx=np.full(T, W / 2.0), fy=np.full(T, W / 2.0),
         model_size=np.array([W, H]), frame_indices=np.arange(T), points=pts,
         colors=np.array([[255, 0, 0], [0, 255, 0]], np.uint8))


class S:  # stand-in for scene settings
    animate_focal = False


scene = bpy.context.scene
cam, lens, _ = cm.import_result(scene, clip, tmp, S)

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

check("lens 18mm (fx=W/2, 36mm sensor)", abs(lens - 18.0) < 1e-4)
for f in (1, 5):
    scene.frame_set(f)
    mw = cam.matrix_world
    fwd = -mw.col[2].xyz
    up = mw.col[1].xyz
    check(f"frame {f} position x={f - 1}", (mw.translation - type(mw.translation)((f - 1, 0, 0))).length < 1e-4)
    check(f"frame {f} looks +Y", fwd.dot((0, 1, 0)) > 0.999)
    check(f"frame {f} up +Z", up.dot((0, 0, 1)) > 0.999)
pobj = bpy.data.objects[f"Pi4_{clip.name}_points"]
v = [tuple(round(c, 4) for c in vv.co) for vv in pobj.data.vertices]
check("point 1 at (0,5,0)", v[0] == (0, 5, 0))
check("point 2 at (1,5,1) - above", v[1] == (1, 5, 1))
check("scene range 1-5", (scene.frame_start, scene.frame_end) == (1, 5))
deps = bpy.context.evaluated_depsgraph_get()
ev = pobj.evaluated_get(deps)
check("geometry nodes produce a point cloud", ev.type in {"POINTCLOUD", "MESH"})
print("SMOKE TEST", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
