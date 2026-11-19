"""Showcase shot with Mixamo characters: Venice alley, a Mixamo soldier walking away up the alley and a Mixamo
X Bot walking toward the camera, stepping around him.

blender -b <solved.blend> --python tests/build_mixamo_shot.py -- <soldier.glb> <xbot.glb> <out.blend> <render_dir> [samples]
Uses the add-on as a user would: Build Set, Add Character (scaled, on the floor), Walk Along Path with Avoid
Collisions on (the second character avoids the first), Render setup.
"""
import importlib
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
soldier, xbot, out, rdir = (os.path.abspath(a) for a in argv[:4])
samples = int(argv[4]) if len(argv) > 4 else 192
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
s = sc.cameramatch
clip = cm._main_clip(bpy.context)
cm.set_builder.build_set(sc, clip, s.last_job, print, street_walls=True)
cm.hide_camera_blockers(sc, print)
fz = lambda x, y: cm.set_builder.floor_z(sc, x, y)  # noqa: E731
sc["cm_movers"] = "{}"
s.avoid_collisions = True

fl = bpy.data.objects["CM_floor_catcher"]
V = np.array([(fl.matrix_world @ v.co)[:] for v in fl.data.vertices])


def xc(y):
    m = np.abs(V[:, 1] - y) < 0.3
    if not m.any():
        m = np.abs(V[:, 1] - V[np.argmin(np.abs(V[:, 1] - y)), 1]) < 0.3
    return float(np.median(V[m, 0]))


class Op:
    def report(self, t, m):
        print("REPORT", m, flush=True)


def walk(char_file, height, pts, speed, name):
    s.char_height = height
    sc.cursor.location = (pts[0][0], pts[0][1], fz(*pts[0]))
    r = cm.add_character(Op(), bpy.context, char_file)
    assert r == {"FINISHED"}, r
    root = bpy.context.view_layer.objects.active
    arm = cm.characters.find_armature(root)
    crv = cm._new_floor_curve(bpy.context, name, pts)
    for o in sc.objects:
        o.select_set(o in (root, arm, crv))
    bpy.context.view_layer.objects.active = crv
    with bpy.context.temp_override(active_object=crv, selected_objects=[root, arm, crv]):
        print(name, bpy.ops.cameramatch.move_along_path(speed=speed), flush=True)
    return root


# the soldier walks away up the right half of the alley; the X Bot comes toward the camera on an S-curve and
# has to step around him
walk(soldier, 1.80, [(xc(y) + 0.3, y) for y in np.linspace(6.5, 20.0, 6)], 1.05, "Soldier_path")
walk(xbot, 1.72, [(xc(y) - 0.1 + 0.25 * np.sin((y - 6) * 0.6), y) for y in np.linspace(17.0, 6.5, 7)], 0.95,
     "Xbot_path")
for line in cm._STATUS["log"][-14:]:
    print("LOG", line, flush=True)

cm.set_builder.setup_render(sc, rdir, samples, True, log=print)
sc.render.use_persistent_data = True
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out, flush=True)
