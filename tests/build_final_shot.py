"""Final showcase shot: Venice alley, imported character walking away + robot walking toward camera, crossing.

blender -b <solved.blend> --python tests/build_final_shot.py -- <character.glb> <out.blend> <render_dir> [samples]
Uses the add-on operators exactly as a user would (Build Set, Move Along Path, Add Walking Robot with Avoid
Collisions on, Render setup); the character goes first, the robot avoids it.
"""
import importlib
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out, rdir = os.path.abspath(argv[0]), os.path.abspath(argv[1]), os.path.abspath(argv[2])
samples = int(argv[3]) if len(argv) > 3 else 192
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
s = sc.cameramatch
clip = cm._main_clip(bpy.context)
cm.set_builder.build_set(sc, clip, s.last_job, print, street_walls=True)
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


def bezier(name, pts):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = (p[0], p[1], fz(p[0], p[1]))
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    ob = bpy.data.objects.new(name, cu)
    sc.collection.objects.link(ob)
    ob.hide_render = True
    return ob


def select_only(objs, active):
    for o in sc.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = active


# ---- imported character, walking away up the right side of the street
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=glb)
new = [o for o in bpy.data.objects if o not in before]
arm = next(o for o in new if o.type == "ARMATURE")
root = cm.characters.hierarchy_root(arm)
bpy.context.view_layer.update()
zs = [(o.matrix_world @ Vector(c)).z for o in new if o.type == "MESH" for c in o.bound_box]
root.scale = tuple(v * 1.7 / (max(zs) - min(zs)) for v in root.scale)
for o in new:                                     # the glTF ships a stray reference sphere: keep only the rig
    if o.type == "MESH" and cm.characters.hierarchy_root(o) is not root:
        o.hide_render = o.hide_viewport = True
bpy.context.view_layer.update()
c_char = bezier("Character_path", [(xc(y) + 0.35, y) for y in np.linspace(7.5, 22.0, 6)])
select_only([arm, c_char], c_char)
with bpy.context.temp_override(active_object=c_char, selected_objects=[arm, c_char]):
    print("character:", bpy.ops.cameramatch.move_along_path(speed=1.0))

# ---- robot, walking toward the camera down an S-curve on the left, avoiding the character
c_rob = bezier("Robot_path", [(xc(y) - 0.15 + 0.3 * math.sin((y - 6) * 0.6), y) for y in np.linspace(17.5, 7.0, 7)])
select_only([c_rob], c_rob)
with bpy.context.temp_override(active_object=c_rob, selected_objects=[c_rob]):
    print("robot:", bpy.ops.cameramatch.add_robot())
for line in cm._STATUS["log"][-8:]:
    print("LOG", line)

cm.set_builder.setup_render(sc, rdir, samples, True, log=print)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out)
