"""Robot along a drawn curve + Move Along Path for an arbitrary object, on a solved shot (headless).

blender -b <solved.blend> --python tests/check_paths.py -- <out.blend>
Builds the set (floor fit), draws an S-curve down the street for the robot and a second curve for a box
'vehicle', runs both operators and checks: robot pelvis on its curve, box on the floor, box heading = tangent.
"""
import importlib
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
clip = cm._main_clip(bpy.context)
cm.set_builder.build_set(sc, clip, sc.cameramatch.last_job, print, street_walls=True)
fz = lambda x, y: cm.set_builder.floor_z(sc, x, y)  # noqa: E731
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[paths] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


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
    return ob


# street centre from the street-walls fit is not exposed; use the floor-catcher mesh centre line per y
fl = bpy.data.objects["CM_floor_catcher"]
V = np.array([(fl.matrix_world @ v.co)[:] for v in fl.data.vertices])


def xc(y):
    m = np.abs(V[:, 1] - y) < 0.3
    return float(np.median(V[m, 0])) if m.any() else 0.5


s_curve = [(xc(y) + 0.35 * math.sin((y - 6) * 0.7), y) for y in np.linspace(16.0, 7.0, 7)]
c_robot = bezier("Robot_path", s_curve)
c_car = bezier("Box_path", [(xc(y) - 0.2, y) for y in np.linspace(6.5, 20.0, 5)])

# --- robot along the curve
for o in sc.objects:
    o.select_set(False)
c_robot.select_set(True)
bpy.context.view_layer.objects.active = c_robot
with bpy.context.temp_override(active_object=c_robot, selected_objects=[c_robot]):
    r = bpy.ops.cameramatch.add_robot()
check("add_robot along curve", r == {"FINISHED"}, str(r))
pel = next(o for o in bpy.data.objects if o.name.startswith("Robot_pelvis"))
pts = cm.paths.curve_points(c_robot)[:, :2]
d = []
for fr in range(sc.frame_start, sc.frame_end + 1, 10):
    sc.frame_set(fr)
    p = np.array(pel.matrix_world.translation[:2])
    d.append(float(np.min(np.linalg.norm(pts - p, axis=1))))
check("robot follows the curve", np.max(d) < 0.08, f"max distance pelvis-curve {np.max(d) * 100:.1f} cm (sway incl.)")

# --- box 'vehicle' along the second curve
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
box = bpy.context.active_object
box.name = "Box_vehicle"
box.scale = (1.0, 2.0, 0.7)        # 1 x 2 x 1.4 m, long axis = local Y -> forward -Y
for o in sc.objects:
    o.select_set(o in (box, c_car))
bpy.context.view_layer.objects.active = c_car
with bpy.context.temp_override(active_object=c_car, selected_objects=[box, c_car]):
    r = bpy.ops.cameramatch.move_along_path(speed=1.2, forward="-Y")
check("move_along_path", r == {"FINISHED"}, str(r))
pts2 = cm.paths.curve_points(c_car)[:, :2]
errs_floor, errs_head = [], []
for fr in range(sc.frame_start, sc.frame_end + 1, 15):
    sc.frame_set(fr)
    bb = [box.matrix_world @ Vector(c) for c in box.bound_box]
    low = min(c.z for c in bb)
    c = box.matrix_world.translation
    errs_floor.append(abs(low - fz(c.x, c.y)))
    fwd = -(box.matrix_world.to_3x3() @ Vector((0, 1, 0)))
    # its ACTUAL travel direction (the path may detour around the robot / the walls): motion over +-3 frames
    sc.frame_set(min(fr + 3, sc.frame_end))
    c2 = box.matrix_world.translation.copy()
    sc.frame_set(max(fr - 3, sc.frame_start))
    c1 = box.matrix_world.translation.copy()
    sc.frame_set(fr)
    t = (c2 - c1).xy
    if t.length < 1e-4:
        continue
    errs_head.append(math.degrees(abs(math.atan2(fwd.y, fwd.x) - math.atan2(t.y, t.x))) % 360)
check("box rests on the floor", max(errs_floor) < 0.02, f"max gap {max(errs_floor) * 100:.1f} cm")
check("box faces its direction of travel", max(min(e, 360 - e) for e in errs_head) < 5,
      f"max heading error {max(min(e, 360 - e) for e in errs_head):.1f} deg")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("[paths]", "ALL PASS" if ok else "SOME FAILED", "->", out, flush=True)
