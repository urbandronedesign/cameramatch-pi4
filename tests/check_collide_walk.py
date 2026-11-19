"""Collision avoidance + foot-slide checks on a solved shot (headless).

blender -b <solved.blend> --python tests/check_collide_walk.py -- <character.glb> <out.blend>
1. walls: robot curve hugging the right wall, avoidance OFF vs ON -> min distance pelvis track / walls.
2. movers: box along the street first, then a robot S-curve crossing it -> min footprint gap, OFF vs ON.
3. feet: imported character (glTF) along a curve with the walk cycle's own speed vs a naive fixed speed ->
   planted-foot slide (cm/s).
"""
import importlib
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
clip = cm._main_clip(bpy.context)
cm.set_builder.build_set(sc, clip, sc.cameramatch.last_job, print, street_walls=True)
fz = lambda x, y: cm.set_builder.floor_z(sc, x, y)  # noqa: E731
s = sc.cameramatch
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[collide] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


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


def select_only(objs, active):
    for o in sc.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = active


fl = bpy.data.objects["CM_floor_catcher"]
V = np.array([(fl.matrix_world @ v.co)[:] for v in fl.data.vertices])
kd, _ = cm.collide.obstacle_tree(sc, fz)


def wall_x(y, side):
    """Nearest wall sample to the street centre at this y, on one side."""
    m = np.abs(V[:, 1] - y) < 0.3
    if not m.any():                                 # beyond the floor-catcher support: nearest covered row
        m = np.abs(V[:, 1] - V[np.argmin(np.abs(V[:, 1] - y)), 1]) < 0.3
    xc = float(np.median(V[m, 0]))
    for dx in np.arange(0, 3, 0.05):
        x = xc + side * dx
        if kd.find((x, y, 0))[2] < 0.06:
            return x, xc
    return xc + side * 1.2, xc


def robot_min_wall(track):
    return min(kd.find((x, y, 0))[2] for x, y, _ in track)


def robot_track():
    pel = max((o for o in bpy.data.objects if o.name.startswith("Robot_pelvis")), key=lambda o: o.name)
    tr = []
    for fr in range(sc.frame_start, sc.frame_end + 1, 2):
        sc.frame_set(fr)
        tr.append((pel.matrix_world.translation.x, pel.matrix_world.translation.y, 0))
    return tr, pel


def remove_robots():
    for c in [c for c in bpy.data.collections if c.name.startswith("CM_Robot")]:
        for o in list(c.objects):
            bpy.data.objects.remove(o)
        bpy.data.collections.remove(c)


# ------------------------------------------------------------------ 1. walls
hug = []
for y in np.linspace(15.0, 8.0, 6):
    wx, xc = wall_x(y, +1)
    hug.append((wx - 0.1, y))                       # 10 cm from the right wall: the robot would scrape it
c1 = bezier("Hug_path", hug)
res = {}
for avoid in (False, True):
    remove_robots()
    sc["cm_movers"] = "{}"
    s.avoid_collisions = avoid
    select_only([c1], c1)
    with bpy.context.temp_override(active_object=c1, selected_objects=[c1]):
        bpy.ops.cameramatch.add_robot()
    tr, _ = robot_track()
    res[avoid] = robot_min_wall(tr)
check("robot keeps clear of the walls", res[True] >= 0.33 and res[True] > res[False],
      f"closest wall: {res[False]:.2f} m without avoidance -> {res[True]:.2f} m with")

# ------------------------------------------------------------------ 2. movers (box then crossing robot)
remove_robots()
sc["cm_movers"] = "{}"
s.avoid_collisions = True
xs = {y: wall_x(y, -1) for y in (6.5, 10.0, 13.5, 17.0, 20.0)}
c_box = bezier("Box_path", [(xc - 0.25, y) for y, (_, xc) in xs.items()])
bpy.ops.mesh.primitive_cube_add(size=1.0)
box = bpy.context.active_object
box.name = "Box_vehicle"
box.scale = (0.8, 1.6, 0.7)
select_only([box, c_box], c_box)
with bpy.context.temp_override(active_object=c_box, selected_objects=[box, c_box]):
    bpy.ops.cameramatch.move_along_path(speed=1.2, forward="-Y")
s_curve = [(wall_x(y, 1)[1] + 0.3 * math.sin((y - 6) * 0.7) - 0.2, y) for y in np.linspace(16.0, 7.0, 7)]
c_rob = bezier("Robot_path", s_curve)
gaps = {}
for avoid in (False, True):
    remove_robots()
    reg = cm.collide.movers(sc)
    sc["cm_movers"] = __import__("json").dumps({k: v for k, v in reg.items() if k == box.name})
    s.avoid_collisions = avoid
    select_only([c_rob], c_rob)
    with bpy.context.temp_override(active_object=c_rob, selected_objects=[c_rob]):
        bpy.ops.cameramatch.add_robot()
    m_box = cm.collide.movers(sc)[box.name]
    pel = max((o for o in bpy.data.objects if o.name.startswith("Robot_pelvis")), key=lambda o: o.name)
    g = float("inf")
    for k, fr in enumerate(range(sc.frame_start, sc.frame_end + 1)):
        sc.frame_set(fr)
        p = pel.matrix_world.translation
        g = min(g, cm.collide.mover_gap((p.x, p.y), 0.0, 0.3, 0.3, m_box, k))
    gaps[avoid] = g
check("robot and box never collide", gaps[True] >= 0.0 and gaps[True] > gaps[False],
      f"min footprint gap: {gaps[False]:+.2f} m without avoidance -> {gaps[True]:+.2f} m with")

# ------------------------------------------------------------------ 3. character feet
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=glb)
new = [o for o in bpy.data.objects if o not in before]
root = cm.characters.hierarchy_root(next(o for o in new if o.type == "ARMATURE"))
arm = cm.characters.find_armature(root)
bpy.context.view_layer.update()
pts = [(o.matrix_world @ Vector(c)).z for o in new if o.type == "MESH" for c in o.bound_box]
h = max(pts) - min(pts)
root.scale = tuple(v * 1.7 / h for v in root.scale)          # a 1.7 m character
bpy.context.view_layer.update()
c_ch = bezier("Char_path", [(wall_x(y, 1)[1] - 0.3, y) for y in np.linspace(24.0, 6.0, 7)])
sc["cm_movers"] = "{}"
select_only([arm, c_ch], c_ch)
with bpy.context.temp_override(active_object=c_ch, selected_objects=[arm, c_ch]):
    r = bpy.ops.cameramatch.move_along_path(speed=1.4)       # realistic pace for 1.7 m: cycle slowed to match
check("character moved", r == {"FINISHED"}, str(r))
act = cm.characters.walk_action(arm)
sl_auto = cm.characters.foot_slide(sc, arm, cm.characters.foot_bones(arm))
# analyse the cycle at rest (the root is animated now: mute its keys while measuring)
root.animation_data.action_influence = 0.0 if hasattr(root.animation_data, "action_influence") else 1.0
saved = root.animation_data.action
root.animation_data.action = None
loc0 = root.location.copy()
info = cm.characters.analyse_walk(sc, root, arm, act)
root.animation_data.action = saved
# naive: same path at the cycle's AVERAGE speed, constant (what v0.10 did) -> slides within each step
fps = sc.render.fps / sc.render.fps_base
v_bad = 1.4          # same travel speed and cycle playback, but constant instead of the planted-foot profile
pts_c = cm.paths.curve_points(c_ch)[:, :2]
path, L = cm.paths.ground_path(pts_c)
tr = arm.animation_data.nla_tracks["CM_walk"]
fwd = info["forward"]
cm.paths.move_one(root, path, L, fz, sc, v_bad, root.rotation_euler.z - math.atan2(fwd[1], fwd[0]),
                  info["sole_above"])
sl_bad = cm.characters.foot_slide(sc, arm, info["feet"])
check("planted feet do not slide", sl_auto["median_cm_s"] < 0.35 * sl_bad["median_cm_s"],
      f"walk '{info['action']}' {info['profile_mean_mpf'] * fps:.2f} m/s, root motion {info['root_motion'] * 100:.0f}%: "
      f"planted-foot slide median/p90 {sl_bad['median_cm_s']:.1f}/{sl_bad['p90_cm_s']:.1f} cm/s at constant speed -> "
      f"{sl_auto['median_cm_s']:.1f}/{sl_auto['p90_cm_s']:.1f} cm/s with the cycle's speed profile")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("[collide]", "ALL PASS" if ok else "SOME FAILED", flush=True)

# planted foot on the floor (soles): lowest character point vs floor, over the walking frames
dg = bpy.context.evaluated_depsgraph_get()
gaps = []
for f in range(sc.frame_start, sc.frame_start + 80, 4):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    zs = [((o.evaluated_get(dg)).matrix_world @ Vector(c)) for o in cm.characters._descendant_meshes(root)
          for c in o.evaluated_get(dg).bound_box]
    low = min(zs, key=lambda v: v.z)
    gaps.append(low.z - fz(low.x, low.y))
print(f"[collide] sole height above the floor over frames: median {np.median(gaps) * 100:+.1f} cm, "
      f"range {min(gaps) * 100:+.1f}..{max(gaps) * 100:+.1f} cm", flush=True)
