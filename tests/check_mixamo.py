"""Mixamo characters through Add Character + Walk to Cursor on a solved shot with a built set (headless).

blender -b <scene with CM_Set>.blend --python tests/check_mixamo.py -- <character file> <out_dir> [expect]
expect: "walk" (default) or "not_walk" (an animation that is not a walk cycle must be refused with a clear message)
Checks: imported and scaled to 1.75 m, walks from the cursor 5 m ahead to 12 m ahead, soles on the floor,
planted feet not sliding, body facing the direction of travel; renders 3 frames at 50 % for review
(<out_dir>/r_####.exr) and saves <out_dir>/scene.blend.
"""
import importlib
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
expect = argv[2] if len(argv) > 2 else "walk"
os.makedirs(out, exist_ok=True)
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
name = os.path.basename(path)
ok = True
res = {"file": name}


def check(label, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[mixamo] {'PASS' if cond else 'FAIL'} {name}: {label} {info}", flush=True)


class Op:
    def report(self, t, m):
        print("[mixamo] report", m, flush=True)


# isolate: remove earlier characters / robots, empty the mover registry
for c in [c for c in bpy.data.collections if c.name.startswith(("CM_Character", "CM_Robot"))]:
    for o in list(c.all_objects):
        bpy.data.objects.remove(o)
    bpy.data.collections.remove(c)
for o in [o for o in bpy.data.objects if o.name.startswith("CM_path")]:
    bpy.data.objects.remove(o)
sc["cm_movers"] = "{}"
sc.cameramatch.char_height = 1.75
cm.hide_camera_blockers(sc)

sc.frame_set(sc.frame_start)
cam = sc.camera.matrix_world
c0 = cam.translation.xy.copy()
fwd = (cam.to_3x3() @ Vector((0, 0, -1))).xy.normalized()
a = c0 + fwd * 5.0
sc.cursor.location = (a.x, a.y, cm.set_builder.floor_z(sc, a.x, a.y))
r = cm.add_character(Op(), bpy.context, path)
check("Add Character", r == {"FINISHED"})
root = bpy.context.view_layer.objects.active
arm = cm.characters.find_armature(root)
meshes = [o for o in bpy.data.objects if o.type == "MESH" and cm.characters.hierarchy_root(o) is root
          and not o.hide_render]
bpy.context.view_layer.update()
zs = [(o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box]
check("scaled to 1.75 m", abs((max(zs) - min(zs)) - 1.75) < 0.05, f"{max(zs) - min(zs):.2f} m")

b = c0 + fwd * 12.0
sc.cursor.location = (b.x, b.y, cm.set_builder.floor_z(sc, b.x, b.y))
for o in sc.objects:
    o.select_set(o is root or o is arm)
bpy.context.view_layer.objects.active = root
n_log = len(cm._STATUS["log"])
try:
    with bpy.context.temp_override(selected_objects=[root, arm], active_object=root):
        r = bpy.ops.cameramatch.walk_to_cursor()
except RuntimeError as e:
    r = {"ERROR": str(e)}
new_log = cm._STATUS["log"][n_log:]
for ln in new_log:
    print("[mixamo] LOG", ln[9:220], flush=True)
if expect == "not_walk":
    msg = " ".join(new_log)
    check("refused with a clear message", r != {"FINISHED"} and "does not look like a walk cycle" in msg,
          str(r)[:160])
    print("[mixamo]", "ALL PASS" if ok else "SOME FAILED", flush=True)
    sys.exit(0)
check("Walk to Cursor", r == {"FINISHED"}, str(r)[:200])

# per-frame measurements
dg = bpy.context.evaluated_depsgraph_get()
gaps, pos, facing = [], [], []
info = None
for f in range(sc.frame_start, sc.frame_end + 1, 2):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    low = []
    for m in meshes:
        ev = m.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        M = np.array(ev.matrix_world)
        W = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        low.append(W[np.argmin(W[:, 2])])
        ev.to_mesh_clear()
    p = min(low, key=lambda q: q[2])
    gaps.append(float(p[2] - cm.set_builder.floor_z(sc, p[0], p[1])))
    hips = arm.pose.bones[cm.characters.foot_bones(arm)[0]].id_data.matrix_world @ \
        arm.pose.bones[next(bb.name for bb in arm.data.bones if bb.parent is None)].head
    pos.append(np.array(hips[:2]))
    # body forward: the direction from the left foot to the right foot, turned 90 deg (left-right axis x up)
    fb = cm.characters.foot_bones(arm)
    lf = next(n for n in fb if "left" in n.lower() or n.endswith(("L", ".L", "_L", "_l")))
    rf = next(n for n in fb if n != lf and ("right" in n.lower() or n.endswith(("R", ".R", "_R", "_r"))))
    # left-right axis of the body: the feet's rest-pose offset, turned with their common parent bone (works for
    # rigs without legs too, e.g. RobotExpressive's feet hang off the body bone)
    bl, br = arm.data.bones[lf], arm.data.bones[rf]
    anc_l = [bl.parent]
    while anc_l[-1] is not None:
        anc_l.append(anc_l[-1].parent)
    anc = br.parent
    while anc is not None and anc not in anc_l:
        anc = anc.parent
    off = bl.head_local - br.head_local
    if anc is not None:
        Ra = (arm.matrix_world @ arm.pose.bones[anc.name].matrix).to_3x3() @ anc.matrix_local.to_3x3().inverted()
    else:
        Ra = arm.matrix_world.to_3x3()
    v = Ra @ off
    sh_l, sh_r = v, v * 0
    lr = np.array((sh_l - sh_r).xy[:])                      # points to the character's left
    facing.append(np.array([lr[1], -lr[0]]) / max(np.linalg.norm(lr), 1e-9))   # left rotated -90 = forward
pos = np.array(pos)
travel = float(np.linalg.norm(pos[-1] - pos[0]))
gaps = np.array(gaps)
check("walks toward the cursor", travel > 3.0, f"travelled {travel:.1f} m")
check("soles on the floor", abs(np.median(gaps)) < 0.03, f"lowest point above floor: median {np.median(gaps) * 100:+.1f} cm, "
      f"5-95% {np.percentile(gaps, 5) * 100:+.1f}..{np.percentile(gaps, 95) * 100:+.1f} cm")
errs = []
for k in range(2, len(pos) - 2):
    mv = pos[k + 2] - pos[k - 2]
    if np.linalg.norm(mv) > 0.05:
        mv /= np.linalg.norm(mv)
        errs.append(math.degrees(math.acos(float(np.clip(mv @ facing[k], -1, 1)))))
check("faces the direction of travel", errs and np.median(errs) < 20, f"median {np.median(errs):.0f} deg" if errs else "")
sl = [ln for ln in new_log if "planted-foot slide" in ln]
v = float(sl[-1].split("slide ")[1].split(" cm/s")[0]) if sl else 1e9
check("planted feet do not slide", v < 10.0, f"{v:.1f} cm/s median")
res.update(travel=travel, gap_med_cm=float(np.median(gaps) * 100), facing_err=float(np.median(errs)) if errs else None,
           slide=v)

# review render: 3 frames, 50 %, 32 samples
cm.set_builder.setup_render(sc, out, 32, True, log=lambda *a: None, percentage=50)
sc.render.use_persistent_data = True
for f in (sc.frame_start + 15, (sc.frame_start + sc.frame_end) // 2, sc.frame_end - 10):
    sc.frame_set(f)
    sc.render.filepath = os.path.join(out, f"r_{f:04d}")
    bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene.blend"))
with open(os.path.join(out, "result.json"), "w") as fh:
    json.dump(res, fh)
print("[mixamo]", "ALL PASS" if ok else "SOME FAILED", flush=True)
