"""Paths that start behind a wall are moved onto the street; paths past the far end of the floor are kept.

blender -b <scene with a built set and a curve 'CM_path'> --python tests/check_offfloor.py
"""
import importlib

import bpy
import numpy as np
from mathutils import Vector

cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
fz = lambda x, y: cm.set_builder.floor_z(sc, x, y)  # noqa: E731
sc["cm_movers"] = "{}"
kd, n = cm.collide.obstacle_tree(sc, fz)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[offfloor] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


P = cm.paths.curve_points(bpy.data.objects["CM_path"])[:, :2]
d0 = min(kd.find((p[0], p[1], 0.0))[2] for p in cm.collide.resample(P))
P2, dmin, _ = cm.collide.avoid(P, 0.85, sc, 0.3, 0.3, kd, log=print)
check("path through the wall moved onto the street", dmin >= 0.3, f"closest wall {d0:.2f} -> {dmin:.2f} m")
# a straight path down the street centre, running past the end of the visible floor
sc.frame_set(sc.frame_start)
cam = sc.camera.matrix_world
c = cam.translation.xy
f = (cam.to_3x3() @ Vector((0, 0, -1))).xy.normalized()
fl = bpy.data.objects["CM_floor_catcher"]
far = max((fl.matrix_world @ v.co).xy.dot(f) for v in fl.data.vertices) - c.dot(f)
Q = np.array([(c + f * t)[:] for t in np.linspace(2.0, far + 6.0, 12)])
Q2, dq, _ = cm.collide.avoid(Q, 0.85, sc, 0.3, 0.3, kd, log=print)
L = lambda X: float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())  # noqa: E731
check("path past the floor's far end kept", L(Q2) > 0.95 * L(Q), f"length {L(Q):.1f} -> {L(Q2):.1f} m (floor ends {far:.1f} m ahead)")
print("[offfloor]", "ALL PASS" if ok else "SOME FAILED", flush=True)


def max_turn(X, w=10):
    t = X[w:] - X[:-w]
    a = np.arctan2(t[:, 1], t[:, 0])
    da = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.degrees(da.max())) / (w * 0.05)   # degrees per metre-ish over 0.5 m chords


for nm, X in (("wall path", P2), ("far path", Q2)):
    print(f"[offfloor] {nm}: max turn {max_turn(X):.0f} deg/m, length {L(X):.1f} m", flush=True)
