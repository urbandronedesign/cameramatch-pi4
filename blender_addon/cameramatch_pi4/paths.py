# SPDX-License-Identifier: GPL-3.0-or-later
"""Paths on the fitted floor: a Blender curve as an arc-length parameterised ground path, and a generic
'move along path' animator for any object (vehicles, characters with an in-place walk cycle...)."""
import math

import bpy
import numpy as np
from mathutils import Vector

FORWARD_YAW = {"-Y": math.pi / 2, "+Y": -math.pi / 2, "+X": 0.0, "-X": math.pi}


def curve_points(curve_obj):
    """Ordered world-space points along the curve's first spline (evaluated: resolution, modifiers)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = curve_obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        pts = np.array([(curve_obj.matrix_world @ v.co)[:] for v in me.vertices], np.float64)
        if len(me.edges):                          # follow the edge chain from an end vertex (robust ordering)
            nb = {}
            for e in me.edges:
                a, b = e.vertices
                nb.setdefault(a, []).append(b)
                nb.setdefault(b, []).append(a)
            start = next((k for k, v in nb.items() if len(v) == 1), 0)     # an open end, else any vertex
            order, seen, cur = [start], {start}, start
            while True:
                nxt = [n for n in nb.get(cur, []) if n not in seen]
                if not nxt:
                    break
                cur = nxt[0]
                seen.add(cur)
                order.append(cur)
            pts = pts[order]
    finally:
        ev.to_mesh_clear()
    if len(pts) < 2:
        raise RuntimeError(f"curve '{curve_obj.name}' has no length")
    # orient from the end nearest the curve's first control point
    sp = curve_obj.data.splines[0]
    p0 = sp.bezier_points[0].co if sp.type == "BEZIER" and len(sp.bezier_points) else sp.points[0].co
    p0w = np.array((curve_obj.matrix_world @ Vector(p0[:3]))[:])
    if np.linalg.norm(pts[-1] - p0w) < np.linalg.norm(pts[0] - p0w):
        pts = pts[::-1]
    return pts


def ground_path(xy):
    """xy: (N, 2) ordered points -> (path(s) -> (Vector p, Vector unit tangent), total length)."""
    xy = np.asarray(xy, np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy = xy[keep]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s_arc = np.r_[0, np.cumsum(seg)]

    def path(s):
        s = min(max(s, 0.0), s_arc[-1] - 1e-6)
        i = int(np.clip(np.searchsorted(s_arc, s) - 1, 0, len(seg) - 1))
        u = (s - s_arc[i]) / seg[i]
        p = xy[i] * (1 - u) + xy[i + 1] * u
        # smooth tangent: average over a short window so polyline corners do not snap the heading
        a = xy[max(i - 2, 0)]
        b = xy[min(i + 3, len(xy) - 1)]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        return Vector((p[0], p[1])), Vector((t[0], t[1]))

    return path, float(s_arc[-1])


def move_along(objs, path, length, floor_fn, scene, speed, forward="-Y", log=print):
    """Keyframe each object's location (on the floor, lowest point touching it) and heading along the path over
    the scene frame range. speed <= 0: the whole path over the range. Returns the speed used (m/s)."""
    f0, f1 = scene.frame_start, scene.frame_end
    fps = scene.render.fps / scene.render.fps_base
    dur = max((f1 - f0) / fps, 1e-3)
    v = length / dur if speed <= 0 else min(speed, length / dur)
    yaw0 = FORWARD_YAW[forward]
    for ob in objs:
        # height of the origin above the object's lowest point (local bbox, current scale, no rotation tilt)
        zmin = min((ob.matrix_basis.to_3x3() @ Vector(c)).z for c in ob.bound_box) if ob.type != "EMPTY" else 0.0
        rx, ry = ob.rotation_euler.x, ob.rotation_euler.y
        ob.rotation_mode = "XYZ"
        prev = None
        for fr in range(f0, f1 + 1):
            p, t = path(v * (fr - f0) / fps)
            yaw = math.atan2(t.y, t.x) + yaw0
            if prev is not None:                 # unwrap so the heading never spins through +-180
                yaw += round((prev - yaw) / (2 * math.pi)) * 2 * math.pi
            prev = yaw
            ob.location = (p.x, p.y, floor_fn(p.x, p.y) - zmin)
            ob.rotation_euler = (rx, ry, yaw)
            ob.keyframe_insert("location", frame=fr)
            ob.keyframe_insert("rotation_euler", frame=fr)
    log(f"moved {len(objs)} object(s) along {length:.1f} m at {v:.2f} m/s over frames {f0}-{f1}")
    return v
