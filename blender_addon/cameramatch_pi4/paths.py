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


def ground_path(xy, heading_window=0.4):
    """xy: (N, 2) ordered points -> (path(s) -> (Vector p, Vector unit tangent), total length).
    The tangent is taken over +-heading_window metres of arc length, so a finely resampled / nudged polyline
    does not make the heading jitter (a planted foot to the side of the root would slide with every wobble)."""
    xy = np.asarray(xy, np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy = xy[keep]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s_arc = np.r_[0, np.cumsum(seg)]

    def at(s):
        s = min(max(s, 0.0), s_arc[-1] - 1e-6)
        return np.array([np.interp(s, s_arc, xy[:, 0]), np.interp(s, s_arc, xy[:, 1])])

    def path(s):
        p = at(s)
        w = min(heading_window, s_arc[-1] / 2)
        a, b = at(s - w), at(s + w)
        if np.linalg.norm(b - a) < 1e-9:
            a, b = at(0.0), at(s_arc[-1])
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        return Vector((p[0], p[1])), Vector((t[0], t[1]))

    return path, float(s_arc[-1])


def move_one(ob, path, length, floor_fn, scene, speed, yaw_off, z_off):
    """Keyframe one object (hierarchy root) along the path over the scene range at `speed` m/s, or with `speed`
    an array of arc-length positions per frame (walk-cycle speed profile): origin at
    floor + z_off, heading = path tangent angle + yaw_off (X/Y tilt kept). Returns the per-frame track
    [(x, y, tangent_angle)] for collision bookkeeping."""
    f0, f1 = scene.frame_start, scene.frame_end
    fps = scene.render.fps / scene.render.fps_base
    rx, ry = ob.rotation_euler.x, ob.rotation_euler.y
    ob.rotation_mode = "XYZ"
    prev, track = None, []
    for fr in range(f0, f1 + 1):
        s_fr = speed[fr - f0] if hasattr(speed, "__len__") else speed * (fr - f0) / fps
        p, t = path(min(s_fr, length))
        th = math.atan2(t.y, t.x)
        yaw = th + yaw_off
        if prev is not None:                     # unwrap so the heading never spins through +-180
            yaw += round((prev - yaw) / (2 * math.pi)) * 2 * math.pi
        prev = yaw
        ob.location = (p.x, p.y, floor_fn(p.x, p.y) + z_off)
        ob.rotation_euler = (rx, ry, yaw)
        ob.keyframe_insert("location", frame=fr)
        ob.keyframe_insert("rotation_euler", frame=fr)
        track.append((p.x, p.y, th))
    return track


def footprint(root, forward_xy):
    """Half width / half length of the hierarchy's meshes on the ground, length measured along forward_xy."""
    pts = []
    stack = [root]
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            pts += [(o.matrix_world @ Vector(c))[:2] for c in o.bound_box]
        stack.extend(o.children)
    if not pts:
        return 0.3, 0.3
    P = np.array(pts) - np.array(root.matrix_world.translation[:2])
    f = np.asarray(forward_xy, np.float64)
    f = f / max(np.linalg.norm(f), 1e-9)
    side = np.array([-f[1], f[0]])
    hl = float(np.max(np.abs(P @ f)))
    hw = float(np.max(np.abs(P @ side)))
    return max(hw, 0.1), max(hl, 0.1)
