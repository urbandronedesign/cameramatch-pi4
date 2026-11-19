# SPDX-License-Identifier: GPL-3.0-or-later
"""Collision avoidance for paths on the fitted floor.

Static: the set (CM_walls low-poly mesh, CM_street_walls) and static CG meshes, sampled in the body-height band
above the floor into a 2D KD-tree; the path is pushed out to a clearance and smoothed.
Moving: every mover placed by the add-on (robot, Move Along Path) registers its per-frame footprint (oriented
rectangle) in scene["cm_movers"]; a new mover bends its path around the earlier ones, frame by frame.
"""
import json
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

BAND = (0.15, 2.0)        # metres above the floor that count as an obstacle (feet clear kerbs, heads clear signs)
MARGIN = 0.10


# --------------------------------------------------------------------------- static obstacles
def obstacle_tree(scene, floor_fn, exclude=(), step=0.08):
    pts = []
    mover_names = set(movers(scene))

    def add(xyz):
        z0 = floor_fn(xyz[0], xyz[1])
        if BAND[0] < xyz[2] - z0 < BAND[1]:
            pts.append((xyz[0], xyz[1], 0.0))

    for name in ("CM_walls", "CM_street_walls"):
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        M = ob.matrix_world
        W = [M @ v.co for v in ob.data.vertices]
        for v in W:
            add(v)
        for e in ob.data.edges:                       # densify along edges: walls must not have gaps
            a_, b_ = W[e.vertices[0]], W[e.vertices[1]]
            n_ = int((a_ - b_).length / step)
            for k in range(1, n_):
                add(a_.lerp(b_, k / n_))
    skip_coll = {"CM_Set"}
    for ob in scene.objects:
        if (ob.type != "MESH" or ob.hide_render or ob.name in exclude or ob.name in mover_names
                or ob.name.startswith(("Robot_", "Pi4_", "CM_")) or any(c.name in skip_coll for c in ob.users_collection)
                or (ob.animation_data and ob.animation_data.action)):
            continue
        bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        zlo, zhi = min(c.z for c in bb), max(c.z for c in bb)
        x0, x1 = min(c.x for c in bb), max(c.x for c in bb)
        y0, y1 = min(c.y for c in bb), max(c.y for c in bb)
        fz = floor_fn((x0 + x1) / 2, (y0 + y1) / 2)
        if zhi - fz < BAND[0] or zlo - fz > BAND[1]:
            continue
        for x in np.arange(x0, x1 + 1e-6, step):
            pts += [(x, y0, 0.0), (x, y1, 0.0)]
        for y in np.arange(y0, y1 + 1e-6, step):
            pts += [(x0, y, 0.0), (x1, y, 0.0)]
    if not pts:
        return None, 0
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    return kd, len(pts)


def resample(xy, step=0.05):
    xy = np.asarray(xy, np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(s[-1] / step), 2)
    t = np.linspace(0, s[-1], n + 1)
    return np.c_[np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]


def push_static(P, kd, clearance, iters=40, smooth=0.25):
    """Push path points out of the obstacles to `clearance`, with Laplacian smoothing in between."""
    if kd is None:
        return P, float("inf")
    P = P.copy()
    for it in range(iters):
        moved = 0
        for i in range(len(P)):
            co, _, d = kd.find((P[i, 0], P[i, 1], 0.0))
            if d < clearance:
                dv = P[i] - np.array(co[:2])
                n = np.linalg.norm(dv)
                if n > 1e-6:
                    P[i] += dv / n * (clearance - d) * 0.8
                    moved += 1
        if it < iters - 5:
            P[1:-1] = P[1:-1] * (1 - smooth) + smooth * 0.5 * (P[:-2] + P[2:])
        if moved == 0 and it >= 3:
            break
    dmin = min(kd.find((p[0], p[1], 0.0))[2] for p in P)
    return P, dmin


# --------------------------------------------------------------------------- moving obstacles (movers)
def movers(scene):
    try:
        reg = json.loads(scene.get("cm_movers", "{}"))
    except ValueError:
        reg = {}
    return {k: v for k, v in reg.items() if bpy.data.objects.get(k) is not None}


def register_mover(scene, name, f0, track, half_w, half_l):
    """track: list of (x, y, heading_rad) per frame from f0."""
    reg = movers(scene)
    reg[name] = dict(f0=int(f0), hw=float(half_w), hl=float(half_l),
                     track=[[round(float(a), 4) for a in t] for t in track])
    scene["cm_movers"] = json.dumps(reg)


def _rect_dist(p, c, h, hw, hl):
    """Distance from point p to an oriented rectangle (centre c, heading h = direction of its long axis)."""
    d = np.asarray(p) - np.asarray(c)
    ca, sa = math.cos(h), math.sin(h)
    along, side = d[0] * ca + d[1] * sa, -d[0] * sa + d[1] * ca
    ex, ey = abs(along) - hl, abs(side) - hw
    return math.hypot(max(ex, 0), max(ey, 0)) + min(max(ex, ey), 0)


def mover_gap(p, h, hw, hl, m, k):
    """Clearance between a footprint (p, h, hw, hl) and mover m at its frame index k (sampled both ways)."""
    if k < 0 or k >= len(m["track"]):
        return float("inf")
    qx, qy, qh = m["track"][k]

    def samples(c, hh, a, b):
        ca, sa = math.cos(hh), math.sin(hh)
        out = [c]
        for u, v in ((b, a), (b, -a), (-b, a), (-b, -a), (b, 0), (-b, 0), (0, a), (0, -a)):
            out.append((c[0] + u * ca - v * sa, c[1] + u * sa + v * ca))
        return out

    g1 = min(_rect_dist(s, (qx, qy), qh, m["hw"], m["hl"]) for s in samples(p, h, hw, hl))
    g2 = min(_rect_dist(s, p, h, hw, hl) for s in samples((qx, qy), qh, m["hw"], m["hl"]))
    return min(g1, g2)


def _normals(P):
    t = np.gradient(P, axis=0)
    k = max(1, int(0.4 / 0.05))                      # tangent over ~0.8 m (P is resampled every 5 cm)
    t = np.array([P[min(j + k, len(P) - 1)] - P[max(j - k, 0)] for j in range(len(P))])
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.c_[-t[:, 1], t[:, 0]]


def avoid(P, speed, scene, hw, hl, kd, self_name="", log=print, passes=8, width=1.0):
    """Bend path P (N, 2) so a mover of half-size (hw, hl) keeps MARGIN from the walls and from every registered
    mover. speed: m/s, an array of arc positions per frame, or a rule L -> arc positions per frame (when the
    mover's timing depends on the path length, e.g. 'the whole curve over the shot').

    Walls: points pushed out to a clearance, lightly smoothed. Movers: the path is offset SIDEWAYS (along its
    normal); for every conflicting frame the smallest lateral offset that clears the other mover AND keeps wall
    clearance is found by line search (both sides), and the requirements are combined with a smooth envelope
    (max, not sum). Offsets only grow, so a few passes converge.
    Returns (P, worst static clearance, worst mover gap)."""
    f0, f1 = scene.frame_start, scene.frame_end
    fps = scene.render.fps / scene.render.fps_base
    others = {k: v for k, v in movers(scene).items() if k != self_name}
    r_static = hw + MARGIN          # walls run alongside the path: the half WIDTH is what must clear them
    base = resample(P)
    base, dmin = push_static(base, kd, r_static)
    if not others:
        return base, dmin, float("inf")
    nrm = _normals(base)
    off = np.zeros(len(base))
    s_base = np.r_[0, np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))]

    def wall_ok(q):
        return kd is None or kd.find((q[0], q[1], 0.0))[2] >= r_static - 1e-3

    def arc(Pc):
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]

    def frames_s(Pc):
        L = arc(Pc)[-1]
        if callable(speed):
            return np.minimum(speed(L), L), L
        if hasattr(speed, "__len__"):
            return np.minimum(np.asarray(speed, np.float64), L), L
        return np.minimum(speed * np.arange(f1 - f0 + 1) / fps, L), L

    worst = float("inf")
    for it in range(passes):
        cur = base + off[:, None] * nrm
        s_cur = arc(cur)
        s_fr, _ = frames_s(cur)
        worst = float("inf")
        need = []                                    # (index on the path, required offset)
        for k, fr in enumerate(range(f0, f1 + 1)):
            j = int(np.clip(np.searchsorted(s_cur, s_fr[k]), 0, len(cur) - 1))
            p = cur[j]
            jj = (max(j - 8, 0), min(j + 8, len(cur) - 1))
            t = cur[jj[1]] - cur[jj[0]]
            h = math.atan2(t[1], t[0])
            for m in others.values():
                g = mover_gap(p, h, hw, hl, m, fr - m["f0"])
                worst = min(worst, g)
                if g >= MARGIN:
                    continue
                best = None
                for d in np.arange(0.05, 3.0, 0.05):     # smallest sideways shift (either side) that clears both
                    for sg in (1.0, -1.0):
                        q = base[j] + (off[j] + sg * d) * nrm[j]
                        if mover_gap(q, h, hw, hl, m, fr - m["f0"]) >= MARGIN + 0.02 and wall_ok(q):
                            best = off[j] + sg * d
                            break
                    if best is not None:
                        break
                if best is not None:
                    need.append((j, best))
        if not need:
            break
        env = off.copy()
        for j, o in need:
            w = np.exp(-0.5 * ((s_base - s_base[j]) / width) ** 2)
            cand = o * w
            take = np.abs(cand) > np.abs(env)
            env[take] = cand[take]
        off = env
    final = base + off[:, None] * nrm
    if kd is not None:
        dmin = min(kd.find((q[0], q[1], 0.0))[2] for q in final)
    log(f"collision avoidance: {it + 1} pass(es), worst mover gap {worst:+.2f} m (target {MARGIN:.2f})")
    return final, dmin, worst
