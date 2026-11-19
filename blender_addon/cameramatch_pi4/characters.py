# SPDX-License-Identifier: GPL-3.0-or-later
"""Imported rigged characters: walk-cycle analysis (speed, forward axis, sole height) and foot-slide measurement.

An in-place walk cycle encodes its own walking speed: while a foot is planted (lowest part of its arc), it
moves backward relative to the character at exactly the speed the character must travel to keep it still on
the ground. Measuring that, and using it as the path speed, removes foot sliding.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

FOOT_KEYS = ("foot", "toe", "ankle")


def hierarchy_root(ob):
    while ob.parent is not None:
        ob = ob.parent
    return ob


def find_armature(root):
    stack = [root]
    while stack:
        o = stack.pop()
        if o.type == "ARMATURE":
            return o
        stack.extend(o.children)
    return None


def foot_bones(arm):
    names = [b.name for b in arm.data.bones if any(k in b.name.lower() for k in FOOT_KEYS)]
    # prefer 'foot' bones (one per side); toes only if no foot bone exists
    feet = [n for n in names if "foot" in n.lower() and "end" not in n.lower()]
    return feet or names


def walk_action(arm, name=""):
    if name and name in bpy.data.actions:
        return bpy.data.actions[name]
    pool = list(bpy.data.actions)
    own = [bpy.data.actions[n] for n in arm.get("cm_actions", []) if n in bpy.data.actions]
    if own:                                          # actions imported with THIS character (Add Character)
        pool = own
        if len(own) == 1:                            # e.g. a Mixamo FBX: one 'mixamo.com' action = the walk
            return own[0]
    acts = [a for a in pool if "walk" in a.name.lower() and "jump" not in a.name.lower()]
    if acts:
        return sorted(acts, key=lambda a: len(a.name))[0]
    ad = arm.animation_data
    if ad and ad.action:
        return ad.action
    raise RuntimeError(f"no walk action found for '{arm.name}'")


def _assign(ad, action):
    ad.action = action
    if hasattr(ad, "action_slot") and getattr(action, "slots", None):
        try:
            ad.action_slot = action.slots[0]
        except (TypeError, AttributeError):
            pass


def _descendant_meshes(root):
    out, stack = [], [root]
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            out.append(o)
        stack.extend(o.children)
    return out


def analyse_walk(scene, root, arm, action):
    """Play `action` (temporarily, from frame 0) and measure the cycle. Returns a dict:
    speed_mpf (world m per action frame), forward (world unit xy at the root's current yaw), sole_above (root
    origin height above the lowest sole point), cycle (frames), root_motion (fraction of the walk distance the
    hips travel by themselves), feet."""
    ad = arm.animation_data or arm.animation_data_create()
    saved = (ad.action, [t.mute for t in ad.nla_tracks], scene.frame_current)
    for t in ad.nla_tracks:
        t.mute = True
    _assign(ad, action)
    a0, a1 = float(action.frame_range[0]), float(action.frame_range[1])
    clen = a1 - a0                                   # cycle length can be fractional (23.958 frames)
    nsamp = max(int(round(clen)), 4)
    hstep = clen / nsamp
    feet = foot_bones(arm)
    if not feet:
        raise RuntimeError(f"'{arm.name}': no foot bones (names containing foot / toe / ankle)")
    hips = next((b.name for b in arm.data.bones if b.parent is None), None)
    F, Hp, sole = [], [], []
    meshes = _descendant_meshes(root)
    dg = bpy.context.evaluated_depsgraph_get()
    try:
        for i_ in range(nsamp + 1):
            tt = a0 + i_ * hstep
            scene.frame_set(int(math.floor(tt)), subframe=tt - math.floor(tt))
            M = arm.matrix_world
            F.append([np.array((M @ arm.pose.bones[n].head)[:]) for n in feet])
            Hp.append(np.array((M @ arm.pose.bones[hips].head)[:]) if hips else np.zeros(3))
            zs = []
            for m in meshes:
                ev = m.evaluated_get(dg)
                zs += [(ev.matrix_world @ Vector(c)).z for c in ev.bound_box]
            sole.append(min(zs) if zs else min(p[2] for p in F[-1]))
    finally:
        _assign(ad, saved[0]) if saved[0] else setattr(ad, "action", None)
        for t, m in zip(ad.nla_tracks, saved[1]):
            t.mute = m
        scene.frame_set(saved[2])
    F = np.array(F)                                   # (frames, feet, 3)
    vel = []
    for k in range(F.shape[1]):
        z = F[:, k, 2]
        lo = z.min() + 0.15 * max(z.max() - z.min(), 1e-6)
        st = z <= lo
        for i in range(len(z) - 1):
            if st[i] and st[i + 1]:
                vel.append(F[i + 1, k, :2] - F[i, k, :2])
    if not vel:
        raise RuntimeError("could not find planted-foot phases in the walk cycle")
    v = np.median(np.array(vel), axis=0)              # planted foot moves BACKWARD at the walk speed
    speed = float(np.linalg.norm(v))
    fwd = -v / max(speed, 1e-9)
    Hp = np.array(Hp)
    hip_travel = float((Hp[-1, :2] - Hp[0, :2]) @ fwd)
    speed /= hstep                                     # per action frame
    cycle = clen
    root_motion = hip_travel / max(speed * cycle, 1e-9)
    # speed profile over the cycle: how far the PLANTED (lowest) foot moves backward between action frames.
    # Driving the character with it (not the average) keeps the foot still during the whole stance.
    prof = np.zeros(nsamp)                           # metres per ACTION FRAME, sampled every hstep frames
    for i in range(nsamp):
        k = int(np.argmin(F[i, :, 2]))
        prof[i] = max(0.0, -float((F[i + 1, k, :2] - F[i, k, :2]) @ fwd)) / hstep
    # sole height: the MEDIAN per-frame lowest point = the planted foot on most frames (min() would follow a
    # swing foot that dips below the planted one in stylised cycles and leave the planted foot hovering)
    return dict(speed_mpf=speed, forward=fwd, sole_above=float(root.matrix_world.translation.z - np.median(sole)),
                cycle=cycle, root_motion=root_motion, feet=feet, action=action.name, profile=prof,
                profile_step=hstep, profile_mean_mpf=float(prof.mean()), hips=hips,
                hip_travel_m=float(np.linalg.norm(Hp[-1, :2] - Hp[0, :2])))


def _fcurves(action):
    """All F-curves of an action (Blender 5 layered actions: layers > strips > channelbags)."""
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            for cb in getattr(strip, "channelbags", []):
                out += list(cb.fcurves)
    return out


OBJECT_TRANSFORM = ("location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle", "scale",
                    "delta_location", "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale")


def strip_object_transform(action):
    """Remove an action's keys on the OBJECT's own transform (glTF exporters key the armature node: e.g. three.js'
    Soldier). Placement belongs to Move Along Path; left in, they fight its keys and undo the character's scale.
    Bone animation (pose.bones[...]) is kept. Returns how many F-curves were removed."""
    n = 0
    if hasattr(action, "fcurves"):
        for fc in [f for f in action.fcurves if f.data_path in OBJECT_TRANSFORM]:
            action.fcurves.remove(fc)
            n += 1
        return n
    for layer in action.layers:
        for strip in layer.strips:
            for cb in getattr(strip, "channelbags", []):
                for fc in [f for f in cb.fcurves if f.data_path in OBJECT_TRANSFORM]:
                    cb.fcurves.remove(fc)
                    n += 1
    return n


def make_in_place(action, hips):
    """Copy of a walk with ROOT MOTION (e.g. a Mixamo walk downloaded without 'In Place') where the hips no
    longer travel: the linear drift of the hips' location over the cycle is removed, so the cycle loops on the
    spot and Move Along Path does the travelling. Returns the copy ('<name>_inplace')."""
    name = f"{action.name}_inplace"
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    new = action.copy()
    new.name = name
    path = f'pose.bones["{hips}"].location'
    n = 0
    for fc in _fcurves(new):
        if fc.data_path != path or len(fc.keyframe_points) < 2:
            continue
        kp = fc.keyframe_points
        t0, t1 = kp[0].co.x, kp[-1].co.x
        d = kp[-1].co.y - kp[0].co.y
        if abs(t1 - t0) < 1e-6:
            continue
        for k in kp:
            u = (k.co.x - t0) / (t1 - t0)
            k.co.y -= d * u
            k.handle_left.y -= d * u
            k.handle_right.y -= d * u
        fc.update()
        n += 1
    if n == 0:
        raise RuntimeError(f"'{action.name}': no location keys on '{hips}' to remove the root motion from")
    return new


def loop_action(scene, arm, action):
    """Play `action` on a loop over the scene range (NLA strip with repeats; other tracks muted)."""
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = None
    for t in ad.nla_tracks:
        t.mute = True
    old = ad.nla_tracks.get("CM_walk")
    if old:
        ad.nla_tracks.remove(old)
    tr = ad.nla_tracks.new()
    tr.name = "CM_walk"
    st = tr.strips.new(action.name, int(scene.frame_start), action)
    if hasattr(st, "action_slot") and getattr(action, "slots", None):
        try:
            st.action_slot = action.slots[0]
        except (TypeError, AttributeError):
            pass
    a0, a1 = action.frame_range
    st.repeat = max(1.0, (scene.frame_end - scene.frame_start + 1) / max(a1 - a0, 1) + 1)
    tr.mute = False
    return tr


def finish_at(scene, arm, track, info, rate, arrival):
    """The character reaches the end of its path at scene frame `arrival`: end the walk strip there and play an
    'idle' action afterwards if the rig has one (otherwise the last walk pose holds), so it never walks on the spot."""
    st = track.strips[0]
    walk_frames = max(arrival - scene.frame_start, 1)
    st.repeat = max(walk_frames * rate / info["cycle"], 0.05)
    idle = next((a for a in bpy.data.actions if "idle" in a.name.lower()), None)
    if idle is not None and arrival < scene.frame_end:
        s2 = track.strips.new(idle.name, int(st.frame_end) + 1, idle)
        if hasattr(s2, "action_slot") and getattr(idle, "slots", None):
            try:
                s2.action_slot = idle.slots[0]
            except (TypeError, AttributeError):
                pass
        i0, i1 = idle.frame_range
        s2.repeat = max(1.0, (scene.frame_end - arrival) / max(i1 - i0, 1) + 1)
        return idle.name
    st.extrapolation = "HOLD"
    return None


def arc_positions(info, scene, playback=1.0):
    """Arc length (m) travelled at each scene frame when the cycle plays at `playback` x its own rate and the
    character advances with the cycle's planted-foot speed profile."""
    prof, hs, clen = info["profile"], info["profile_step"], info["cycle"]
    n = len(prof)
    s, out = 0.0, []
    sub = 4                                           # integrate each scene frame in sub-steps
    for k in range(scene.frame_end - scene.frame_start + 1):
        out.append(s)
        for q in range(sub):
            ph = ((k + (q + 0.5) / sub) * playback) % clen    # action time within the cycle
            i = int(ph / hs) % n
            s += playback * prof[i] / sub
    return np.array(out)


def foot_slide(scene, arm, feet, f0=None, f1=None):
    """Measured slide of the planted (lowest) foot in world space over the scene range (cm/s, median and p90):
    horizontal motion between consecutive frames where the same foot is the lowest one."""
    f0 = scene.frame_start if f0 is None else f0
    f1 = scene.frame_end if f1 is None else f1
    fps = scene.render.fps / scene.render.fps_base
    P = []
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        M = arm.matrix_world
        P.append([np.array((M @ arm.pose.bones[n].head)[:]) for n in feet])
    P = np.array(P)
    sl = []
    low = np.argmin(P[:, :, 2], axis=1)
    for i in range(len(P) - 1):
        k = low[i]
        if low[i + 1] == k:
            sl.append(np.linalg.norm(P[i + 1, k, :2] - P[i, k, :2]) * fps * 100)
    sl = np.array(sl) if sl else np.array([np.nan])
    return dict(median_cm_s=float(np.median(sl)), p90_cm_s=float(np.percentile(sl, 90)), samples=int(len(sl)))
