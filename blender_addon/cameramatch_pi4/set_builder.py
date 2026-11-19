# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the CG-integration set from a Pi4 solve: fitted floor (shadow catcher), plate-projected walls,
optional street proxy walls, plate environment world, and the Cycles render setup for Render & Comp.

Rules learned on real shots (see docs/PROJECT_GUIDE.md):
- every real-world proxy is a shadow catcher, otherwise its occlusion of the sky darkens the plate;
- the scanned set stops at the frame edges: close it (proxy walls) or the CG is lit by open sky.
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector

SET_COLL = "CM_Set"


# --------------------------------------------------------------------------- floor
def floor_eval(coef, x, y):
    a, b, c, d, e, f = coef
    return a + b * x + c * y + d * x * x + e * x * y + f * y * y


def floor_z(scene, x, y):
    """Fitted floor height at (x, y), from the last Build Set."""
    coef = scene.get("cm_floor_coef")
    if coef is None:
        raise RuntimeError("no fitted floor: run Build Set + Lighting first")
    return float(floor_eval(list(coef), x, y))


def _design(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.c_[np.ones(len(x)), x, y, x * x, x * y, y * y]


def _robust_fit(xy, z, iters=8, floor_sigma=0.015):
    keep = np.ones(len(z), bool)
    A = _design(xy)
    coef, sig = np.zeros(6), 1.0
    for _ in range(iters):
        if keep.sum() < 12:
            break
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < max(2.5 * sig, floor_sigma)
    return coef, sig, keep


def fit_floor(P, cam_z, log):
    """Robust quadratic floor z = f(x, y) from world points (Z up). Returns coef, sigma, inlier points."""
    Q = P[P[:, 2] < cam_z - 0.5]
    if len(Q) < 500:
        raise RuntimeError("too few points below the camera to fit a floor")
    # lowest decile per 0.5 m cell -> candidate floor samples (walls/objects mostly rejected)
    cell = np.floor(Q[:, :2] / 0.5).astype(np.int64)
    key = cell[:, 0] * 1_000_003 + cell[:, 1]
    order = np.lexsort((Q[:, 2], key))
    ks = key[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    counts = np.diff(np.r_[starts, len(ks)])
    ok = counts >= 15
    pick = order[starts[ok] + (counts[ok] * 0.1).astype(int)]
    C = Q[pick]
    coef, sig, keep = _robust_fit(C[:, :2], C[:, 2], floor_sigma=0.05)
    if np.any(np.abs(coef[3:]) > 0.05):                    # unstable curvature: fall back to a plane
        coef = np.r_[np.linalg.lstsq(_design(C[keep, :2])[:, :3], C[keep, 2], rcond=None)[0], 0, 0, 0]
    near = np.abs(Q[:, 2] - _design(Q[:, :2]) @ coef) < 0.08
    coef, sig, keep = _robust_fit(Q[near, :2], Q[near, 2])
    if np.any(np.abs(coef[3:]) > 0.05):
        coef3 = np.linalg.lstsq(_design(Q[near][keep, :2])[:, :3], Q[near][keep, 2], rcond=None)[0]
        coef = np.r_[coef3, 0, 0, 0]
    inl = Q[near][keep]
    log(f"floor fit: {len(inl)} inliers, robust sigma {sig * 100:.1f} cm, "
        f"slope {math.degrees(math.atan(math.hypot(coef[1], coef[2]))):.1f} deg")
    return coef, sig, inl


def _floor_mesh(coef, inl, cell=0.25, dilate=2):
    """Grid mesh over the floor's support region (cells holding inliers, dilated)."""
    ij = np.floor(inl[:, :2] / cell).astype(int)
    occ = set(map(tuple, ij))
    grown = set()
    for i, j in occ:
        for di in range(-dilate, dilate + 1):
            for dj in range(-dilate, dilate + 1):
                grown.add((i + di, j + dj))
    vid, verts, faces = {}, [], []

    def v(i, j):
        if (i, j) not in vid:
            x, y = i * cell, j * cell
            vid[(i, j)] = len(verts)
            verts.append((x, y, floor_eval(coef, x, y)))
        return vid[(i, j)]

    for i, j in grown:
        faces.append((v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)))
    me = bpy.data.meshes.new("CM_floor_catcher")
    me.from_pydata(verts, [], faces)
    me.update()
    return me


# --------------------------------------------------------------------------- materials
def projector_material(name, cam, clip, f_px, k1, k2, emissive, albedo_gain=1.0):
    """The plate projected from the solved camera (fallback: vertex colour 'Col' where it never sees)."""
    W, H = clip.size
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links

    def mth(op, a, b=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for i, val in enumerate((a, b)):
            if val is None:
                continue
            if isinstance(val, (int, float)):
                n.inputs[i].default_value = val
            else:
                L.new(val, n.inputs[i])
        return n.outputs[0]

    tc = N.new("ShaderNodeTexCoord")
    tc.object = cam
    sp = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sp.inputs[0])
    nz = mth("MULTIPLY", sp.outputs["Z"], -1.0)
    x_ = mth("DIVIDE", sp.outputs["X"], nz)
    y_ = mth("DIVIDE", sp.outputs["Y"], nz)
    q_ = mth("ADD", mth("MULTIPLY", x_, x_), mth("MULTIPLY", y_, y_))
    dd = mth("ADD", mth("ADD", 1.0, mth("MULTIPLY", q_, k1)), mth("MULTIPLY", mth("MULTIPLY", q_, q_), k2))
    uu = mth("ADD", mth("DIVIDE", mth("MULTIPLY", mth("MULTIPLY", x_, dd), f_px), W), 0.5)
    vv = mth("ADD", mth("DIVIDE", mth("MULTIPLY", mth("MULTIPLY", y_, dd), f_px), H), 0.5)
    cb = N.new("ShaderNodeCombineXYZ")
    L.new(uu, cb.inputs["X"])
    L.new(vv, cb.inputs["Y"])
    img = bpy.data.images.get("CM_plate") or bpy.data.images.load(bpy.path.abspath(clip.filepath))
    img.name = "CM_plate"
    img.source = "MOVIE" if clip.source == "MOVIE" else "SEQUENCE"
    tx = N.new("ShaderNodeTexImage")
    tx.image = img
    tx.extension = "CLIP"
    tx.image_user.frame_duration = clip.frame_duration
    tx.image_user.frame_start = clip.frame_start
    tx.image_user.use_auto_refresh = True
    L.new(cb.outputs[0], tx.inputs["Vector"])
    vis = mth("MULTIPLY", tx.outputs["Alpha"], mth("GREATER_THAN", nz, 1e-4))
    vc = N.new("ShaderNodeAttribute")
    vc.attribute_name = "Col"
    mx = N.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    L.new(vis, mx.inputs["Factor"])
    L.new(vc.outputs["Color"], mx.inputs["A"])
    L.new(tx.outputs["Color"], mx.inputs["B"])
    o = N.new("ShaderNodeOutputMaterial")
    if emissive:
        em = N.new("ShaderNodeEmission")
        L.new(mx.outputs["Result"], em.inputs["Color"])
        L.new(em.outputs[0], o.inputs["Surface"])
    else:
        g = N.new("ShaderNodeMix")
        g.data_type = "RGBA"
        g.blend_type = "MULTIPLY"
        g.inputs["Factor"].default_value = 1.0
        L.new(mx.outputs["Result"], g.inputs["A"])
        g.inputs["B"].default_value = (albedo_gain, albedo_gain, albedo_gain, 1)
        df = N.new("ShaderNodeBsdfDiffuse")
        L.new(g.outputs["Result"], df.inputs["Color"])
        L.new(df.outputs[0], o.inputs["Surface"])
    return mat


def _set_colours(me, cols_lin):
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.c_[cols_lin, np.ones(len(cols_lin))].astype(np.float32).ravel())


# --------------------------------------------------------------------------- street proxy walls
def detect_street(P, coef, cams, log, wall_height=14.0):
    """Two street walls along the camera's travel direction, or None. Returns (verts, faces) of proxy walls."""
    c0 = cams[0]
    travel = cams[-1] - cams[0]
    travel[2] = 0
    fwd = travel / np.linalg.norm(travel) if np.linalg.norm(travel) > 0.5 else None
    if fwd is None:
        return None, "camera barely moves: no street axis"
    n = np.array([fwd[1], -fwd[0], 0.0])
    h = P[:, 2] - floor_eval(coef, P[:, 0], P[:, 1])
    M = P[(h > 0.4) & (h < 3.0)]
    v = (M - c0) @ fwd
    u = (M - c0) @ n
    vs, ls, rs = [], [], []
    for v0 in np.arange(np.floor(v.min()), np.ceil(v.max()), 1.0):
        sel = (v >= v0) & (v < v0 + 1)
        if sel.sum() < 150:
            continue
        lo, hi = np.percentile(u[sel], 2), np.percentile(u[sel], 98)
        if lo < -0.3 and hi > 0.3 and 1.0 < hi - lo < 25:
            vs.append(v0 + 0.5)
            ls.append(lo)
            rs.append(hi)
    if len(vs) < 4:
        return None, f"no street walls detected ({len(vs)} wall bins)"
    cl, cr = np.polyfit(vs, ls, 2 if len(vs) > 6 else 1), np.polyfit(vs, rs, 2 if len(vs) > 6 else 1)
    verts, faces = [], []
    vgrid = np.linspace(min(vs) - 30.0, max(vs) + 20.0, 141)
    zs = np.linspace(-0.6, wall_height, 30)
    for sgn, cf in ((-1, cl), (1, cr)):
        base = len(verts)
        for vv in vgrid:
            u_ = float(np.polyval(cf, min(max(vv, vs[0]), vs[-1]))) + sgn * 0.08
            p = c0 + fwd * vv + n * u_
            fz = floor_eval(coef, p[0], p[1])
            for z in zs:
                verts.append((p[0], p[1], fz + z))
        nzs = len(zs)
        for j in range(len(vgrid) - 1):
            for i in range(nzs - 1):
                a = base + j * nzs + i
                faces.append((a, a + nzs, a + nzs + 1, a + 1))
    width = float(np.median(np.array(rs) - np.array(ls)))
    return (verts, faces), f"street walls: {len(vs)} bins, width ~{width:.1f} m, height {wall_height:.0f} m"


# --------------------------------------------------------------------------- main entry
def _job_scale(job_dir):
    import json
    try:
        with open(os.path.join(job_dir, "scene_scale.json")) as fh:
            return float(json.load(fh)["scale"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1.0


def _clear_set():
    coll = bpy.data.collections.get(SET_COLL)
    if coll:
        for ob in list(coll.objects):
            me = ob.data
            bpy.data.objects.remove(ob)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        bpy.data.collections.remove(coll)
    for m in [m for m in bpy.data.materials if m.name.startswith("CM_") and m.users == 0]:
        bpy.data.materials.remove(m)


def build_set(scene, clip, job_dir, log, street_walls=True, wall_height=14.0):
    cam = scene.camera
    if cam is None:
        raise RuntimeError("scene has no camera: solve first")
    r = np.load(os.path.join(job_dir, "result.npz"))
    W, H = clip.size
    f_px = float(r["f_px"]) * W / int(r["plate_size"][0]) if "f_px" in r else \
        cam.data.lens / cam.data.sensor_width * W
    k1 = float(r["k1"]) if "k1" in r else 0.0
    k2 = float(r["k2"]) if "k2" in r else 0.0
    _clear_set()
    coll = bpy.data.collections.new(SET_COLL)
    scene.collection.children.link(coll)

    # points: the imported Pi4 cloud
    po = next((o for o in scene.objects if o.name.endswith("_points") and o.type == "MESH"), None)
    if po is None:
        raise RuntimeError("no Pi4 point cloud in the scene (re-import the solve)")
    Mw = np.array(po.matrix_world)
    V = np.empty(len(po.data.vertices) * 3, np.float32)
    po.data.vertices.foreach_get("co", V)
    P = V.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
    cams = []
    for fr in range(scene.frame_start, scene.frame_end + 1, max(1, (scene.frame_end - scene.frame_start) // 40)):
        scene.frame_set(fr)
        cams.append(np.array(cam.matrix_world.translation))
    cams = np.array(cams)
    scene.frame_set(scene.frame_start)

    coef, sig, inl = fit_floor(P, float(np.median(cams[:, 2])), log)
    scene["cm_floor_coef"] = [float(c) for c in coef]
    scene["cm_floor_sigma"] = float(sig)

    # floor shadow catcher (plate-projected albedo -> bounce light)
    fme = _floor_mesh(coef, inl)
    _set_colours(fme, np.full((len(fme.vertices), 3), 0.1))
    fme.materials.append(projector_material("CM_floor_projected", cam, clip, f_px, k1, k2, False, 1.6))
    floor = bpy.data.objects.new("CM_floor_catcher", fme)
    coll.objects.link(floor)
    floor.is_shadow_catcher = True
    log(f"floor catcher: {len(fme.polygons)} cells")

    # walls: low-poly Pi4 mesh minus the floor, emitting the projected plate
    lp_path = os.path.join(job_dir, "lowpoly.npz")
    wall_rgb = np.array([0.06, 0.055, 0.05])
    if os.path.isfile(lp_path):
        lp = np.load(lp_path)
        Vl, Fl, Cl = lp["verts"] * _job_scale(job_dir), lp["faces"], lp["colors"]
        me = bpy.data.meshes.new("CM_walls")
        me.vertices.add(len(Vl))
        me.vertices.foreach_set("co", Vl.ravel())
        me.loops.add(Fl.size)
        me.loops.foreach_set("vertex_index", Fl.ravel())
        me.polygons.add(len(Fl))
        me.polygons.foreach_set("loop_start", np.arange(0, Fl.size, 3))
        me.update(calc_edges=True)
        lin = np.where(Cl <= 0.04045, Cl / 12.92, ((Cl + 0.055) / 1.055) ** 2.4)
        wall_rgb = np.median(lin, axis=0)
        _set_colours(me, lin)
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        gone = [fc for fc in bm.faces if abs(fc.normal.z) > 0.7 and
                abs(fc.calc_center_median().z - floor_eval(coef, *fc.calc_center_median().xy)) < 0.35]
        bmesh.ops.delete(bm, geom=gone, context="FACES")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(projector_material("CM_walls_projected", cam, clip, f_px, k1, k2, True))
        walls = bpy.data.objects.new("CM_walls", me)
        coll.objects.link(walls)
        walls.visible_camera = False
        walls.is_shadow_catcher = True
        log(f"walls: {len(me.polygons)} faces from the low-poly set mesh")
    else:
        log("no lowpoly.npz: walls skipped (run Build Set with the matte server up)")

    if street_walls:
        geo, msg = detect_street(P, coef, cams, log, wall_height)
        log(msg)
        if geo is not None:
            pm = bpy.data.meshes.new("CM_street_walls")
            pm.from_pydata(geo[0], [], geo[1])
            pm.update()
            _set_colours(pm, np.tile(wall_rgb, (len(pm.vertices), 1)))
            mat = bpy.data.materials.get("CM_walls_projected") or \
                projector_material("CM_walls_projected", cam, clip, f_px, k1, k2, True)
            pm.materials.append(mat)
            pw = bpy.data.objects.new("CM_street_walls", pm)
            coll.objects.link(pw)
            pw.visible_camera = False
            pw.is_shadow_catcher = True

    # world: plate environment, no added lights
    env_path = os.path.join(job_dir, "plate_env.png")
    if os.path.isfile(env_path):
        w = scene.world or bpy.data.worlds.new("World")
        scene.world = w
        if w.node_tree is None:
            w.use_nodes = True
        wn = w.node_tree
        wn.nodes.clear()
        env = wn.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(env_path, check_existing=True)
        bg = wn.nodes.new("ShaderNodeBackground")
        wn.links.new(env.outputs["Color"], bg.inputs["Color"])
        wo = wn.nodes.new("ShaderNodeOutputWorld")
        wn.links.new(bg.outputs["Background"], wo.inputs["Surface"])
        log("world: plate environment (no added lights)")
    else:
        log("no plate_env.png: world unchanged")
    for ob in scene.objects:
        if ob.name.endswith(("_points", "_tracks")):
            ob.hide_render = True
    return coef


# --------------------------------------------------------------------------- render setup
def setup_render(scene, out_dir, samples=192, motion_blur=True, exposure=0.5, log=print):
    scene.render.engine = "CYCLES"
    try:
        cp = bpy.context.preferences.addons["cycles"].preferences
        for dev in ("OPTIX", "CUDA"):
            cp.compute_device_type = dev
            cp.get_devices()
            if any(d.type == dev for d in cp.devices):
                for d in cp.devices:
                    d.use = d.type == dev
                scene.cycles.device = "GPU"
                break
    except Exception:
        pass
    scene.cycles.samples = samples
    scene.cycles.adaptive_threshold = 0.008
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.use_motion_blur = motion_blur
    scene.render.motion_blur_shutter = exposure
    # Match the plate's rolling shutter (solved readout, fraction of a frame). Cycles: the shutter window is
    # readout + per-row exposure, each row exposed for rolling_shutter_duration of it, top row first.
    rs = float(scene.get("cm_rolling_shutter", 0.0))
    axis = scene.get("cm_rolling_shutter_axis", "rows")
    try:
        scene.cycles.rolling_shutter_type = "NONE"
        if abs(rs) > 0.05:
            if axis == "rows" and rs > 0:
                scene.render.use_motion_blur = True
                win = rs + (exposure if motion_blur else 0.02)
                scene.render.motion_blur_shutter = win
                scene.cycles.rolling_shutter_type = "TOP"
                scene.cycles.rolling_shutter_duration = (exposure if motion_blur else 0.02) / win
                log(f"render: rolling shutter {rs:.2f} frame (top first), shutter window {win:.2f}")
            else:
                log(f"render: plate rolling shutter {rs:+.2f} ({axis}) cannot be reproduced by Cycles "
                    "(top-to-bottom rows only): CG rendered with a global shutter")
    except AttributeError:
        pass
    scene.view_settings.view_transform = "Standard"    # plate-consistent (sRGB in = sRGB out)
    scene.view_settings.look = "None"
    scene.render.resolution_percentage = 100
    vl = scene.view_layers[0]
    vl.cycles.use_pass_shadow_catcher = True
    scene.render.use_compositing = False
    im = scene.render.image_settings
    if hasattr(im, "media_type"):
        im.media_type = "MULTI_LAYER_IMAGE"
    im.file_format = "OPEN_EXR_MULTILAYER"
    im.color_depth = "16"
    im.exr_codec = "DWAA"
    os.makedirs(out_dir, exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, "r_")
