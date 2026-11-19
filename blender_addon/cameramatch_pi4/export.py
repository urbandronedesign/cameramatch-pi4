# SPDX-License-Identifier: GPL-3.0-or-later
"""Export the solved camera and scene for other software (DCCs).

Writes into one folder:
  <name>.abc    Alembic: camera (animated transform + focal), set meshes, track points, locators [+ CG]
  <name>.fbx    FBX (Y up, -Z forward, keys on every frame, nothing simplified)
  <name>.usdc   USD (Y up)
  <name>.nk     Nuke: Camera3 (animated), plate Read, undistort / redistort STMaps, ReadGeo of the Alembic
  stmap_undistort.exr / stmap_redistort.exr   32-bit ST maps (Nuke convention) for the solved lens distortion
  camera.json   everything: per-frame matrices (Z up and Y up), focal, sensor, distortion, rolling shutter
  camera.csv    frame, translate (Y up), rotate XYZ degrees (Y up), focal mm
Conventions: metres; the camera looks down its local -Z with +Y up (Blender, Maya, Nuke, USD). Y-up world =
Blender (x, z, -y).
"""
import csv
import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

ZUP_TO_YUP = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
FORMATS = ("abc", "fbx", "usd", "nk", "stmap", "json")


# --------------------------------------------------------------------------- lens distortion (Blender Polynomial)
def distort(x, y, k1, k2, k3=0.0):
    r2 = x * x + y * y
    g = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return x * g, y * g


def undistort(xd, yd, k1, k2, k3=0.0, iters=30):
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):                          # fixed point: x = xd / g(x)
        r2 = x * x + y * y
        g = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x, y = xd / g, yd / g
    return x, y


def stmaps(W, H, f_px, k1, k2, k3=0.0):
    """(undistort, redistort) ST maps, float32 (H, W, 4) RGBA, rows bottom-up (Blender image order).
    Pixel coordinates: u right / v down, pixel centres at integer + 0.5, principal point at the centre.
    undistort: an undistorted pixel reads the plate at distort(p). redistort: a distorted pixel reads the
    undistorted render at undistort(p)."""
    u = np.arange(W) + 0.5
    v = np.arange(H) + 0.5
    U, V = np.meshgrid(u, v)                        # V: rows top-down
    cx, cy = W / 2.0, H / 2.0
    x, y = (U - cx) / f_px, (V - cy) / f_px
    out = []
    for fn in (lambda a, b: distort(a, b, k1, k2, k3), lambda a, b: undistort(a, b, k1, k2, k3)):
        xs, ys = fn(x, y)
        us, vs = xs * f_px + cx, ys * f_px + cy     # source position, pixels from top-left
        img = np.zeros((H, W, 4), np.float32)
        img[..., 0] = us / W
        img[..., 1] = 1.0 - vs / H                  # Nuke / ST maps: origin bottom-left
        img[..., 3] = 1.0
        out.append(img[::-1])                       # Blender image rows: bottom-up
    return out


def _save_exr(path, img):
    H, W = img.shape[:2]
    name = os.path.basename(path)
    im = bpy.data.images.new(name, W, H, alpha=True, float_buffer=True)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(img.ravel())
    im.filepath_raw = path
    im.file_format = "OPEN_EXR"
    try:
        im.save()                                   # 32-bit float EXR from a float buffer
    finally:
        bpy.data.images.remove(im)
    return path


# --------------------------------------------------------------------------- camera data
def camera_samples(scene, cam, f0, f1):
    keep = scene.frame_current
    rows = []
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        M = cam.matrix_world.copy()
        rows.append(dict(frame=f, M=M, lens=float(cam.data.lens)))
    scene.frame_set(keep)
    return rows


def _flat(M):
    return [round(float(M[i][j]), 9) for i in range(4) for j in range(4)]


def write_json_csv(path_json, path_csv, scene, cam, clip, rows, d, meta):
    W, H = scene.render.resolution_x, scene.render.resolution_y
    sw = cam.data.sensor_width
    k1, k2 = float(d.get("k1", 0.0)), float(d.get("k2", 0.0))
    out = dict(
        generator=meta.get("generator", "CameraMatch Pi4"), units="metres",
        conventions=("camera looks down its local -Z, +Y up; matrix_world_zup = Blender world (Z up); "
                     "matrix_world_yup = Y-up world (x, z, -y); matrices row-major, camera-to-world"),
        frame_range=[rows[0]["frame"], rows[-1]["frame"]], fps=scene.render.fps / scene.render.fps_base,
        resolution=[W, H], sensor_width_mm=sw, sensor_height_mm=sw * H / W, sensor_fit="HORIZONTAL",
        principal_point_px=[W / 2.0, H / 2.0],
        distortion=dict(model="Blender Polynomial (libmv, radial Brown)", k1=k1, k2=k2, k3=0.0,
                        formula="x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6), same for y; r^2 = x^2 + y^2",
                        normalisation="x = (u - cx) / f_px, y = (v - cy) / f_px; u right, v down, pixels, "
                                      "pixel centres at integer + 0.5; f_px = focal_mm / sensor_width_mm * width",
                        stmaps="stmap_undistort.exr (plate -> undistorted), stmap_redistort.exr (CG -> plate lens)"),
        rolling_shutter=dict(readout_frames=float(scene.get("cm_rolling_shutter", 0.0)),
                             axis=str(scene.get("cm_rolling_shutter_axis", "rows")),
                             note="fraction of a frame between the first and the last image row (or column)"),
        solve=meta.get("solve", {}), scene_scale=float(scene.get("cm_applied_scale", 1.0)),
        plate=meta.get("plate", ""),
        frames=[dict(frame=r["frame"], focal_mm=r["lens"], focal_px=r["lens"] / sw * W,
                     matrix_world_zup=_flat(r["M"]), matrix_world_yup=_flat(ZUP_TO_YUP @ r["M"])) for r in rows])
    with open(path_json, "w") as fh:
        json.dump(out, fh, indent=1)
    with open(path_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg", "focal_mm"])
        for r in rows:
            My = ZUP_TO_YUP @ r["M"]
            t = My.translation
            e = My.to_3x3().normalized().to_euler("XYZ")
            w.writerow([r["frame"], *(f"{v:.6f}" for v in t), *(f"{math.degrees(a):.6f}" for a in e),
                        f"{r['lens']:.6f}"])


# --------------------------------------------------------------------------- Nuke
def _curve(vals, f0):
    return "{curve x%d %s}" % (f0, " ".join(f"{v:.6g}" for v in vals))


def write_nuke(path, scene, cam, rows, plate, abc_name, W, H, has_maps, seqs=None):
    """seqs: {node name: (file pattern with ####, first, last)} - plate / masks / depth sequences to Read."""
    f0, f1 = rows[0]["frame"], rows[-1]["frame"]
    T, R, F = [], [], []
    for r in rows:
        My = ZUP_TO_YUP @ r["M"]
        T.append(My.translation[:])
        R.append([math.degrees(a) for a in My.to_3x3().normalized().to_euler("XYZ")])
        F.append(r["lens"])
    T, R = np.array(T), np.array(R)
    for k in range(3):                              # unwrap Euler angles for clean curves
        R[:, k] = np.degrees(np.unwrap(np.radians(R[:, k])))
    sw = cam.data.sensor_width
    fps = scene.render.fps / scene.render.fps_base
    p = lambda s: s.replace("\\", "/")  # noqa: E731
    L = [f"#! CameraMatch Pi4 export (Nuke script, Y up, metres)",
         f"Root {{\n inputs 0\n first_frame {f0}\n last_frame {f1}\n fps {fps:g}\n"
         f" format \"{W} {H} 0 0 {W} {H} 1 CameraMatch\"\n}}",
         f"Read {{\n inputs 0\n file \"{p(plate)}\"\n first {f0}\n last {f1}\n origfirst {f0}\n origlast {f1}\n"
         f" name Plate\n xpos 0\n ypos 0\n}}"]
    if has_maps:
        L += ["set N_plate [stack 0]",
              f"Read {{\n inputs 0\n file \"{p(os.path.join(os.path.dirname(path), 'stmap_undistort.exr'))}\"\n"
              f" raw true\n name STMap_undistort_map\n xpos 150\n ypos 0\n}}",
              "push $N_plate",
              "STMap {\n inputs 2\n channels rgba\n uv rgb\n name Undistort_plate\n xpos 0\n ypos 100\n}",
              f"Read {{\n inputs 0\n file \"{p(os.path.join(os.path.dirname(path), 'stmap_redistort.exr'))}\"\n"
              f" raw true\n name STMap_redistort_map\n xpos 300\n ypos 0\n}}",
              "Constant {\n inputs 0\n channels rgb\n name CG_render_here\n xpos 450\n ypos 0\n}",
              "STMap {\n inputs 2\n channels rgba\n uv rgb\n name Redistort_CG\n xpos 450\n ypos 100\n}"]
    L += [f"Camera3 {{\n inputs 0\n rot_order XYZ\n"
          f" translate {{{_curve(T[:, 0], f0)} {_curve(T[:, 1], f0)} {_curve(T[:, 2], f0)}}}\n"
          f" rotate {{{_curve(R[:, 0], f0)} {_curve(R[:, 1], f0)} {_curve(R[:, 2], f0)}}}\n"
          f" focal {_curve(F, f0) if max(F) - min(F) > 1e-6 else f'{F[0]:.6g}'}\n"
          f" haperture {sw:.6g}\n vaperture {sw * H / W:.6g}\n"
          f" name CameraMatch_cam\n xpos 600\n ypos 0\n}}"]
    for k, (nm, (pat, a_, b_)) in enumerate(sorted((seqs or {}).items())):
        L += [f"Read {{\n inputs 0\n file \"{p(pat)}\"\n first {a_}\n last {b_}\n origfirst {a_}\n"
              f" origlast {b_}\n{' raw true' + chr(10) if not pat.endswith('plate_####.png') else ''}"
              f" name {nm}\n xpos {k * 150}\n ypos 250\n}}"]
    if abc_name:
        L += [f"ReadGeo2 {{\n inputs 0\n file \"{p(os.path.join(os.path.dirname(path), abc_name))}\"\n"
              f" name CameraMatch_geo\n xpos 750\n ypos 0\n}}"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


README = """CameraMatch Pi4 export
=====================
Units: metres. Camera: looks down its local -Z, +Y up. Y-up files (Alembic, FBX, USD, Nuke, camera.csv) use
Y-up world = Blender (x, z, -y). Frames are numbered like the Blender scene ({f0}-{f1}, {fps:g} fps): set the
same frame rate in the other application before importing (Alembic is time-based).

{name}.abc / .fbx / .usdc   camera (animated transform + focal), set meshes, track points, locators (trk_#####)
                            CG characters keep their materials in USD (preview surfaces, textures/ folder) and
                            FBX (textures embedded); Alembic carries geometry only (no materials).
{name}.nk                   Nuke: Camera3, plate, undistort / redistort STMaps, sequences below, ReadGeo
camera.json / camera.csv    per-frame camera, sensor, lens distortion (formula inside), rolling shutter
stmap_undistort.exr         plate -> undistorted (Nuke STMap convention, 32-bit)
stmap_redistort.exr         undistorted CG -> the plate's lens
plate/plate_####.png        the footage (8-bit, as decoded)
masks/moving_####.png       moving objects (Pi4, a soft probability: it can also flag static areas, e.g. 5%
                            of the frame on a wall in shadow; for roto-quality masks use masks/matte)
masks/trees_water_sky_####.png  trees / water / sky (semantic, when the solve used them)
masks/matte_####.png        foreground matte (MatAnyone 2, when Refine Matte was run)
masks/ignore_####.png       your Ignore Mask (when drawn)
depth/depth_####.exr        depth, 32-bit float channel "Z", metres along the optical axis (Z-depth)
lighting/plate_env.png      the lighting environment the add-on renders with: an equirectangular map built from
                            the plate (Blender world convention, Z up: in a Y-up application rotate it -90 deg about X
                            if it does not convert). Use it as the world / dome light, no other lights.
Set geometry (CM_floor_catcher, CM_walls, CM_street_walls): use as shadow catchers / matte objects, hidden from
the camera except for shadows (the add-on renders them as Cycles shadow catchers).

Masks and depth are at plate resolution, upscaled from the model's resolution, and follow the plate's lens
(distorted, like the plate). Depth is Pi4's per-frame depth scaled to agree with the solved camera (median over
the solved track points); expect a few % per pixel, more at object edges and on the sky. The export reports
the measured agreement: above ~15 % per point, use the depth for rough work only.
Lens distortion is not stored in Alembic / FBX / USD cameras: use the ST maps.
Blender's own FBX importer shifts animation by 1 frame by default: set Animation Offset to 0.
Non-commercial use only (Pi4: CC BY-NC 4.0; MatAnyone 2: NTU S-Lab License 1.0).
"""


def write_readme(out_dir, name, scene):
    with open(os.path.join(out_dir, "README.txt"), "w") as fh:
        fh.write(README.format(name=name, f0=scene.frame_start, f1=scene.frame_end,
                               fps=scene.render.fps / scene.render.fps_base))


def finish_nuke(out_dir, name, scene, rows=None, plate="", seqs=None):
    """(Re)write the Nuke script once the sequences exist."""
    cam = scene.camera
    rows = rows or camera_samples(scene, cam, scene.frame_start, scene.frame_end)
    abc = name + ".abc" if os.path.isfile(os.path.join(out_dir, name + ".abc")) else ""
    p = os.path.join(out_dir, name + ".nk")
    write_nuke(p, scene, cam, rows, plate, abc, scene.render.resolution_x, scene.render.resolution_y,
               os.path.isfile(os.path.join(out_dir, "stmap_undistort.exr")), seqs)
    return p


def sequences(out_dir):
    """Exported image sequences in out_dir -> {Nuke node name: (pattern, first, last)}."""
    import re as _re
    out = {}
    for sub, rx, node in (("plate", r"plate_(\d{4})\.png", "Plate_sequence"),
                          ("masks", r"moving_(\d{4})\.png", "Mask_moving"),
                          ("masks", r"trees_water_sky_(\d{4})\.png", "Mask_trees_water_sky"),
                          ("masks", r"matte_(\d{4})\.png", "Mask_matte"),
                          ("masks", r"ignore_(\d{4})\.png", "Mask_ignore"),
                          ("depth", r"depth_(\d{4})\.exr", "Depth")):
        d = os.path.join(out_dir, sub)
        if not os.path.isdir(d):
            continue
        nums = sorted(int(m.group(1)) for f in os.listdir(d) if (m := _re.fullmatch(rx, f)))
        if nums:
            pat = os.path.join(d, rx.split("(")[0].replace("\\", "") + "####" + rx.split(")")[1].replace("\\", ""))
            out[node] = (pat, nums[0], nums[-1])
    return out


# --------------------------------------------------------------------------- scene export
def _locators(scene, tracks_ob, obs, n, coll):
    """Empties at the n most observed track points (the classic matchmove locator set)."""
    if tracks_ob is None or n <= 0:
        return []
    V = [tracks_ob.matrix_world @ v.co for v in tracks_ob.data.vertices]
    if obs is not None and len(obs) and int(obs[:, 1].max()) < len(V):
        cnt = np.bincount(obs[:, 1].astype(np.int64), minlength=len(V))[:len(V)]
        order = np.argsort(-cnt)
    else:
        order = np.arange(len(V))
    out = []
    for i in order[:n]:
        e = bpy.data.objects.new(f"trk_{int(i):05d}", None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.05
        e.location = V[int(i)]
        coll.objects.link(e)
        out.append(e)
    return out


def export_scene(context, out_dir, name="cameramatch", formats=FORMATS, include_set=True, include_tracks=True,
                 n_locators=150, include_cg=False, log=print, meta=None):
    """Write the selected formats into out_dir. Returns {format: path}."""
    sc = context.scene
    cam = sc.camera
    if cam is None:
        raise RuntimeError("no scene camera: solve first")
    meta = meta or {}
    os.makedirs(out_dir, exist_ok=True)
    f0, f1 = sc.frame_start, sc.frame_end
    W, H = sc.render.resolution_x, sc.render.resolution_y
    d = meta.get("result", {})
    written = {}

    # objects to export
    objs = [cam]
    tracks_ob = next((o for o in sc.objects if o.name.endswith("_tracks") and o.type == "MESH"), None)
    if include_tracks and tracks_ob is not None:
        objs.append(tracks_ob)
    if include_set and bpy.data.collections.get("CM_Set"):
        objs += [o for o in bpy.data.collections["CM_Set"].all_objects if o.type == "MESH"]
    if include_cg:
        for c in bpy.data.collections:
            if c.name.startswith(("CM_Character", "CM_Robot")):
                objs += [o for o in c.all_objects if o.type in {"MESH", "ARMATURE", "EMPTY"} and not o.hide_render]
    tmp = bpy.data.collections.new("CM_Export_Locators")
    sc.collection.children.link(tmp)
    obs = meta.get("track_obs")
    if tracks_ob is not None and obs is not None and len(obs) and int(obs[:, 1].max()) + 1 != len(tracks_ob.data.vertices):
        log("export: warning: the solve on disk does not match the scene's track points (re-solved without "
            "re-importing?). Exporting the scene as it is; More Tools > Re-import Last Result syncs them")
    locs = _locators(sc, tracks_ob, obs, n_locators if include_tracks else 0, tmp)
    objs += locs
    sel = [o for o in context.view_layer.objects if o.select_get()]
    act = context.view_layer.objects.active
    try:
        for o in context.view_layer.objects:
            o.select_set(False)
        for o in objs:
            if o.name in context.view_layer.objects:
                o.hide_set(False)
                o.select_set(True)
        context.view_layer.objects.active = cam
        with context.temp_override(selected_objects=objs, active_object=cam):
            if "abc" in formats:
                p = os.path.join(out_dir, name + ".abc")
                bpy.ops.wm.alembic_export(filepath=p, start=f0, end=f1, selected=True, flatten=False, uvs=True,
                                          normals=True, export_custom_properties=False, as_background_job=False,
                                          evaluation_mode="RENDER", init_scene_frame_range=False)
                written["abc"] = p
            if "fbx" in formats:
                p = os.path.join(out_dir, name + ".fbx")
                bpy.ops.export_scene.fbx(filepath=p, use_selection=True,
                                         object_types={"CAMERA", "MESH", "EMPTY", "ARMATURE"}, bake_anim=True,
                                         bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                                         bake_anim_force_startend_keying=True, bake_anim_step=1.0,
                                         bake_anim_simplify_factor=0.0, add_leaf_bones=False,
                                         axis_forward="-Z", axis_up="Y", apply_scale_options="FBX_SCALE_ALL",
                                         use_custom_props=False, path_mode="COPY", embed_textures=True)
                written["fbx"] = p
            if "usd" in formats:
                p = os.path.join(out_dir, name + ".usdc")
                bpy.ops.wm.usd_export(filepath=p, selected_objects_only=True, export_animation=True,
                                      export_materials=True, generate_preview_surface=True,
                                      export_textures_mode="NEW", export_cameras=True, export_meshes=True,
                                      export_lights=False, export_armatures=False,   # characters: baked meshes
                                      convert_orientation=True, export_global_up_selection="Y",
                                      export_global_forward_selection="NEGATIVE_Z")
                written["usd"] = p
    finally:
        for o in locs:
            bpy.data.objects.remove(o)
        bpy.data.collections.remove(tmp)
        for o in context.view_layer.objects:
            o.select_set(o in sel)
        context.view_layer.objects.active = act

    rows = camera_samples(sc, cam, f0, f1)
    k1, k2 = float(d.get("k1", 0.0)), float(d.get("k2", 0.0))
    f_px = float(np.median([r["lens"] for r in rows])) / cam.data.sensor_width * W
    has_maps = "stmap" in formats
    if has_maps:
        if any(abs(r["lens"] - rows[0]["lens"]) > 1e-4 for r in rows):
            log("export: zoom shot: the ST maps use the median focal (distortion is relative to the focal)")
        um, rm = stmaps(W, H, f_px, k1, k2)
        written["stmap_undistort"] = _save_exr(os.path.join(out_dir, "stmap_undistort.exr"), um)
        written["stmap_redistort"] = _save_exr(os.path.join(out_dir, "stmap_redistort.exr"), rm)
    if "json" in formats:
        pj, pc = os.path.join(out_dir, "camera.json"), os.path.join(out_dir, "camera.csv")
        write_json_csv(pj, pc, sc, cam, None, rows, d, meta)
        written["json"], written["csv"] = pj, pc
    if "nk" in formats:
        p = os.path.join(out_dir, name + ".nk")
        write_nuke(p, sc, cam, rows, meta.get("plate", ""), os.path.basename(written["abc"]) if "abc" in written
                   else "", W, H, has_maps)
        written["nk"] = p
    env = os.path.join(bpy.path.abspath(meta.get("job_dir", "")), "plate_env.png") if meta.get("job_dir") else ""
    if env and os.path.isfile(env):
        import shutil
        os.makedirs(os.path.join(out_dir, "lighting"), exist_ok=True)
        shutil.copy2(env, os.path.join(out_dir, "lighting", "plate_env.png"))
        written["lighting"] = os.path.join(out_dir, "lighting", "plate_env.png")
    write_readme(out_dir, name, sc)
    log(f"export: {len(objs)} objects, frames {f0}-{f1} -> " + ", ".join(os.path.basename(v) for v in written.values()))
    return written
