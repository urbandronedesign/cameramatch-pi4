# SPDX-License-Identifier: GPL-3.0-or-later
"""CameraMatch Pi4: send a movie clip to a local Pi4 server, import the solved camera.

Workflow: Sidebar > Pi4 tab (Movie Clip Editor or 3D Viewport): six guided steps, from Load Footage to
Render. The heavy lifting runs in Docker containers (the "engine", started from the panel); this add-on
copies the clip to the shared work folder, calls the servers over HTTP and imports the results.
"""
import json
import math
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector

from . import characters, collide, engine, export, guide, ignore_mask, paths, robot, set_builder

# The work folder is the repo's ./work, mounted as /work by docker-compose. Set it in the add-on preferences,
# or with the CAMERAMATCH_WORK environment variable.
_WORK_ENV = "CAMERAMATCH_WORK"
# Shared between the UI (main thread) and worker/poller threads. Only plain values are written from threads.
_STATUS = {"text": "", "running": False, "pct": 0.0, "stage": "", "op": "", "t0": 0.0, "t_end": 0.0,
           "log": [], "synced": 0, "server": "", "job": "", "need_engine": False, "cancel": False}
_LOG_TEXT = "CameraMatch_Log"
_STAGE_LABELS = {
    "copy": "Copying clip", "decode": "Decoding frames", "pi4": "Pi4 inference", "post": "Post-processing",
    "klt": "KLT tracking", "ba": "Bundle adjustment", "write": "Writing result", "import": "Importing into Blender",
    "load": "Loading frames", "sam_seeds": "SAM 2.1 keyframes", "sam_video": "SAM 2.1 propagation",
    "matanyone": "MatAnyone 2", "fuse": "Fusing mattes", "export": "Exporting mattes", "done": "Done",
    "error": "Failed", "mesh": "Set mesh (Poisson)", "envmap": "Plate environment", "set": "Building set in Blender",
    "render": "Rendering (Cycles)", "comp": "Compositing", "encode": "Encoding video",
    "check": "Overlay check", "download": "Downloading", "maps": "Plate, masks, depth",
}
SAMPLE_CLIP_URL = "https://assets.mixkit.co/videos/4600/4600-1080.mp4"      # Mixkit free license
SAMPLE_CHAR_URL = ("https://github.com/mrdoob/three.js/raw/r170/examples/models/gltf/RobotExpressive/"
                   "RobotExpressive.glb")                                    # CC0, Tomas Laulhe / Quaternius
DOCS_URL = "https://github.com/urbandronedesign/cameramatch-pi4#quickstart"


def _log(line):
    stamp = time.strftime("%H:%M:%S")
    _STATUS["log"].append(f"{stamp} {line}")
    print(f"[CameraMatch] {line}")


def _start_progress(op, server_url, job):
    """Begin a tracked operation; a poller thread mirrors the server's /progress into _STATUS."""
    _STATUS.update(running=True, pct=0.0, stage="copy" if op == "Solve" else "load", op=op,
                   t0=time.time(), t_end=0.0, text=f"{op} starting...", server=server_url, job=job,
                   need_engine=False, cancel=False)
    _log(f"===== {op} | job {job} =====")
    url = f"{server_url.rstrip('/')}/progress?job={urllib.parse.quote(job)}"

    def poll():
        since, live = 0, False
        while _STATUS["running"]:
            try:
                p = _http(f"{url}&since={since}", timeout=2)
                # the server may still hold this job's previous run (finished): wait until the new one starts
                live = live or p.get("state") == "running"
                if live and _STATUS["running"]:     # never overwrite the final summary after the job ended
                    _STATUS["pct"] = float(p.get("pct", 0.0))
                    _STATUS["stage"] = p.get("stage", _STATUS["stage"])
                    if p.get("msg"):
                        _STATUS["text"] = p["msg"]
                    for line in p.get("log", []):
                        _log("  " + line)
                    since = int(p.get("log_len", since))
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=poll, daemon=True).start()


def _end_progress(ok, text):
    """Finish the tracked operation. Failures are logged in full and shown in plain language."""
    shown = text
    if not ok:
        _log("FAILED: " + text)
        shown, need = guide.friendly(text)
        _STATUS["need_engine"] = need
    else:
        _log("DONE: " + text)
    _STATUS.update(running=False, pct=1.0 if ok else _STATUS["pct"], stage="done" if ok else "error",
                   text=shown, t_end=time.time())
    _sync_log_text()
    try:
        bpy.context.workspace.status_text_set(None)
    except Exception:
        pass


def _sync_log_text():
    """Mirror the log into a Text datablock (main thread only)."""
    txt = bpy.data.texts.get(_LOG_TEXT) or bpy.data.texts.new(_LOG_TEXT)
    new = _STATUS["log"][_STATUS["synced"]:]
    if new:
        txt.cursor_set(len(txt.lines) - 1, character=len(txt.lines[-1].body))
        txt.write("\n".join(new) + "\n")
        _STATUS["synced"] = len(_STATUS["log"])


def _redraw(context):
    _sync_log_text()
    try:  # status bar is a global area: status_text_set both shows the text and forces its redraw
        if _STATUS["running"]:
            pct = max(0.0, min(1.0, _STATUS["pct"]))
            el = time.time() - _STATUS["t0"]
            eta = f" | ETA {el * (1 - pct) / pct:.0f}s" if pct > 0.03 else ""
            context.workspace.status_text_set(
                f"CameraMatch {_STATUS['op']}: {_STAGE_LABELS.get(_STATUS['stage'], _STATUS['stage'])} "
                f"{pct * 100:.0f}% | {el:.0f}s{eta} | {_STATUS['text'][:70]}")
        else:
            context.workspace.status_text_set(None)
    except Exception:
        pass
    _tag_redraw(context.window_manager)


def _tag_redraw(wm):
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in {"CLIP_EDITOR", "TEXT_EDITOR", "STATUSBAR", "VIEW_3D", "PREFERENCES"}:
                area.tag_redraw()

# OpenCV camera (x right, y down, z fwd) -> Blender camera (x right, y up, looks -z)
CV_TO_BL_CAM = np.diag([1.0, -1.0, -1.0, 1.0])
# OpenCV world (y down, z fwd) -> Blender world (z up, y fwd)
CV_TO_BL_WORLD = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


# --------------------------------------------------------------------------- prefs / props
class CAMERAMATCH_Prefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    server_url: StringProperty(name="Server URL", default="http://127.0.0.1:8765")
    matte_server_url: StringProperty(name="Matte Server URL", default="http://127.0.0.1:8766",
                                     description="SAM 2.1 + MatAnyone 2 server (docker compose service matte-server)")
    work_dir: StringProperty(
        name="Work folder", subtype="DIR_PATH", default="",
        description="Your clone's work/ folder: the host folder docker-compose mounts as /work "
                    "(empty: use the CAMERAMATCH_WORK environment variable)",
    )
    engine_dir: StringProperty(
        name="Engine folder", subtype="DIR_PATH", default="",
        description="The CameraMatch folder holding docker-compose.yml (empty: the Work folder's parent)",
    )

    def draw(self, context):
        lay = self.layout
        icon, text = engine.summary()
        lay.label(text=text, icon=icon)
        lay.prop(self, "work_dir")
        lay.prop(self, "engine_dir")
        if not _work_dir():
            lay.label(text="Set the Work folder to <CameraMatch folder>/work (setup.ps1 prints it)", icon="ERROR")
        elif not _engine_dir():
            lay.label(text="docker-compose.yml not found: set the Engine folder", icon="ERROR")
        row = lay.row()
        row.operator("cameramatch.start_engine", icon="PLAY")
        row.operator("cameramatch.stop_engine", icon="PAUSE")
        row.operator("cameramatch.open_docs", icon="HELP")
        col = lay.column()
        col.prop(self, "server_url")
        col.prop(self, "matte_server_url")


def _preset_update(self, context):
    guide.apply_preset(self, self.preset)


class CAMERAMATCH_Settings(bpy.types.PropertyGroup):
    clip: PointerProperty(type=bpy.types.MovieClip, name="Footage",
                          description="The clip to solve (Load Footage sets it)")
    preset: EnumProperty(name="Footage Type", items=guide.PRESET_ITEMS, default="AUTO", update=_preset_update,
                         description="Sets the solve options for the kind of shot (Advanced: Solve shows them)")
    zoom_mode: EnumProperty(name="Zoom", default="AUTO",
                            items=[("AUTO", "Auto", "Detect a zoom from the model's per-frame focal"),
                                   ("ON", "Zoom Lens", "Solve one focal per frame"),
                                   ("OFF", "Fixed Focal", "One focal for the whole shot")])
    rs_mode: EnumProperty(name="Rolling Shutter", default="AUTO",
                          items=[("AUTO", "Auto", "On for portrait (phone) footage"),
                                 ("ON", "On", "Solve the sensor readout time (CMOS video, fast handheld moves)"),
                                 ("OFF", "Off", "Global shutter / slow moves")])
    char_height: FloatProperty(name="Character Height", default=1.75, min=0.0, max=20.0, unit="LENGTH",
                               description="Scale imported characters to this height (0: keep the file's size)")
    show_done: BoolProperty(name="Show Finished Steps", default=False,
                            description="Show the buttons of the steps already done")
    stride: IntProperty(name="Frame Stride", default=1, min=1, max=50,
                        description="Use every Nth frame (in-between frames are interpolated)")
    max_frames: IntProperty(name="Max Frames", default=300, min=2, max=2000,
                            description="Upper bound on frames sent to the model (VRAM)")
    max_size: IntProperty(name="Model Resolution", default=512, min=128, max=1024, step=16,
                          description="Long side in pixels the model runs at (multiple of 16)")
    track: BoolProperty(name="Detect Moving Objects", default=True,
                        description="Run Pi4's tracking/motion heads: masks moving objects, costs more VRAM")
    conf_threshold: FloatProperty(name="Point Confidence", default=0.05, min=0.0, max=1.0,
                                  description="Minimum Pi4 geometry confidence for point cloud pixels")
    num_points: IntProperty(name="Max Points", default=200_000, min=0, max=5_000_000)
    animate_focal: BoolProperty(name="Zoom Lens (Animate Focal)", default=False,
                                description="Zoom shots: solve one focal length per frame (smooth), keyframed on "
                                            "the camera. Off: one constant focal for the whole shot")
    refine: BoolProperty(name="Refine at Plate Resolution", default=True,
                         description="KLT tracking + bundle adjustment at full resolution: refines poses, "
                                     "focal and k1/k2 lens distortion (Blender Polynomial model)")
    solve_distortion: BoolProperty(name="Solve Lens Distortion", default=True,
                                   description="Solve k1/k2. Turn off for rectilinear / cinema lenses and slow, "
                                               "low-parallax shots, where focal and distortion trade off")
    lock_focal: BoolProperty(name="Lock Focal to Pi4", default=False,
                             description="Hold the focal length at Pi4's estimate. Use on slow / low-parallax shots "
                                         "where the tracks cannot determine focal (keeps Pi4 depth consistent)")
    rolling_shutter: BoolProperty(name="Rolling Shutter", default=False,
                                  description="Solve the sensor readout time (CMOS video cameras, fast handheld "
                                              "moves): each image row gets the pose at its capture time. Render & "
                                              "Comp then renders the CG with the same rolling shutter")
    ignore_nonrigid: BoolProperty(name="Ignore Trees / Water / Sky", default=True,
                                  description="Never track features on trees, plants, water or sky (OneFormer "
                                              "semantic masks, MIT): leaves in the wind, waves and clouds are not "
                                              "static and pull the camera solve. About 25 s extra")
    ignore_mask: PointerProperty(type=bpy.types.Mask, name="Ignore Mask",
                                 description="A mask drawn in the Movie Clip Editor (Mask mode): no feature is tracked "
                                             "inside it (screens, reflections, moving things Pi4 / the semantic "
                                             "masks miss). Animated and track-parented masks are supported")
    max_tracks: IntProperty(name="Active Tracks", default=600, min=50, max=5000,
                            description="Target number of simultaneously tracked features for refinement")
    auto_matte: BoolProperty(name="Refine Matte After Solve", default=False,
                             description="Run SAM 2.1 + MatAnyone 2 foreground matte right after the camera solve")
    last_job: StringProperty(name="Last Job", default="")
    # CG integration
    set_street_walls: BoolProperty(name="Street Walls", default=True,
                                   description="Detect the two walls of a street along the camera's travel and close "
                                               "the set with tall proxy walls (sky light then only comes from above)")
    set_wall_height: FloatProperty(name="Wall Height", default=14.0, min=2.0, max=60.0, unit="LENGTH")
    set_rebuild: BoolProperty(name="Rebuild Mesh", default=False,
                              description="Recompute the low-poly set mesh and environment even if they exist")
    set_tris: IntProperty(name="Set Triangles", default=60000, min=5000, max=500000)
    robot_speed: FloatProperty(name="Walk Speed", default=0.85, min=0.2, max=2.0, unit="VELOCITY",
                               description="Capped so the robot stops 3 m short of the camera")
    avoid_collisions: BoolProperty(name="Avoid Collisions", default=True,
                                   description="Robot / Move Along Path bend their path around the set's walls, "
                                               "static CG objects and earlier robots / moved objects")
    robot_look: BoolProperty(name="Look at Camera", default=True,
                             description="The robot turns its head to the camera in the last third of the shot")
    render_samples: IntProperty(name="Samples", default=192, min=1, max=4096)
    render_motion_blur: BoolProperty(name="Motion Blur", default=True)
    comp_soft: FloatProperty(name="CG Softness (px)", default=0.6, min=0.0, max=3.0,
                             description="Gaussian blur on the CG to match the plate's sharpness")
    comp_wrap: FloatProperty(name="Light Wrap", default=0.25, min=0.0, max=1.0)


def _rs_on(s, clip):
    if s.rs_mode == "AUTO":
        return bool(clip is not None and clip.size[1] > clip.size[0])      # portrait = phone
    return s.rs_mode == "ON"


def _refine_settings(s, clip=None):
    """The plate-resolution refinement settings shared by Solve and Re-solve."""
    zoom = "auto" if s.zoom_mode == "AUTO" else s.zoom_mode == "ON"
    s.rolling_shutter = _rs_on(s, clip)
    return dict(max_tracks=s.max_tracks, solve_distortion=s.solve_distortion,
                focal_sigma=0.002 if s.lock_focal else 0.3, zoom=zoom, solver="lm",
                rolling_shutter=s.rolling_shutter, rs_axis="auto", ignore_classes=bool(s.ignore_nonrigid))


def _prefs():
    return bpy.context.preferences.addons[__package__].preferences


def _work_dir():
    """Work folder from the preferences, else $CAMERAMATCH_WORK, else '' (not configured)."""
    w = _prefs().work_dir.strip() or os.environ.get(_WORK_ENV, "").strip()
    return bpy.path.abspath(w) if w else ""


def _engine_dir():
    p = _prefs()
    return engine.find_engine_dir(bpy.path.abspath(p.engine_dir.strip()) if p.engine_dir.strip() else "",
                                  _work_dir())


def _engine_down(op):
    """Refuse to start a job when the last health check says the engine is down. Returns True if down."""
    if engine.STATE["pi4"] == "down" and engine.STATE["matte"] == "down":
        _STATUS.update(op=op, stage="error", text="The engine is not running. Click Start Engine.",
                       need_engine=True, running=False, t0=time.time(), t_end=time.time())
        return True
    return False


# --------------------------------------------------------------------------- server I/O
def _http(url, payload=None, timeout=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}"}


def _copy_clip_input(clip_path, source, job_dir):
    """Copy the clip (movie file or image sequence) into job_dir/input. Returns rel path."""
    in_dir = os.path.join(job_dir, "input")
    os.makedirs(in_dir, exist_ok=True)
    if source == "MOVIE":
        dst = os.path.join(in_dir, os.path.basename(clip_path))
        shutil.copy2(clip_path, dst)
        return "input/" + os.path.basename(clip_path)
    # Image sequence: copy every file sharing the prefix/extension of the first frame.
    folder, name = os.path.split(clip_path)
    m = re.match(r"^(.*?)(\d+)(\.\w+)$", name)
    if not m:
        raise RuntimeError(f"Cannot parse sequence filename: {name}")
    prefix, ext = m.group(1), m.group(3).lower()
    pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(ext) + "$", re.IGNORECASE)
    files = sorted((int(pat.match(f).group(1)), f) for f in os.listdir(folder) if pat.match(f))
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):  # Pi4's loader ignores anything else
        raise RuntimeError(f"Unsupported sequence format {ext} (use PNG/JPG; TIFF/EXR not supported yet)")
    seq_dir = os.path.join(in_dir, "seq")
    os.makedirs(seq_dir, exist_ok=True)
    for i, (_, f) in enumerate(files):
        shutil.copy2(os.path.join(folder, f), os.path.join(seq_dir, f"{i:06d}{ext}"))
    return "input/seq"


# --------------------------------------------------------------------------- import
def _srgb_to_linear(c):
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _points_node_group():
    """Mesh verts -> points; radius = per-point 'pi4_radius' attribute * 'Point Size' factor."""
    ng = bpy.data.node_groups.new("Pi4 Points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Point Size", in_out="INPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    ng.interface.items_tree["Point Size"].default_value = 1.0
    n_in = ng.nodes.new("NodeGroupInput")
    n_out = ng.nodes.new("NodeGroupOutput")
    named = ng.nodes.new("GeometryNodeInputNamedAttribute")
    named.data_type = "FLOAT"
    named.inputs["Name"].default_value = "pi4_radius"
    mul = ng.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    n_in.location, named.location, mul.location = (-600, 0), (-600, -200), (-350, -100)
    m2p.location, setmat.location, n_out.location = (-150, 0), (100, 0), (350, 0)
    ng.links.new(named.outputs["Attribute"], mul.inputs[0])
    ng.links.new(n_in.outputs["Point Size"], mul.inputs[1])
    ng.links.new(mul.outputs["Value"], m2p.inputs["Radius"])

    mat = bpy.data.materials.new("Pi4 Points")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    attr.location, emit.location, out.location = (-400, 0), (-150, 0), (100, 0)
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    setmat.inputs["Material"].default_value = mat

    ng.links.new(n_in.outputs["Geometry"], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], n_out.inputs["Geometry"])
    return ng


def job_scale(job_dir):
    try:
        with open(os.path.join(job_dir, "scene_scale.json")) as fh:
            return float(json.load(fh)["scale"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1.0


def remove_imported(scene, clip):
    """Delete a previous import of this clip's solve (collection, objects, mesh/camera data, mask clip)."""
    name = f"Pi4_{clip.name}"
    coll = bpy.data.collections.get(name)
    if coll:
        for ob in list(coll.objects):
            data = ob.data
            bpy.data.objects.remove(ob)
            if data is not None and data.users == 0:
                (bpy.data.cameras if isinstance(data, bpy.types.Camera) else bpy.data.meshes).remove(data)
        bpy.data.collections.remove(coll)
    mc = bpy.data.movieclips.get(name + "_motion_mask")
    if mc:
        bpy.data.movieclips.remove(mc)


def import_result(scene, clip, job_dir, settings):
    d = np.load(os.path.join(job_dir, "result.npz"))
    refined = "refined" in d and bool(d["refined"])
    c2w = d["c2w_refined"] if refined else d["c2w"]
    T = c2w.shape[0]
    model_w = int(d["model_size"][0])
    frames = clip.frame_start + d["frame_indices"].astype(int)

    # Express everything relative to the first solved camera, then convert axes, then apply the job's
    # real-world scale (Set Scale from Distance writes scene_scale.json).
    sc_val = job_scale(job_dir)
    scene["cm_applied_scale"] = sc_val          # what this import really used (Set Scale works from it)
    scene["cm_rolling_shutter"] = float(d["rolling_shutter"]) if "rolling_shutter" in d else 0.0
    scene["cm_rolling_shutter_axis"] = str(d["rolling_shutter_axis"]) if "rolling_shutter_axis" in d else "rows"
    anchor = np.diag([sc_val, sc_val, sc_val, 1.0]) @ CV_TO_BL_WORLD @ np.linalg.inv(c2w[0])
    mats = [anchor @ c2w[i] @ CV_TO_BL_CAM for i in range(T)]

    tcam = clip.tracking.camera
    sensor_w = tcam.sensor_width
    if refined:
        f_fr = d["f_px_frames"] if "f_px_frames" in d and bool(d.get("zoom", False)) else np.full(T, float(d["f_px"]))
        lens = np.asarray(f_fr, np.float64) * sensor_w / int(d["plate_size"][0])
    else:
        lens = d["fx"] * sensor_w / model_w
    lens_med = float(np.median(lens))

    name = f"Pi4_{clip.name}"
    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = sensor_w
    cam_data.lens = lens_med
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    cam = bpy.data.objects.new(name, cam_data)
    coll = bpy.data.collections.new(name)
    scene.collection.children.link(coll)
    coll.objects.link(cam)

    cam.rotation_mode = "QUATERNION"
    prev_q = None
    for f, M, ln in zip(frames, mats, lens):
        loc, q, _ = Matrix(M.tolist()).decompose()
        if prev_q is not None and q.dot(prev_q) < 0:  # keep quaternion path continuous
            q.negate()
        prev_q = q
        cam.location, cam.rotation_quaternion = loc, q
        cam.keyframe_insert("location", frame=int(f))
        cam.keyframe_insert("rotation_quaternion", frame=int(f))
        if (settings.animate_focal and not refined) or (refined and bool(d.get("zoom", False))):
            cam_data.lens = float(ln)
            cam_data.keyframe_insert("lens", frame=int(f))

    # Refined: sparse bundle-adjusted track points (accurate, plate-consistent).
    if refined and len(d["track_points"]):
        tp = d["track_points"].astype(np.float64)
        tp_bl = (anchor @ np.c_[tp, np.ones(len(tp))].T).T[:, :3].astype(np.float32)
        tmesh = bpy.data.meshes.new(name + "_tracks")
        tmesh.vertices.add(len(tp_bl))
        tmesh.vertices.foreach_set("co", tp_bl.ravel())
        tmesh.update()
        tobj = bpy.data.objects.new(name + "_tracks", tmesh)
        tobj.show_in_front = True
        coll.objects.link(tobj)

    # Point cloud.
    pts = d["points"]
    if len(pts):
        pts_h = np.c_[pts.astype(np.float64), np.ones(len(pts))]
        pts_bl = (anchor @ pts_h.T).T[:, :3].astype(np.float32)
        mesh = bpy.data.meshes.new(name + "_points")
        mesh.vertices.add(len(pts_bl))
        mesh.vertices.foreach_set("co", pts_bl.ravel())
        rgba = np.c_[_srgb_to_linear(d["colors"]), np.ones(len(pts_bl), np.float32)]
        col = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        col.data.foreach_set("color", rgba.ravel())
        if "radii" in d:
            radii = d["radii"].astype(np.float32)
        else:  # results from older servers
            ext = float(np.linalg.norm(np.percentile(pts_bl, 95, 0) - np.percentile(pts_bl, 5, 0)))
            radii = np.full(len(pts_bl), max(ext, 1e-3) * 0.0015, np.float32)
        rad = mesh.attributes.new("pi4_radius", "FLOAT", "POINT")
        rad.data.foreach_set("value", radii * sc_val)
        mesh.update()
        pobj = bpy.data.objects.new(name + "_points", mesh)
        mod = pobj.modifiers.new("Points", "NODES")
        mod.node_group = _points_node_group()
        coll.objects.link(pobj)

    # Motion masks as a separate clip for reference / mask work.
    mask_dir = os.path.join(job_dir, "masks")
    if os.path.isdir(mask_dir):
        files = sorted(os.listdir(mask_dir))
        if files:
            mclip = bpy.data.movieclips.load(os.path.join(mask_dir, files[0]))
            mclip.name = name + "_motion_mask"
            mclip.frame_start = clip.frame_start

    # Scene setup + feed the estimate back into the clip's tracking camera.
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = clip.size
    scene.frame_start, scene.frame_end = int(frames[0]), int(frames[-1])
    if clip.fps > 0:  # match the plate's frame rate (motion blur, timing of any animation added later)
        scene.render.fps, scene.render.fps_base = round(clip.fps), round(clip.fps) / clip.fps
    tcam.focal_length = lens_med
    hide_camera_blockers(scene, _log)
    if refined:
        # Same model as libmv's Polynomial: x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6), centred principal point.
        tcam.distortion_model = "POLYNOMIAL"
        tcam.k1, tcam.k2, tcam.k3 = float(d["k1"]), float(d["k2"]), 0.0
        tcam.principal_point = (0.0, 0.0)
    return cam, lens_med, float(np.std(lens))


def hide_camera_blockers(scene, log=None):
    """Hide (viewport + render) objects the solved camera sits inside at some frame, e.g. the startup Cube at the
    origin, where the first solved camera is: it would fill every rendered frame. Returns their names."""
    cam = scene.camera
    if cam is None:
        return []
    ad = cam.animation_data
    if ad and ad.action:
        f0, f1 = (int(v) for v in ad.action.frame_range)
    else:
        f0, f1 = scene.frame_start, scene.frame_end
    keep = scene.frame_current
    pos = []
    for f in range(f0, f1 + 1, max(1, (f1 - f0) // 40)):
        scene.frame_set(f)
        pos.append(cam.matrix_world.translation.copy())
    scene.frame_set(keep)
    ours = ("Pi4_", "CM_", "Robot")
    hidden = []
    for ob in scene.objects:
        if ob.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"} or ob.hide_render or ob.name.startswith(ours):
            continue
        if any(c.name.startswith(ours) for c in ob.users_collection):
            continue
        inv = ob.matrix_world.inverted()
        bb = [Vector(c) for c in ob.bound_box]
        lo = Vector([min(c[i] for c in bb) for i in range(3)])
        hi = Vector([max(c[i] for c in bb) for i in range(3)])
        for p in pos:
            q = inv @ p
            if all(lo[i] - 1e-4 <= q[i] <= hi[i] + 1e-4 for i in range(3)):
                ob.hide_render = True
                ob.hide_set(True)
                hidden.append(ob.name)
                break
    if hidden and log:
        log(f"hid {', '.join(hidden)} from render: the solved camera is inside it (it would cover every frame)")
    return hidden


def _load_sequence_clip(name, first_file, frame_start):
    old = bpy.data.movieclips.get(name)
    if old is not None:
        bpy.data.movieclips.remove(old)
    mc = bpy.data.movieclips.load(first_file)
    mc.name = name
    mc.frame_start = frame_start
    return mc


def import_matte(clip, job_dir):
    """Load the fused matte as '<clip>_matte' clip and Pi4 inverse depth as a Non-Color image sequence."""
    mdir = os.path.join(job_dir, "matte", "fused")
    files = sorted(f for f in os.listdir(mdir) if f.startswith("alpha_"))
    mclip = _load_sequence_clip(f"{clip.name}_matte", os.path.join(mdir, files[0]), clip.frame_start)
    odir = os.path.join(job_dir, "occlusion")
    inv = sorted(f for f in os.listdir(odir) if f.startswith("invdepth_")) if os.path.isdir(odir) else []
    if inv:
        nm = f"{clip.name}_pi4_invdepth"
        old = bpy.data.images.get(nm)
        if old is not None:
            bpy.data.images.remove(old)
        img = bpy.data.images.load(os.path.join(odir, inv[0]))
        img.name = nm
        img.source = "SEQUENCE"
        img.colorspace_settings.name = "Non-Color"
    return mclip


# --------------------------------------------------------------------------- operators
class CAMERAMATCH_OT_check_server(bpy.types.Operator):
    bl_idname = "cameramatch.check_server"
    bl_label = "Check Server"
    bl_description = "Ping the Pi4 inference server"

    def execute(self, context):
        try:
            info = _http(_prefs().server_url.rstrip("/") + "/health", timeout=3)
        except Exception as e:
            self.report({"ERROR"}, f"Server unreachable: {e}. Is the container running? (docker compose up -d)")
            return {"CANCELLED"}
        msg = f"Pi4 server: {info.get('status')} | {info.get('gpu', '?')} | torch {info.get('torch', '?')}"
        self.report({"INFO"} if info.get("status") == "ready" else {"WARNING"}, msg)
        return {"FINISHED"}


class CAMERAMATCH_OT_solve(bpy.types.Operator):
    bl_idname = "cameramatch.solve"
    bl_label = "Solve Camera (Pi4)"
    bl_description = "Send the active clip to Pi4 and import an animated camera and point cloud"

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return _main_clip(context) is not None and not _STATUS["running"]

    def invoke(self, context, event):
        clip = _main_clip(context)
        prefs = _prefs()
        if _engine_down("Solve"):
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        s = context.scene.cameramatch
        work = _work_dir()
        if not work:
            self.report({"ERROR"}, "Work folder not set: Edit > Preferences > Add-ons > CameraMatch Pi4 > "
                                   "Work folder = <your clone>/work")
            return {"CANCELLED"}
        if not os.path.isdir(work):
            self.report({"ERROR"}, f"Work folder not found: {work} (set it in the add-on preferences)")
            return {"CANCELLED"}
        if clip.source not in {"MOVIE", "SEQUENCE"}:
            self.report({"ERROR"}, "Clip must be a movie file or image sequence")
            return {"CANCELLED"}

        job = re.sub(r"[^\w.-]", "_", clip.name) + time.strftime("_%Y%m%d_%H%M%S")
        self.job_dir = os.path.join(work, job)
        os.makedirs(self.job_dir)
        self.clip_name = clip.name
        self.result = None
        clip_path = bpy.path.abspath(clip.filepath)
        payload = dict(job=job, stride=s.stride, max_frames=s.max_frames,
                       max_size=s.max_size - s.max_size % 16, track=s.track,
                       conf_threshold=s.conf_threshold, num_points=s.num_points, refine=s.refine,
                       **_refine_settings(s, clip))
        url = prefs.server_url.rstrip("/") + "/solve"
        if s.ignore_mask is not None:
            if not s.refine:
                self.report({"WARNING"}, "Ignore Mask is used by the plate-resolution refinement: enable Refine")
            else:
                _STATUS["text"] = f"rasterising ignore mask '{s.ignore_mask.name}'..."
                ignore_mask.rasterize(s.ignore_mask, clip, context.scene, self.job_dir, log=_log)

        def worker():
            try:
                _STATUS["text"] = "Copying clip to the work folder..."
                _log(f"copying {clip_path}")
                payload["input"] = _copy_clip_input(clip_path, clip.source, self.job_dir)
                _log(f"sending to Pi4 server: {s.max_frames} frames max, stride {s.stride}, "
                     f"model {payload['max_size']}px, refine {'on' if payload['refine'] else 'off'}")
                self.result = _http(url, payload, timeout=None)
            except Exception as e:
                self.result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        _start_progress("Solve", prefs.server_url, job)
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        _redraw(context)
        if self._thread.is_alive():
            return {"PASS_THROUGH"}

        context.window_manager.event_timer_remove(self._timer)
        r = self.result or {"ok": False, "error": "no result"}
        if not r.get("ok"):
            _end_progress(False, r.get("error", "?"))
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}

        scene = context.scene
        clip = bpy.data.movieclips[self.clip_name]
        scene.cameramatch.last_job = self.job_dir
        _STATUS["stage"] = "import"
        _log("importing camera, lens, point cloud and masks into Blender")
        cam, lens, lens_std = import_result(scene, clip, self.job_dir, scene.cameramatch)
        text = (f"{r['frames']} frames, Pi4 {r['inference_s']}s | f={lens:.1f}mm | VRAM {r['peak_vram_gb']} GB")
        if "refine" in r:
            rr = r["refine"]
            text += (f" | refined {rr['final']['median']:.2f}px median, {rr['final']['p90']:.2f}px p90, "
                     f"k1 {rr['k1']:.3f}")
            text = guide.grade(rr, scene.cameramatch, None, tuple(clip.size))[1] + " | " + text
        _end_progress(True, text)
        self.report({"INFO"}, "Pi4 solve imported: " + text)
        if scene.cameramatch.auto_matte:
            bpy.ops.cameramatch.refine_matte("INVOKE_DEFAULT")
        return {"FINISHED"}


class CAMERAMATCH_OT_resolve(bpy.types.Operator):
    bl_idname = "cameramatch.resolve"
    bl_label = "Re-solve (Refine Only)"
    bl_description = ("Re-run only the plate-resolution tracking + bundle adjustment of the last solve with the current "
                      "settings (distortion, focal lock, zoom, rolling shutter, masks), without Pi4, then re-import")

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        j = context.scene.cameramatch.last_job
        return (_main_clip(context) is not None and not _STATUS["running"] and bool(j)
                and os.path.isfile(os.path.join(j, "pi4_cache.npz")))

    def invoke(self, context, event):
        s = context.scene.cameramatch
        clip = _main_clip(context)
        if _engine_down("Re-solve"):
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        self.job_dir = s.last_job
        self.clip_name = clip.name
        self.result = None
        job = os.path.basename(os.path.normpath(self.job_dir))
        mj = os.path.join(self.job_dir, "ignore_mask.json")
        if s.ignore_mask is not None:
            ignore_mask.rasterize(s.ignore_mask, clip, context.scene, self.job_dir, log=_log)
        elif os.path.isfile(mj):
            os.remove(mj)                                     # the mask was cleared: stop applying the old one
            _log("ignore mask cleared for this re-solve")
        payload = dict(job=job, **_refine_settings(s, clip))
        url = _prefs().server_url.rstrip("/") + "/refine"

        def worker():
            try:
                self.result = _http(url, payload, timeout=None)
            except Exception as e:
                self.result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        _start_progress("Re-solve", _prefs().server_url, job)
        _STATUS["stage"] = "klt"
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        _redraw(context)
        if self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        r = self.result or {"ok": False, "error": "no result"}
        if not r.get("ok"):
            _end_progress(False, "Re-solve failed: " + r.get("error", "?"))
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        sc = context.scene
        clip = bpy.data.movieclips[self.clip_name]
        frame = sc.frame_current
        remove_imported(sc, clip)
        cam, lens, _ = import_result(sc, clip, self.job_dir, sc.cameramatch)
        sc.frame_set(frame)
        rr = r["refine"]
        text = (guide.grade(rr, sc.cameramatch, None, tuple(clip.size))[1] + " | "
                f"re-solved in {rr.get('refine_s', '?')}s: {rr['final']['median']:.2f}px median, "
                f"{rr['final']['p90']:.2f}px p90, f={lens:.1f}mm, k1 {rr['k1']:.3f}"
                + (f", readout {rr.get('rolling_shutter', 0):+.2f}" if sc.cameramatch.rolling_shutter else ""))
        if bpy.data.collections.get(set_builder.SET_COLL):
            text += " | rebuild the set (Build Set + Lighting) for the new cameras"
        _end_progress(True, text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


class CAMERAMATCH_OT_refine_matte(bpy.types.Operator):
    bl_idname = "cameramatch.refine_matte"
    bl_label = "Refine Matte (MatAnyone 2)"
    bl_description = ("Foreground matte for the last solve: SAM 2.1 refines Pi4's motion mask, MatAnyone 2 "
                      "produces a soft temporally-stable alpha. Loads '<clip>_matte' and Pi4 inverse depth")

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        j = context.scene.cameramatch.last_job
        return (_main_clip(context) is not None and not _STATUS["running"] and j
                and os.path.isfile(os.path.join(j, "pi4_cache.npz")))

    def invoke(self, context, event):
        if _engine_down("Matte"):
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        self.job_dir = context.scene.cameramatch.last_job
        self.clip_name = _main_clip(context).name
        self.result = None
        url = _prefs().matte_server_url.rstrip("/") + "/matte"
        payload = dict(job=os.path.basename(os.path.normpath(self.job_dir)))

        def worker():
            try:
                self.result = _http(url, payload, timeout=None)
            except Exception as e:
                self.result = {"ok": False, "error": f"{type(e).__name__}: {e} (is matte-server running?)"}

        _start_progress("Matte", _prefs().matte_server_url, payload["job"])
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        _redraw(context)
        if self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        r = self.result or {"ok": False, "error": "no result"}
        if not r.get("ok"):
            _end_progress(False, "Matte failed: " + r.get("error", "?"))
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        clip = bpy.data.movieclips[self.clip_name]
        _STATUS["stage"] = "import"
        _log("loading matte clip and Pi4 depth sequence")
        mclip = import_matte(clip, self.job_dir)
        m = r.get("metrics", {})
        text = (f"Matte '{mclip.name}' in {r.get('seconds')}s | edges on image edges "
                f"{100 * m.get('edge_within_1p5px', 0):.0f}% | flicker {m.get('flicker_x1000', 0):.1f}")
        _end_progress(True, text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


class CAMERAMATCH_OT_show_log(bpy.types.Operator):
    bl_idname = "cameramatch.show_log"
    bl_label = "Show Log"
    bl_description = "Open the full CameraMatch log (all steps, server messages) in a Text Editor window"

    def execute(self, context):
        _sync_log_text()
        txt = bpy.data.texts[_LOG_TEXT]
        bpy.ops.wm.window_new()
        win = context.window_manager.windows[-1]
        area = max(win.screen.areas, key=lambda a: a.width * a.height)
        area.type = "TEXT_EDITOR"
        area.spaces.active.text = txt
        area.spaces.active.show_word_wrap = True
        return {"FINISHED"}


class CAMERAMATCH_OT_clear_log(bpy.types.Operator):
    bl_idname = "cameramatch.clear_log"
    bl_label = "Clear Log"
    bl_description = "Clear the CameraMatch log"

    def execute(self, context):
        _STATUS.update(log=[], synced=0)
        if _LOG_TEXT in bpy.data.texts:
            bpy.data.texts[_LOG_TEXT].clear()
        return {"FINISHED"}


class CAMERAMATCH_OT_reimport(bpy.types.Operator):
    bl_idname = "cameramatch.reimport"
    bl_label = "Re-import Last Result"
    bl_description = "Import the last solve again without re-running the model"

    @classmethod
    def poll(cls, context):
        j = context.scene.cameramatch.last_job
        return _main_clip(context) is not None and j and os.path.isfile(os.path.join(j, "result.npz"))

    def execute(self, context):
        s = context.scene.cameramatch
        remove_imported(context.scene, _main_clip(context))
        import_result(context.scene, _main_clip(context), s.last_job, s)
        return {"FINISHED"}


# --------------------------------------------------------------------------- CG integration
def _job_ok(context, fname="result.npz"):
    j = context.scene.cameramatch.last_job
    return bool(j) and os.path.isfile(os.path.join(j, fname))


_DERIVED = ("_motion_mask", "_matte", "_comp", "_check", "_preview")


def _main_clip(context):
    """The footage: the Footage pointer, else the clip shown in the Clip Editor, else the first plain clip."""
    s = context.scene.cameramatch
    if s.clip is not None:
        return s.clip
    clip = getattr(context, "edit_movieclip", None)
    if clip is not None and not clip.name.endswith(_DERIVED):
        return clip
    cands = [c for c in bpy.data.movieclips if not c.name.endswith(_DERIVED)]
    return cands[0] if cands else None


class CAMERAMATCH_OT_build_set(bpy.types.Operator):
    bl_idname = "cameramatch.build_set"
    bl_label = "Build Set + Lighting"
    bl_description = ("Low-poly set mesh + plate environment (matte server), then a fitted floor shadow catcher, "
                      "plate-projected walls (+ street proxy walls) and the plate environment as world lighting")

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return _job_ok(context) and not _STATUS["running"] and _main_clip(context) is not None

    def invoke(self, context, event):
        s = context.scene.cameramatch
        if _engine_down("Build Set"):
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        self.job_dir = s.last_job
        self.clip_name = _main_clip(context).name
        self.result = None
        url = _prefs().matte_server_url.rstrip("/") + "/set"
        payload = dict(job=os.path.basename(os.path.normpath(self.job_dir)), tris=s.set_tris, rebuild=s.set_rebuild)

        def worker():
            try:
                self.result = _http(url, payload, timeout=None)
            except Exception as e:
                self.result = {"ok": False, "error": f"{type(e).__name__}: {e} (is matte-server running?)"}

        _start_progress("Build Set", _prefs().matte_server_url, payload["job"])
        _STATUS["stage"] = "mesh"
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        _redraw(context)
        if self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        r = self.result or {"ok": False, "error": "no result"}
        if not r.get("ok"):
            _end_progress(False, "Build Set failed: " + r.get("error", "?"))
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        _STATUS["stage"] = "set"
        s = context.scene.cameramatch
        try:
            set_builder.build_set(context.scene, bpy.data.movieclips[self.clip_name], self.job_dir, _log,
                                  street_walls=s.set_street_walls, wall_height=s.set_wall_height)
        except Exception as e:
            _end_progress(False, f"Build Set failed: {e}")
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        text = (f"Set built: floor fitted to {context.scene['cm_floor_sigma'] * 100:.1f} cm, walls and plate "
                "lighting in 'CM_Set'. Next: 5. Add Characters")
        _end_progress(True, text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


class CAMERAMATCH_OT_place_on_floor(bpy.types.Operator):
    bl_idname = "cameramatch.place_on_floor"
    bl_label = "Place on Floor"
    bl_description = "Drop the selected objects so their lowest point rests on the fitted floor"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return "cm_floor_coef" in context.scene and any(o.type not in {"CAMERA", "LIGHT"}
                                                        for o in context.selected_objects)

    def execute(self, context):
        from mathutils import Vector
        n = 0
        for ob in context.selected_objects:
            if ob.type in {"CAMERA", "LIGHT"}:
                continue
            corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
            low = min(c.z for c in corners)
            c = ob.matrix_world.translation
            ob.location.z += set_builder.floor_z(context.scene, c.x, c.y) - low
            n += 1
        self.report({"INFO"}, f"placed {n} object(s) on the fitted floor")
        return {"FINISHED"}


class CAMERAMATCH_OT_add_robot(bpy.types.Operator):
    bl_idname = "cameramatch.add_robot"
    bl_label = "Add Walking Robot"
    bl_description = ("Procedural robot walking over the scene frame range, feet planted on the fitted floor: along the "
                      "selected curve (draw one on the floor), or from the 3D cursor straight toward the camera")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return "cm_floor_coef" in context.scene and context.scene.camera is not None and not _STATUS["running"]

    def execute(self, context):
        sc = context.scene
        s = sc.cameramatch
        crv = context.active_object if (context.active_object is not None and context.active_object.type == "CURVE"
                                        and context.active_object.select_get()) else None
        _log("===== Add Walking Robot " + (f"along curve '{crv.name}'" if crv else
                                            f"from cursor {tuple(round(x, 2) for x in sc.cursor.location)}") + " =====")
        try:
            fz = lambda x, y: set_builder.floor_z(sc, x, y)  # noqa: E731
            pxy = paths.curve_points(crv)[:, :2] if crv else None
            avoid_fn = None
            if s.avoid_collisions:
                kd, n_obs = collide.obstacle_tree(sc, fz)

                def avoid_fn(P, v):
                    P2, dmin, gap = collide.avoid(P, v, sc, 0.3, 0.3, kd, log=_log)
                    _log(f"collision avoidance: {n_obs} wall/object samples, closest wall {dmin:.2f} m, "
                         f"closest mover gap {gap if gap < 1e9 else float('inf'):.2f} m")
                    return P2
            coll, _, track, pelvis = robot.add_walking_robot(
                sc, sc.cursor.location.copy(), fz, _log, speed=s.robot_speed, look_at_camera=s.robot_look,
                path_xy=pxy, avoid_fn=avoid_fn)
            collide.register_mover(sc, pelvis.name, sc.frame_start, track, 0.3, 0.3)
        except Exception as e:
            self.report({"ERROR"}, f"Add robot failed: {e}")
            return {"CANCELLED"}
        _sync_log_text()
        self.report({"INFO"}, _STATUS["log"][-1][9:] if _STATUS["log"] else "robot added")
        return {"FINISHED"}


def _selected_pair(context):
    """Two 3D points to measure: 2 selected vertices (Edit Mode) or 2 selected objects. Returns (a, b, label)."""
    ob = context.active_object
    if ob is not None and ob.mode == "EDIT" and ob.type == "MESH":
        import bmesh
        bm = bmesh.from_edit_mesh(ob.data)
        vs = [v for v in bm.verts if v.select]
        if len(vs) == 2:
            return ob.matrix_world @ vs[0].co, ob.matrix_world @ vs[1].co, f"2 vertices of {ob.name}"
        return None, None, f"select exactly 2 vertices ({len(vs)} selected)"
    obs = list(context.selected_objects)
    if len(obs) == 2:
        a, b = obs[0].matrix_world.translation.copy(), obs[1].matrix_world.translation.copy()
        return a, b, f"objects {obs[0].name} / {obs[1].name}"
    return None, None, "select 2 vertices (Edit Mode on the point cloud) or 2 objects (e.g. empties snapped to it)"


class CAMERAMATCH_OT_set_scale(bpy.types.Operator):
    bl_idname = "cameramatch.set_scale"
    bl_label = "Set Scale from Distance"
    bl_description = ("Rescale the whole solve (camera path, point cloud, tracks, set) so the distance between two "
                      "selected points (2 vertices in Edit Mode, or 2 objects) equals a real measurement")
    bl_options = {"REGISTER", "UNDO"}

    real_distance: FloatProperty(name="Real Distance", default=1.0, min=1e-4, unit="LENGTH",
                                 description="The true distance between the two selected points")
    measured: FloatProperty(name="Measured", default=0.0, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _job_ok(context) and not _STATUS["running"] and _main_clip(context) is not None

    def invoke(self, context, event):
        a, b, label = _selected_pair(context)
        if a is None:
            self.report({"ERROR"}, label)
            return {"CANCELLED"}
        self.measured = (a - b).length
        if self.measured < 1e-6:
            self.report({"ERROR"}, "the two points coincide")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        col = self.layout.column()
        col.label(text=f"Measured: {self.measured:.4f} (scene units)")
        col.prop(self, "real_distance")

    def execute(self, context):
        if self.measured <= 0:
            a, b, label = _selected_pair(context)
            if a is None:
                self.report({"ERROR"}, label)
                return {"CANCELLED"}
            self.measured = (a - b).length
        sc = context.scene
        s = sc.cameramatch
        job = s.last_job
        old = float(sc.get("cm_applied_scale", 1.0))   # scale of the solve currently in the scene
        new = old * self.real_distance / self.measured
        if context.active_object is not None and context.active_object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        with open(os.path.join(job, "scene_scale.json"), "w") as fh:
            json.dump({"scale": new, "measured": self.measured, "real": self.real_distance}, fh)
        clip = _main_clip(context)
        frame = sc.frame_current
        remove_imported(sc, clip)
        import_result(sc, clip, job, s)
        sc.frame_set(frame)
        msg = f"scale x{new / old:.4f} (job scale {new:.4f}): {self.measured:.3f} -> {self.real_distance:.3f}"
        if bpy.data.collections.get(set_builder.SET_COLL):
            set_builder.build_set(sc, clip, job, _log, street_walls=s.set_street_walls,
                                  wall_height=s.set_wall_height)
            msg += "; set rebuilt"
        if any(c.name.startswith("CM_Robot") for c in bpy.data.collections):
            msg += "; re-add robots (they were placed at the old scale)"
        _log("Set Scale: " + msg)
        _sync_log_text()
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class CAMERAMATCH_OT_export_tracks(bpy.types.Operator):
    bl_idname = "cameramatch.export_tracks"
    bl_label = "Export Tracks to Clip"
    bl_description = ("Write the refined KLT tracks into the clip's tracker (Movie Clip Editor tracks named "
                      "Pi4_####), replacing previously exported ones")
    bl_options = {"REGISTER", "UNDO"}

    max_tracks: IntProperty(name="Max Tracks", default=1500, min=1, max=100000)

    @classmethod
    def poll(cls, context):
        return _job_ok(context) and _main_clip(context) is not None

    def execute(self, context):
        clip = _main_clip(context)
        d = np.load(os.path.join(context.scene.cameramatch.last_job, "result.npz"))
        key = "track_obs_full" if "track_obs_full" in d else ("track_obs" if "track_obs" in d else None)
        if key is None:
            self.report({"ERROR"}, "no refined tracks in this solve (enable Refine at Plate Resolution)")
            return {"CANCELLED"}
        ob = d[key]
        W, H = clip.size
        fidx = d["frame_indices"].astype(int)
        tracks = clip.tracking.tracks
        for t in [t for t in tracks if t.name.startswith("Pi4_")]:
            tracks.remove(t)
        tid = ob[:, 1].astype(np.int64)
        order = np.lexsort((ob[:, 0], tid))
        ob, tid = ob[order], tid[order]
        starts = np.r_[0, np.nonzero(np.diff(tid))[0] + 1]
        ends = np.r_[starts[1:], len(ob)]
        pick = np.argsort(-(ends - starts))[: self.max_tracks]          # longest tracks first
        n_mark = 0
        for k in pick:
            rows = ob[starts[k]:ends[k]]
            fr = fidx[rows[:, 0].astype(int)] + 1                       # clip-local frame numbers start at 1
            tr = tracks.new(name=f"Pi4_{int(tid[starts[k]]):04d}", frame=int(fr[0]))
            for f_, u, v in zip(fr, rows[:, 2], rows[:, 3]):
                tr.markers.insert_frame(int(f_), co=((u + 0.5) / W, 1.0 - (v + 0.5) / H))
            end = tr.markers.insert_frame(int(fr[-1]) + 1, co=((rows[-1, 2] + 0.5) / W, 1.0 - (rows[-1, 3] + 0.5) / H))
            end.mute = True
            tr.select = False
            n_mark += len(rows)
        msg = f"exported {len(pick)} tracks ({n_mark} markers) from {key}"
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _path_curve(context):
    """The path: the active object if it is a curve, else the only selected curve."""
    a = context.active_object
    if a is not None and a.type == "CURVE" and a.select_get():
        return a
    crvs = [o for o in context.selected_objects if o.type == "CURVE"]
    return crvs[0] if len(crvs) == 1 else None


class CAMERAMATCH_OT_move_along_path(bpy.types.Operator):
    bl_idname = "cameramatch.move_along_path"
    bl_label = "Walk / Move Along Path"
    bl_description = ("Animate the selected objects (character with an in-place walk cycle, vehicle, prop...) along "
                      "the selected curve over the scene frame range, standing on the fitted floor and facing the "
                      "way they move. Characters keep their feet planted and avoid walls and other movers")
    bl_options = {"REGISTER", "UNDO"}

    speed: FloatProperty(name="Speed", default=0.0, min=0.0, max=100.0, unit="VELOCITY",
                         description="Metres per second. 0 = the whole curve over the frame range (props), or the "
                                     "walk cycle's own speed (characters). For characters a set speed scales the "
                                     "cycle's playback so the feet stay planted")
    action: StringProperty(name="Walk Action", default="",
                           description="Character walk cycle (empty: the action named like 'walk', else the "
                                       "active one)")
    forward: bpy.props.EnumProperty(name="Forward Axis", default="-Y",
                                    items=[("-Y", "-Y", "Blender character convention"), ("+Y", "+Y", ""),
                                           ("+X", "+X", ""), ("-X", "-X", "")],
                                    description="The object's local axis that should point along the path")

    @classmethod
    def poll(cls, context):
        return ("cm_floor_coef" in context.scene and _path_curve(context) is not None
                and any(o is not _path_curve(context) for o in context.selected_objects))

    def execute(self, context):
        sc = context.scene
        s = sc.cameramatch
        crv = _path_curve(context)
        fz = lambda x, y: set_builder.floor_z(sc, x, y)  # noqa: E731
        fps = sc.render.fps / sc.render.fps_base
        dur = max((sc.frame_end - sc.frame_start) / fps, 1e-3)
        roots = []
        for o in context.selected_objects:
            if o is not crv:
                r = characters.hierarchy_root(o)
                if r not in roots and r is not crv:
                    roots.append(r)
        msgs = []
        try:
            pts = paths.curve_points(crv)[:, :2]
            _, L0 = paths.ground_path(pts)
            kd = None
            if s.avoid_collisions:
                excl = set()
                for r in roots:
                    stack = [r]
                    while stack:
                        o = stack.pop()
                        excl.add(o.name)
                        stack.extend(o.children)
                kd, n_obs = collide.obstacle_tree(sc, fz, exclude=excl)
            for r in roots:
                arm = characters.find_armature(r)
                info = None
                if arm is not None and (arm.animation_data or bpy.data.actions):
                    act = characters.walk_action(arm, self.action)
                    info = characters.analyse_walk(sc, r, arm, act)
                    if info["hip_travel_m"] > 0.25:                     # an in-place cycle ends where it began
                        # root motion (Mixamo walk without 'In Place'): walk on the spot, the path does the travel
                        act = characters.make_in_place(act, info["hips"])
                        _log(f"'{r.name}': '{act.name[:-8]}' moves its hips {info['hip_travel_m']:.2f} m per cycle by "
                             f"itself (root motion): using an in-place copy '{act.name}'")
                        info = characters.analyse_walk(sc, r, arm, act)
                    if info["profile_mean_mpf"] * fps < 0.25:
                        raise RuntimeError(
                            f"'{act.name}' does not look like a walk cycle (its planted foot moves "
                            f"{info['profile_mean_mpf'] * fps:.2f} m/s). Pick the walk in Walk Action, or download a "
                            "Mixamo walk (FBX, With Skin, In Place ticked)")
                    v_cycle = info["profile_mean_mpf"] * fps
                    v = self.speed if self.speed > 0 else v_cycle
                    rate = v / max(v_cycle, 1e-9)                       # cycle playback follows the travel speed
                    tr = characters.loop_action(sc, arm, act)
                    tr.strips[0].scale = 1.0 / rate
                    tr.strips[0].repeat = max(1.0, (sc.frame_end - sc.frame_start + 1) * rate / info["cycle"] + 1)
                    s_frames = characters.arc_positions(info, sc, rate)
                    fwd = info["forward"]
                    yaw_off = paths.object_euler(r).z - math.atan2(fwd[1], fwd[0])
                    z_off = info["sole_above"]
                    hw, hl = paths.footprint(r, fwd)
                    if info["root_motion"] > 0.5:
                        _log(f"warning: '{act.name}' moves its hips forward by itself (root motion "
                             f"{info['root_motion'] * 100:.0f}%): use an in-place walk cycle")
                else:
                    v = self.speed if self.speed > 0 else L0 / dur
                    s_frames = None
                    yaw_off = paths.FORWARD_YAW[self.forward]
                    zmin = min((r.matrix_basis.to_3x3() @ Vector(c)).z for c in r.bound_box) \
                        if r.type != "EMPTY" else 0.0
                    z_off = -zmin
                    fl = -yaw_off                                       # local forward direction angle
                    rz = paths.object_euler(r).z
                    hw, hl = paths.footprint(r, (math.cos(fl + rz), math.sin(fl + rz)))
                P = pts
                t_arr = np.arange(sc.frame_end - sc.frame_start + 1) / fps
                if s_frames is None:                       # props: fixed speed, or the whole curve over the shot
                    rule = (lambda L, v_=self.speed: np.minimum(v_ * t_arr, L)) if self.speed > 0 else \
                        (lambda L: np.minimum(L / dur * t_arr, L))
                else:
                    rule = s_frames
                if kd is not None or collide.movers(sc):
                    P, dmin, gap = collide.avoid(pts, rule, sc, hw, hl, kd, self_name=r.name, log=_log)
                    _log(f"'{r.name}': closest wall {dmin:.2f} m, closest mover gap "
                         f"{gap if gap < 1e9 else float('inf'):.2f} m")
                path, length = paths.ground_path(P)
                if v * dur > length + 0.05:
                    _log(f"warning: '{r.name}' reaches the end of the curve after {length / v:.1f} s "
                         f"(shot {dur:.1f} s): extend the curve")
                track = paths.move_one(r, path, length, fz, sc, rule(length) if callable(rule) else rule,
                                       yaw_off, z_off)
                arrival = sc.frame_end
                if s_frames is not None and s_frames[-1] > length:
                    arrival = sc.frame_start + int(np.searchsorted(s_frames, length))
                    idle = characters.finish_at(sc, arm, tr, info, rate, arrival)
                    _log(f"'{r.name}' arrives at frame {arrival}: walk cycle ends there"
                         + (f", then '{idle}'" if idle else " (last pose held)"))
                collide.register_mover(sc, r.name, sc.frame_start, track, hw, hl)
                msg = f"'{r.name}' {length:.1f} m at {v:.2f} m/s"
                if info is not None:
                    sl = characters.foot_slide(sc, arm, info["feet"], f1=arrival)
                    msg += (f", walk '{info['action']}' {info['profile_mean_mpf'] * fps:.2f} m/s, "
                            f"planted-foot slide {sl['median_cm_s']:.1f} cm/s median")
                _log(msg)
                msgs.append(msg)
        except Exception as e:
            import traceback
            _log("Move Along Path failed: " + traceback.format_exc().replace(chr(10), " | "))
            _sync_log_text()
            self.report({"ERROR"}, f"Move Along Path failed: {e}")
            return {"CANCELLED"}
        _sync_log_text()
        self.report({"INFO"}, "; ".join(msgs))
        return {"FINISHED"}


_RENDER = {"count": 0, "done": False, "cancel": False}


def _on_render_post(*_):
    _RENDER["count"] += 1


def _on_render_complete(*_):
    _RENDER["done"] = True


def _on_render_cancel(*_):
    _RENDER["cancel"] = True


def _render_handlers(add):
    for lst, fn in ((bpy.app.handlers.render_post, _on_render_post),
                    (bpy.app.handlers.render_complete, _on_render_complete),
                    (bpy.app.handlers.render_cancel, _on_render_cancel)):
        if add and fn not in lst:
            lst.append(fn)
        elif not add and fn in lst:
            lst.remove(fn)


class CAMERAMATCH_OT_render_comp(bpy.types.Operator):
    bl_idname = "cameramatch.render_comp"
    bl_label = "Render & Comp"
    preview: BoolProperty(name="Preview", default=False, options={"SKIP_SAVE"},
                          description="Fast check: half resolution, 24 samples, comp at that size")
    bl_description = ("Cycles render of the scene range (multilayer EXR with the Shadow Catcher pass), then the "
                      "matte server composites it over the plate (shadow pass, measured black level / grain, "
                      "light wrap) and encodes <clip>_comp.mp4 + plate_vs_comp.mp4; the comp loads as a clip")

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return _job_ok(context) and not _STATUS["running"] and context.scene.camera is not None

    def invoke(self, context, event):
        sc = context.scene
        s = sc.cameramatch
        if _engine_down("Render & Comp"):
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        self.job_dir = s.last_job
        self.job = os.path.basename(os.path.normpath(self.job_dir))
        clip = _main_clip(context)
        self.clip_name = clip.name if clip else "plate"
        self.tag = "preview" if self.preview else "render"
        render_dir = os.path.join(self.job_dir, self.tag)
        samples = 24 if self.preview else s.render_samples
        for f in (os.listdir(render_dir) if os.path.isdir(render_dir) else []):
            if f.startswith("r_") and f.endswith(".exr"):         # never comp a stale frame of another size
                os.remove(os.path.join(render_dir, f))
        hide_camera_blockers(sc, _log)
        set_builder.setup_render(sc, render_dir, samples, s.render_motion_blur, log=_log,
                                 percentage=50 if self.preview else 100)
        sc.render.use_persistent_data = True       # no full scene re-sync per frame
        self.frames = list(range(sc.frame_start, sc.frame_end + 1))
        self.fps = sc.render.fps / sc.render.fps_base
        self.phase = "render"
        self.result = None
        _STATUS.update(running=True, pct=0.0, stage="render", op="Preview" if self.preview else "Render & Comp",
                       t0=time.time(), t_end=0.0, text="warm-up render", server="", job=self.job, cancel=False,
                       need_engine=False)
        _log(f"===== {'Preview' if self.preview else 'Render & Comp'} | job {self.job} | frames "
             f"{self.frames[0]}-{self.frames[-1]}, {samples} samples =====")
        # warm-up: the first movie-texture access in a session can come back black
        smp = sc.cycles.samples
        sc.cycles.samples = 1
        sc.frame_set(self.frames[0])
        bpy.ops.render.render(write_still=False)
        sc.cycles.samples = smp
        _RENDER.update(count=0, done=False, cancel=False)
        _render_handlers(True)
        _log(f"rendering {len(self.frames)} frames to {render_dir}")
        if bpy.app.background:
            bpy.ops.render.render(animation=True)
            _RENDER["done"] = True
        else:
            bpy.ops.render.render("INVOKE_DEFAULT", animation=True)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _start_comp(self, context):
        s = context.scene.cameramatch
        url = _prefs().matte_server_url.rstrip("/") + "/comp"
        payload = dict(job=self.job, render_dir=self.tag, out_dir="comp_preview" if self.preview else "comp",
                       frames=self.frames, fps=self.fps, soft=s.comp_soft * (0.5 if self.preview else 1.0),
                       wrap=s.comp_wrap)

        def worker():
            try:
                self.result = _http(url, payload, timeout=None)
            except Exception as e:
                self.result = {"ok": False, "error": f"{type(e).__name__}: {e} (is matte-server running?)"}

        t0 = _STATUS["t0"]
        _start_progress("Preview" if self.preview else "Render & Comp", _prefs().matte_server_url, self.job)
        _STATUS.update(t0=t0, stage="comp", pct=0.0)
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self.phase = "comp"

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self.phase == "render":
            n = min(_RENDER["count"], len(self.frames))
            _STATUS["pct"] = n / len(self.frames)
            _STATUS["text"] = f"rendered {n}/{len(self.frames)} frames"
            _redraw(context)
            if _RENDER["cancel"] or _STATUS["cancel"]:
                _render_handlers(False)
                context.window_manager.event_timer_remove(self._timer)
                _end_progress(False, f"render cancelled after {n} frames")
                return {"CANCELLED"}
            if _RENDER["done"]:
                _render_handlers(False)
                _log(f"render done: {n} frames in {time.time() - _STATUS['t0']:.0f}s")
                self._start_comp(context)
            return {"PASS_THROUGH"}
        _redraw(context)
        if self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        r = self.result or {"ok": False, "error": "no result"}
        if not r.get("ok"):
            _end_progress(False, "Comp failed: " + r.get("error", "?"))
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        video = os.path.join(self.job_dir, r["video"])
        name = f"{self.clip_name}_{'preview' if self.preview else 'comp'}"
        old = bpy.data.movieclips.get(name)
        if old:
            bpy.data.movieclips.remove(old)
        mc = bpy.data.movieclips.load(video)
        mc.name = name
        mc.frame_start = self.frames[0]
        sc = context.scene
        sc.render.resolution_percentage = 100
        _show_clip(context, mc)
        sc["cm_last_comp"] = video
        text = f"{r['frames']} frames composited -> {video} (clip '{name}'; side by side: {r['side_by_side']})"
        _end_progress(True, text)
        self.report({"INFO"}, text)
        return {"FINISHED"}



# --------------------------------------------------------------------------- ease of use: engine, footage, check
def _show_clip(context, clip):
    """Show a clip in every Movie Clip Editor (the footage, an overlay check, a comp)."""
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.spaces.active.clip = clip


class CAMERAMATCH_OT_start_engine(bpy.types.Operator):
    bl_idname = "cameramatch.start_engine"
    bl_label = "Start Engine"
    bl_description = ("Start Docker (if needed) and the CameraMatch engine (Pi4 solver + matte/set/comp server), "
                      "then wait until it answers. The first start downloads the models (several GB)")

    @classmethod
    def poll(cls, context):
        return not engine.STATE["action"]

    def execute(self, context):
        d = _engine_dir()
        if not d:
            self.report({"ERROR"}, "CameraMatch folder not found: set the Work folder (or Engine folder) in the "
                                   "add-on preferences")
            return {"CANCELLED"}
        p = _prefs()
        engine.POLLER.urls = (p.server_url, p.matte_server_url)
        _STATUS["need_engine"] = False
        _log(f"===== Start Engine ({d}) =====")
        engine.start(d, p.server_url, p.matte_server_url, log=_log)
        return {"FINISHED"}


class CAMERAMATCH_OT_stop_engine(bpy.types.Operator):
    bl_idname = "cameramatch.stop_engine"
    bl_label = "Stop Engine"
    bl_description = "Stop the engine's containers and free the GPU memory they hold"

    @classmethod
    def poll(cls, context):
        return not engine.STATE["action"] and not _STATUS["running"]

    def execute(self, context):
        d = _engine_dir()
        if not d:
            self.report({"ERROR"}, "CameraMatch folder not found (add-on preferences)")
            return {"CANCELLED"}
        engine.stop(d, log=_log)
        return {"FINISHED"}


class CAMERAMATCH_OT_cancel(bpy.types.Operator):
    bl_idname = "cameramatch.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running job (it stops at its next step; the model's single pass cannot be cut)"

    @classmethod
    def poll(cls, context):
        return _STATUS["running"]

    def execute(self, context):
        if _STATUS["stage"] == "render":
            self.report({"WARNING"}, "Rendering: press Esc in the render window to stop it")
            return {"CANCELLED"}
        _STATUS["cancel"] = True
        if _STATUS["server"] and _STATUS["job"]:
            try:
                _http(f"{_STATUS['server'].rstrip('/')}/cancel?job={urllib.parse.quote(_STATUS['job'])}",
                      timeout=3)
            except Exception as e:
                _log(f"cancel request failed: {e}")
        _STATUS["text"] = "cancelling..."
        _log("cancel requested")
        return {"FINISHED"}


class CAMERAMATCH_OT_load_footage(bpy.types.Operator, ImportHelper):
    bl_idname = "cameramatch.load_footage"
    bl_label = "Load Footage"
    bl_description = "Open a video (or the first image of a sequence) as the shot to match"
    filter_glob: StringProperty(default="*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.mxf;*.png;*.jpg;*.jpeg",
                                options={"HIDDEN"})

    def execute(self, context):
        return _load_footage(self, context, self.filepath)


def _load_footage(op, context, path):
    try:
        clip = bpy.data.movieclips.load(path, check_existing=True)
    except RuntimeError as e:
        op.report({"ERROR"}, f"Cannot open {os.path.basename(path)}: {e}")
        return {"CANCELLED"}
    sc = context.scene
    sc.cameramatch.clip = clip
    sc.frame_start = clip.frame_start
    sc.frame_end = clip.frame_start + clip.frame_duration - 1
    sc.render.resolution_x, sc.render.resolution_y = clip.size
    if clip.fps > 0:
        sc.render.fps, sc.render.fps_base = round(clip.fps), round(clip.fps) / clip.fps
    _show_clip(context, clip)
    W, H = clip.size
    kind = " (portrait: phone footage, rolling shutter will be solved)" if H > W else ""
    msg = f"Footage '{clip.name}': {W}x{H}, {clip.frame_duration} frames at {clip.fps:.2f} fps{kind}"
    _log(msg)
    op.report({"INFO"}, msg)
    return {"FINISHED"}


class CAMERAMATCH_OT_show_footage(bpy.types.Operator):
    bl_idname = "cameramatch.show_footage"
    bl_label = "Show Footage"
    bl_description = "Show the original footage again in the Movie Clip Editor"

    @classmethod
    def poll(cls, context):
        return _main_clip(context) is not None

    def execute(self, context):
        _show_clip(context, _main_clip(context))
        return {"FINISHED"}


def _http_job(op, context, url, payload, label, stage, on_done):
    """Shared modal pattern: POST in a thread, mirror /progress, call on_done(result) on the main thread."""
    op.result = None

    def worker():
        try:
            op.result = _http(url, payload, timeout=None)
        except Exception as e:
            op.result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    _start_progress(label, url.rsplit("/", 1)[0], payload["job"])
    _STATUS["stage"] = stage
    op._on_done = on_done
    op._thread = threading.Thread(target=worker, daemon=True)
    op._thread.start()
    op._timer = context.window_manager.event_timer_add(0.5, window=context.window)
    context.window_manager.modal_handler_add(op)
    return {"RUNNING_MODAL"}


def _http_job_modal(op, context, event):
    if event.type != "TIMER":
        return {"PASS_THROUGH"}
    _redraw(context)
    if op._thread.is_alive():
        return {"PASS_THROUGH"}
    context.window_manager.event_timer_remove(op._timer)
    r = op.result or {"ok": False, "error": "no result"}
    if not r.get("ok"):
        _end_progress(False, f"{_STATUS['op']} failed: " + r.get("error", "?"))
        op.report({"ERROR"}, _STATUS["text"])
        return {"CANCELLED"}
    try:
        text = op._on_done(context, r)
    except Exception as e:
        _end_progress(False, f"{_STATUS['op']} failed: {e}")
        op.report({"ERROR"}, _STATUS["text"])
        return {"CANCELLED"}
    _end_progress(True, text)
    op.report({"INFO"}, text)
    return {"FINISHED"}


class CAMERAMATCH_OT_check_overlay(bpy.types.Operator):
    bl_idname = "cameramatch.check_overlay"
    bl_label = "Check Overlay"
    bl_description = ("Draw a grid on the ground with the solved camera into every frame and measure whether it "
                      "stays glued to the ground texture. Loads the result as '<clip>_check'")

    @classmethod
    def poll(cls, context):
        return _job_ok(context) and not _STATUS["running"]

    def invoke(self, context, event):
        if _engine_down("Check"):
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        self.job_dir = context.scene.cameramatch.last_job
        clip = _main_clip(context)
        self.clip_name = clip.name if clip else "plate"
        job = os.path.basename(os.path.normpath(self.job_dir))
        return _http_job(self, context, _prefs().matte_server_url.rstrip("/") + "/check", dict(job=job),
                         "Check", "check", self._done)

    def _done(self, context, r):
        name = f"{self.clip_name}_check"
        old = bpy.data.movieclips.get(name)
        if old:
            bpy.data.movieclips.remove(old)
        mc = bpy.data.movieclips.load(os.path.join(self.job_dir, r["video"]))
        mc.name = name
        clip = _main_clip(context)
        mc.frame_start = clip.frame_start if clip else 1
        _show_clip(context, mc)
        m = r.get("metrics", {})
        st = m.get("stick_med_px")
        verdict = ("grid stays on the ground" if st is not None and st < 1.5 else
                   "grid slides a little" if st is not None and st < 3 else "grid slides")
        return (f"Overlay check: {verdict} ({st} px median over {m.get('stick_samples', 0)} samples). "
                f"Play '{name}' in the Clip Editor; Show Footage switches back")

    def modal(self, context, event):
        return _http_job_modal(self, context, event)


FIXES = {
    "ignore_nonrigid": ("Turn on Ignore Trees / Water / Sky and re-solve", lambda s: setattr(s, "ignore_nonrigid", True)),
    "rs": ("Turn on Rolling Shutter and re-solve", lambda s: setattr(s, "rs_mode", "ON")),
    "lock_focal": ("Lock the focal and re-solve", lambda s: setattr(s, "lock_focal", True)),
    "no_distortion": ("Turn off Lens Distortion and re-solve", lambda s: setattr(s, "solve_distortion", False)),
}


class CAMERAMATCH_OT_apply_fix(bpy.types.Operator):
    bl_idname = "cameramatch.apply_fix"
    bl_label = "Fix"
    bl_description = "Apply this suggestion"
    fix: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def description(cls, context, props):
        if props.fix == "mask":
            return "Create an Ignore Mask and switch the Clip Editor to Mask mode: draw over the moving area"
        return FIXES.get(props.fix, ("Apply this suggestion",))[0]

    def invoke(self, context, event):
        s = context.scene.cameramatch
        if self.fix == "mask":
            m = s.ignore_mask or bpy.data.masks.new("Ignore")
            s.ignore_mask = m
            clip = _main_clip(context)
            done = False
            for win in context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type == "CLIP_EDITOR":
                        sp = area.spaces.active
                        sp.clip = clip
                        sp.mode = "MASK"
                        sp.mask = m
                        done = True
            self.report({"INFO"}, "Mask mode: Ctrl+click to draw a shape over the moving area, then Re-solve"
                        if done else "Ignore Mask created: open a Movie Clip Editor (Mask mode) to draw it")
            return {"FINISHED"}
        if self.fix not in FIXES:
            return {"CANCELLED"}
        FIXES[self.fix][1](s)
        _log(f"suggestion applied: {FIXES[self.fix][0]}")
        if bpy.ops.cameramatch.resolve.poll():
            return bpy.ops.cameramatch.resolve("INVOKE_DEFAULT")
        return {"FINISHED"}


# --------------------------------------------------------------------------- characters
class CAMERAMATCH_OT_add_character(bpy.types.Operator, ImportHelper):
    bl_idname = "cameramatch.add_character"
    bl_label = "Add Character"
    bl_description = ("Import a rigged character (glTF / GLB / FBX, e.g. from Mixamo with an in-place walk), scale "
                      "it to Character Height and stand it on the floor at the 3D cursor")
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: StringProperty(default="*.glb;*.gltf;*.fbx", options={"HIDDEN"})

    def execute(self, context):
        return add_character(self, context, self.filepath)


def add_character(op, context, path):
    sc = context.scene
    s = sc.cameramatch
    before_o, before_a = set(bpy.data.objects), set(bpy.data.actions)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=True)
        else:
            raise RuntimeError(f"unsupported file type {ext}")
    except Exception as e:
        op.report({"ERROR"}, f"Import failed: {e}")
        return {"CANCELLED"}
    new = [o for o in bpy.data.objects if o not in before_o]
    arm = next((o for o in new if o.type == "ARMATURE"), None)
    if arm is None:
        op.report({"ERROR"}, "No armature in this file: Add Character needs a rigged character")
        return {"CANCELLED"}
    root = characters.hierarchy_root(arm)
    name = os.path.splitext(os.path.basename(path))[0]
    coll = bpy.data.collections.new(f"CM_Character_{name}")
    sc.collection.children.link(coll)
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    new_acts = [a for a in bpy.data.actions if a not in before_a]
    arm["cm_actions"] = [a.name for a in new_acts]
    # importers leave one active NLA track per animation (they would all play at once under the walk) and may
    # key the armature object's own transform: the add-on owns placement, the animation lives in the bones
    ad = arm.animation_data
    if ad is not None:
        for t in list(ad.nla_tracks):
            ad.nla_tracks.remove(t)
        ad.action = None
    n_obj = sum(characters.strip_object_transform(a) for a in new_acts)
    if n_obj:
        _log(f"'{arm.name}': removed {n_obj} object-transform channels from its actions (placement is set by the "
             "add-on)")
    context.view_layer.update()
    meshes = [o for o in new if o.type == "MESH" and characters.hierarchy_root(o) is root]
    for o in new:                     # glTF files can ship stray helper meshes outside the rig: hide them
        if o.type == "MESH" and characters.hierarchy_root(o) is not root:
            o.hide_render = o.hide_viewport = True
    zs = [(o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box]
    if s.char_height > 0 and zs and max(zs) - min(zs) > 1e-6:
        k = s.char_height / (max(zs) - min(zs))
        root.scale = tuple(v * k for v in root.scale)
    context.view_layer.update()
    cur = sc.cursor.location
    root.location.x, root.location.y = cur.x, cur.y
    if "cm_floor_coef" in sc:
        context.view_layer.update()
        zs = [(o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box]
        if zs:
            root.location.z += set_builder.floor_z(sc, cur.x, cur.y) - min(zs)
    for o in sc.objects:
        o.select_set(False)
    root.select_set(True)
    arm.select_set(True)
    context.view_layer.objects.active = root
    acts = arm["cm_actions"]
    msg = (f"'{name}' added ({len(acts)} action(s){': ' + ', '.join(acts[:4]) if acts else ''}). Next: select it "
           "and a path curve, then Walk Along Path, or put the 3D cursor where it should go and Walk to Cursor")
    _log(msg)
    op.report({"INFO"}, msg)
    return {"FINISHED"}


def _new_floor_curve(context, name, pts):
    sc = context.scene
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        z = set_builder.floor_z(sc, p[0], p[1]) if "cm_floor_coef" in sc else 0.0
        bp.co = (p[0], p[1], z + 0.02)
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    ob = bpy.data.objects.new(name, cu)
    sc.collection.objects.link(ob)
    ob.hide_render = True
    return ob


class CAMERAMATCH_OT_new_path(bpy.types.Operator):
    bl_idname = "cameramatch.new_path"
    bl_label = "New Path"
    bl_description = ("Add a path curve on the floor from the 3D cursor, heading away from the camera, and enter "
                      "Edit Mode to shape it (G to move points, E to extend)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return "cm_floor_coef" in context.scene and context.mode == "OBJECT"

    def execute(self, context):
        sc = context.scene
        c = sc.cursor.location
        d = Vector((0.0, 1.0))
        if sc.camera is not None:
            fwd = sc.camera.matrix_world.to_3x3() @ Vector((0, 0, -1))
            if fwd.xy.length > 1e-3:
                d = fwd.xy.normalized()
        pts = [(c.x + d.x * t, c.y + d.y * t) for t in (0.0, 2.0, 4.0, 6.0)]
        ob = _new_floor_curve(context, "CM_path", pts)
        for o in sc.objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode="EDIT")
        self.report({"INFO"}, "Shape the path (G move, E extend), Tab to finish, then select the character too "
                              "and Walk Along Path")
        return {"FINISHED"}


class CAMERAMATCH_OT_walk_to_cursor(bpy.types.Operator):
    bl_idname = "cameramatch.walk_to_cursor"
    bl_label = "Walk to Cursor"
    bl_description = ("The selected character (or object) walks from where it stands to the 3D cursor over the "
                      "shot, feet planted, around walls and other movers")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return ("cm_floor_coef" in context.scene and context.mode == "OBJECT"
                and any(o.type != "CURVE" for o in context.selected_objects))

    def execute(self, context):
        sc = context.scene
        roots = []
        for o in context.selected_objects:
            r = characters.hierarchy_root(o)
            if r.type != "CURVE" and r not in roots:
                roots.append(r)
        a = roots[0].matrix_world.translation
        b = sc.cursor.location
        if (b.xy - a.xy).length < 0.3:
            self.report({"ERROR"}, "Put the 3D cursor where the character should go (Shift+right-click)")
            return {"CANCELLED"}
        pts = [(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t) for t in (0.0, 0.34, 0.67, 1.0)]
        crv = _new_floor_curve(context, f"CM_path_{roots[0].name}", pts)
        for o in sc.objects:
            o.select_set(o in roots or o is crv)
        context.view_layer.objects.active = crv
        with context.temp_override(active_object=crv, selected_objects=roots + [crv]):
            return bpy.ops.cameramatch.move_along_path()


# --------------------------------------------------------------------------- export to other software
class CAMERAMATCH_OT_export_scene(bpy.types.Operator):
    bl_idname = "cameramatch.export_scene"
    bl_label = "Export Camera + Scene"
    bl_description = ("Export the solved camera, set geometry and track points for other software: Alembic, FBX, "
                      "USD, a Nuke script, lens-distortion ST maps and camera.json / .csv")

    directory: StringProperty(subtype="DIR_PATH", options={"SKIP_SAVE"})
    use_abc: BoolProperty(name="Alembic (.abc)", default=True, description="Maya, Houdini, Nuke, C4D, Unreal")
    use_fbx: BoolProperty(name="FBX", default=True, description="Maya, 3ds Max, Unreal, Unity")
    use_usd: BoolProperty(name="USD (.usdc)", default=True, description="Houdini Solaris, Omniverse, Unreal, Maya")
    use_nk: BoolProperty(name="Nuke script (.nk)", default=True,
                         description="Camera3, plate, undistort / redistort STMaps, ReadGeo of the Alembic")
    use_stmap: BoolProperty(name="Lens ST maps (EXR)", default=True,
                            description="Exact undistort / redistort maps for the solved distortion (any compositor)")
    use_json: BoolProperty(name="camera.json / .csv", default=True,
                           description="Per-frame matrices, focal, sensor, distortion, rolling shutter")
    include_set: BoolProperty(name="Set Geometry", default=True, description="Floor and walls (Build Set)")
    include_tracks: BoolProperty(name="Track Points", default=True, description="The solved 3D track points")
    n_locators: IntProperty(name="Locators", default=150, min=0, max=5000,
                            description="Empties at the most observed track points (0: none)")
    include_cg: BoolProperty(name="CG Characters", default=False,
                             description="Also export the characters / robots you added (animated)")
    use_plate: BoolProperty(name="Plate (PNG sequence)", default=True,
                            description="The footage as plate/plate_####.png (about 2 MB a frame at 1080p)")
    use_masks: BoolProperty(name="Masks", default=True,
                            description="Moving objects, trees / water / sky, matte and Ignore Mask as PNG sequences")
    use_depth: BoolProperty(name="Depth (EXR)", default=True,
                            description="Z-depth per frame, 32-bit EXR in scene units (about 5 MB a frame at 1080p)")

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return _job_ok(context) and context.scene.camera is not None

    def invoke(self, context, event):
        if not self.directory:
            self.directory = os.path.join(context.scene.cameramatch.last_job, "export") + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        col = self.layout.column(heading="Formats")
        for p in ("use_abc", "use_fbx", "use_usd", "use_nk", "use_stmap", "use_json"):
            col.prop(self, p)
        col = self.layout.column(heading="Images")
        col.prop(self, "use_plate")
        col.prop(self, "use_masks")
        col.prop(self, "use_depth")
        col = self.layout.column(heading="Include")
        col.prop(self, "include_set")
        col.prop(self, "include_tracks")
        col.prop(self, "n_locators")
        col.prop(self, "include_cg")

    def execute(self, context):
        s = context.scene.cameramatch
        fm = [f for f, on in (("abc", self.use_abc), ("fbx", self.use_fbx), ("usd", self.use_usd),
                              ("nk", self.use_nk), ("stmap", self.use_stmap), ("json", self.use_json)) if on]
        out = bpy.path.abspath(self.directory or os.path.join(s.last_job, "export"))
        d = np.load(os.path.join(s.last_job, "result.npz"))
        clip = _main_clip(context)
        stats = guide.load_stats(s.last_job) or {}
        meta = dict(result={k: float(d[k]) for k in ("k1", "k2") if k in d},
                    track_obs=d["track_obs_full"] if "track_obs_full" in d else None,
                    plate=bpy.path.abspath(clip.filepath) if clip else "",
                    job_dir=s.last_job,
                    generator="CameraMatch Pi4 " + ".".join(str(v) for v in _version()),
                    solve=dict(median_px=stats.get("final", {}).get("median"), p90_px=stats.get("final", {}).get("p90"),
                               job=os.path.basename(os.path.normpath(s.last_job))))
        name = re.sub(r"[^\w.-]", "_", os.path.splitext(clip.name)[0] if clip else "cameramatch")
        try:
            w = export.export_scene(context, out, name, fm, self.include_set, self.include_tracks, self.n_locators,
                                    self.include_cg, log=_log, meta=meta)
        except Exception as e:
            _log(f"Export failed: {e}")
            self.report({"ERROR"}, f"Export failed: {e}")
            return {"CANCELLED"}
        _sync_log_text()
        self.out, self.name_ = out, name
        self.plate = meta["plate"]
        if not (self.use_plate or self.use_masks or self.use_depth):
            self.report({"INFO"}, f"Exported {len(w)} files to {out}")
            return {"FINISHED"}
        if _engine_down("Export"):
            self.report({"WARNING"}, f"Exported {len(w)} files to {out}; plate / masks / depth need the engine "
                                     "(Start Engine, then export again)")
            return {"FINISHED"}
        sc = context.scene
        clip_ = _main_clip(context)
        payload = dict(job=os.path.basename(os.path.normpath(s.last_job)), first=clip_.frame_start if clip_ else 1,
                       scale=float(sc.get("cm_applied_scale", 1.0)), plate=self.use_plate, masks=self.use_masks,
                       depth=self.use_depth)
        self.job_dir = s.last_job
        return _http_job(self, context, _prefs().matte_server_url.rstrip("/") + "/maps", payload, "Export",
                         "maps", self._maps_done)

    def _maps_done(self, context, r):
        src = os.path.join(self.job_dir, r.get("out", "export_maps"))
        same = os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(self.out))
        n = 0
        for sub in ("plate", "masks", "depth"):
            a, b = os.path.join(src, sub), os.path.join(self.out, sub)
            if not os.path.isdir(a) or same:
                continue
            if os.path.isdir(b):
                shutil.rmtree(b)
            shutil.move(a, b)                         # same drive: a rename; otherwise a copy
            n += len(os.listdir(b))
        seqs = export.sequences(self.out)
        export.finish_nuke(self.out, self.name_, context.scene, plate=self.plate, seqs=seqs)
        st = (r.get("summary") or {}).get("depth_alignment")
        dep = (f", depth x{st['factor']:.3f} to match the solve ({st['point_error_pct']:.1f}% per point)"
               if st else "")
        if st and st["point_error_pct"] > 15:
            dep += ". Warning: the model's depth disagrees with the solve on this shot: use it for rough work only"
        return f"Exported to {self.out}: " + ", ".join(sorted(seqs)) + dep

    def modal(self, context, event):
        return _http_job_modal(self, context, event)


def _version():
    try:
        import tomllib
        with open(os.path.join(os.path.dirname(__file__), "blender_manifest.toml"), "rb") as fh:
            return tuple(int(v) for v in tomllib.load(fh)["version"].split("."))
    except Exception:
        return (0, 0, 0)


# --------------------------------------------------------------------------- learning
class CAMERAMATCH_OT_open_docs(bpy.types.Operator):
    bl_idname = "cameramatch.open_docs"
    bl_label = "Quickstart"
    bl_description = "Open the quickstart guide (shooting tips and the six steps) in the web browser"

    def execute(self, context):
        bpy.ops.wm.url_open(url=DOCS_URL)
        return {"FINISHED"}


class CAMERAMATCH_OT_download_sample(bpy.types.Operator):
    bl_idname = "cameramatch.download_sample"
    bl_label = "Try the Sample Shot"
    bl_description = ("Download a sample street shot (Mixkit #4600, Mixkit free license, 1080p, 33 MB) and a sample "
                      "character (RobotExpressive, CC0) into the work folder, and load the shot")

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return not _STATUS["running"]

    def invoke(self, context, event):
        work = _work_dir()
        if not work or not os.path.isdir(work):
            self.report({"ERROR"}, "Set the Work folder in the add-on preferences first")
            return {"CANCELLED"}
        self.dest = os.path.join(work, "samples")
        os.makedirs(self.dest, exist_ok=True)
        self.files = [(SAMPLE_CLIP_URL, os.path.join(self.dest, "venice_alley.mp4")),
                      (SAMPLE_CHAR_URL, os.path.join(self.dest, "RobotExpressive.glb"))]
        self.error = None
        _STATUS.update(running=True, pct=0.0, stage="download", op="Sample", t0=time.time(), t_end=0.0,
                       text="downloading the sample shot", server="", job="", need_engine=False, cancel=False)
        _log("===== Download sample =====")

        def worker():
            try:
                for k, (url, path) in enumerate(self.files):
                    if os.path.isfile(path) and os.path.getsize(path) > 10000:
                        continue
                    req = urllib.request.Request(url, headers={"User-Agent": "CameraMatch-Blender-addon"})
                    with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as fh:
                        total = int(r.headers.get("Content-Length") or 0)
                        got = 0
                        while True:
                            if _STATUS["cancel"]:
                                raise RuntimeError("cancelled")
                            b = r.read(1 << 16)
                            if not b:
                                break
                            fh.write(b)
                            got += len(b)
                            if total:
                                _STATUS["pct"] = (k + got / total) / len(self.files)
                                _STATUS["text"] = f"{os.path.basename(path)}: {got / 1e6:.1f} / {total / 1e6:.1f} MB"
                    os.replace(path + ".part", path)
                    _log(f"downloaded {os.path.basename(path)}")
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        _redraw(context)
        if self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        if self.error:
            _end_progress(False, "Download failed: " + self.error)
            self.report({"ERROR"}, _STATUS["text"])
            return {"CANCELLED"}
        _load_footage(self, context, self.files[0][1])
        s = context.scene.cameramatch
        s.preset = "TRIPOD"                       # slow forward dolly with a rectilinear lens (see README)
        text = ("Sample loaded: a slow walk down a Venice alley. Preset set to Tripod / slow pan. Next: Solve "
                f"Camera. A character to add later is in {self.dest}")
        _end_progress(True, text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


# --------------------------------------------------------------------------- UI
def _steps(context):
    """Completion of the six steps: list of (key, title, done, summary)."""
    sc = context.scene
    s = sc.cameramatch
    clip = _main_clip(context)
    solved = _job_ok(context)
    st = guide.load_stats(s.last_job) if solved else None
    chk = guide.load_check(s.last_job) if solved else None
    lvl, head, _ = guide.grade(st, s, chk, tuple(clip.size) if clip else None) if solved else ("NONE", "", [])
    has_set = "cm_floor_coef" in sc
    movers = [c for c in bpy.data.collections if c.name.startswith(("CM_Robot", "CM_Character"))]
    comp = sc.get("cm_last_comp", "")
    return [
        ("footage", "Load Footage", clip is not None,
         f"{clip.name}  {clip.size[0]}x{clip.size[1]}, {clip.frame_duration} fr" if clip else ""),
        ("solve", "Solve Camera", solved, os.path.basename(os.path.normpath(s.last_job)) if solved else ""),
        ("check", "Check the Solve", solved and lvl != "NONE", head),
        ("set", "Build Set + Lighting", has_set, "floor, walls, plate lighting" if has_set else ""),
        ("cg", "Add Characters", bool(movers), f"{len(movers)} added" if movers else ""),
        ("render", "Render", bool(comp) and os.path.isfile(comp), os.path.basename(comp) if comp else ""),
    ], (lvl, head, st, chk)


def _draw_engine(layout, nc=50):
    icon, text = engine.summary()
    row = layout.row(align=True)
    row.label(text=text, icon=icon)
    if engine.STATE["action"]:
        return
    if engine.ready():
        row.operator("cameramatch.stop_engine", text="", icon="PAUSE")
    else:
        row.operator("cameramatch.start_engine", text="Start", icon="PLAY")
    if engine.STATE["error"] and not engine.ready():
        box = layout.box()
        for chunk in _wrap(engine.STATE["error"], nc)[:5]:
            box.label(text=chunk)


def _chars(context):
    """How many characters of label text fit the sidebar's current width."""
    try:
        w = context.region.width / context.preferences.system.ui_scale
    except Exception:
        w = 300
    return max(22, int((w - 40) / 6.2))


def _wrap(text, n):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > n and line:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    return out + ([line] if line else [])


def draw_guided(layout, context):
    s = context.scene.cameramatch
    nc = _chars(context)
    _draw_engine(layout, nc)
    steps, (lvl, head, st, chk) = _steps(context)
    current = next((k for k, _, done, _ in steps if not done), None)
    for n, (key, title, done, summ) in enumerate(steps, 1):
        box = layout.box()
        open_ = (key == current or (done and s.show_done)
                 or (key == "check" and done and (lvl in ("FAIR", "BAD") or current == "set")))
        row = box.row()
        row.label(text=f"{n}. {title}", icon="CHECKMARK" if done else ("RIGHTARROW" if key == current else "DOT"))
        if done and summ and not open_:
            box.label(text=summ[:nc])
        if not open_:
            continue
        col = box.column()
        if key == "footage":
            col.operator("cameramatch.load_footage", icon="FILE_MOVIE")
            if s.clip is None and not bpy.data.movieclips:
                col.operator("cameramatch.download_sample", icon="URL")
            if done:
                col.label(text=summ[:nc])
        elif key == "solve":
            col.prop(s, "preset", text="Footage")
            if s.ignore_mask is not None:
                col.label(text=f"Ignore Mask: {s.ignore_mask.name}", icon="MOD_MASK")
            col.operator("cameramatch.solve", icon="CAMERA_DATA")
            if done:
                col.operator("cameramatch.resolve", icon="FILE_REFRESH")
        elif key == "check":
            icon = {"GOOD": "CHECKMARK", "FAIR": "INFO", "BAD": "ERROR"}.get(lvl, "QUESTION")
            for k, chunk in enumerate(_wrap(head, nc - 3)):
                col.label(text=chunk, icon=icon if k == 0 else "BLANK1")
            for text, fix in guide.grade(st, s, chk, tuple(_main_clip(context).size)
                                         if _main_clip(context) else None)[2]:
                sub = col.box().column(align=True)
                for chunk in _wrap(text, nc - 2):
                    sub.label(text=chunk)
                if fix:
                    op = sub.operator("cameramatch.apply_fix",
                                      text="Draw Ignore Mask" if fix == "mask" else FIXES[fix][0].split(" and ")[0],
                                      icon="MOD_MASK" if fix == "mask" else "MODIFIER")
                    op.fix = fix
            row = col.row(align=True)
            row.operator("cameramatch.check_overlay", icon="MESH_GRID")
            row.operator("cameramatch.show_footage", text="", icon="FILE_MOVIE")
            if chk:
                for chunk in _wrap(f"Grid on the ground: {chk.get('stick_med_px', '?')} px median slide", nc):
                    col.label(text=chunk)
            col.operator("cameramatch.set_scale", text="Set Real Scale (optional)", icon="FIXED_SIZE")
        elif key == "set":
            col.operator("cameramatch.build_set", icon="MESH_GRID")
        elif key == "cg":
            col.prop(s, "char_height")
            col.operator("cameramatch.add_character", icon="ARMATURE_DATA")
            row = col.row(align=True)
            row.operator("cameramatch.new_path", icon="CURVE_BEZCURVE")
            row.operator("cameramatch.move_along_path", text="Walk Along Path", icon="CURVE_PATH")
            col.operator("cameramatch.walk_to_cursor", icon="PIVOT_CURSOR")
            col.separator()
            col.operator("cameramatch.add_robot", icon="OUTLINER_OB_ARMATURE")
            col.operator("cameramatch.place_on_floor", icon="SNAP_FACE")
            col.prop(s, "avoid_collisions")
        elif key == "render":
            row = col.row(align=True)
            row.operator("cameramatch.render_comp", text="Preview", icon="RESTRICT_RENDER_OFF").preview = True
            row.operator("cameramatch.render_comp", text="Final Render", icon="RENDER_ANIMATION").preview = False
            if done:
                col.label(text=summ[:nc])
    if _job_ok(context):
        box = layout.box()
        box.label(text="Export to Other Software", icon="EXPORT")
        box.operator("cameramatch.export_scene", text="Export Camera + Scene", icon="EXPORT")
    row = layout.row(align=True)
    row.prop(s, "show_done", text="Show All", toggle=True, icon="HIDE_OFF")
    row.operator("cameramatch.open_docs", icon="HELP")
    draw_progress(layout, nc)
    layout.label(text="Non-commercial use only", icon="ERROR")


def draw_progress(layout, nc=56):
    if not _STATUS["op"]:
        return
    box = layout.box()
    running = _STATUS["running"]
    pct = max(0.0, min(1.0, _STATUS["pct"]))
    stage = _STAGE_LABELS.get(_STATUS["stage"], _STATUS["stage"])
    head = box.row()
    head.label(text=f"{_STATUS['op']}: {stage}", icon="SORTTIME" if running else
               ("CHECKMARK" if _STATUS["stage"] == "done" else "ERROR"))
    if running:
        head.operator("cameramatch.cancel", text="", icon="CANCEL")
        box.progress(factor=pct, type="BAR", text=f"{pct * 100:.0f}%")
        el = time.time() - _STATUS["t0"]
        eta = f", about {_fmt_s(el * (1 - pct) / pct)} left" if pct > 0.03 else ""
        box.label(text=f"{_fmt_s(el)} elapsed{eta}")
    if _STATUS["text"]:
        for chunk in _wrap(_STATUS["text"], nc)[:6]:
            box.label(text=chunk)
    if _STATUS["need_engine"] and not engine.ready():
        box.operator("cameramatch.start_engine", icon="PLAY")
    row = box.row(align=True)
    row.operator("cameramatch.show_log", icon="TEXT")
    row.operator("cameramatch.clear_log", icon="TRASH", text="")


def _fmt_s(t):
    t = int(t)
    return f"{t // 60} min {t % 60:02d} s" if t >= 60 else f"{t} s"


def _make_panels(space, suffix):
    """The guided panel + collapsed Advanced sub-panels, for one editor type."""
    base = dict(bl_space_type=space, bl_region_type="UI", bl_category="Pi4")
    main_id = f"CAMERAMATCH_PT_guided_{suffix}"

    def draw_main(self, context):
        draw_guided(self.layout, context)

    def draw_solve(self, context):
        s = context.scene.cameramatch
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(s, "zoom_mode")
        lay.prop(s, "rs_mode")
        lay.prop(s, "solve_distortion")
        lay.prop(s, "lock_focal")
        lay.prop(s, "ignore_nonrigid")
        lay.prop(s, "ignore_mask")
        lay.separator()
        col = lay.column(align=True)
        col.prop(s, "stride")
        col.prop(s, "max_frames")
        col.prop(s, "max_size")
        lay.prop(s, "track")
        lay.prop(s, "refine")
        lay.prop(s, "max_tracks")
        col = lay.column(align=True)
        col.prop(s, "conf_threshold")
        col.prop(s, "num_points")

    def draw_cg(self, context):
        s = context.scene.cameramatch
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(s, "set_street_walls")
        if s.set_street_walls:
            lay.prop(s, "set_wall_height")
        lay.prop(s, "set_tris")
        lay.prop(s, "set_rebuild")
        lay.separator()
        lay.prop(s, "robot_speed")
        lay.prop(s, "robot_look")

    def draw_render(self, context):
        s = context.scene.cameramatch
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(s, "render_samples")
        lay.prop(s, "render_motion_blur")
        lay.prop(s, "comp_soft")
        lay.prop(s, "comp_wrap")

    def draw_tools(self, context):
        s = context.scene.cameramatch
        lay = self.layout
        lay.operator("cameramatch.refine_matte", icon="MATSHADERBALL")
        lay.prop(s, "auto_matte")
        lay.operator("cameramatch.export_tracks", icon="TRACKER")
        lay.operator("cameramatch.reimport", icon="IMPORT")
        lay.operator("cameramatch.check_server", icon="URL")
        lay.label(text=f"Scene scale: x{job_scale(s.last_job):.4f}" if s.last_job else "Scene scale: not set",
                  icon="DRIVER_DISTANCE")

    out = [type(main_id, (bpy.types.Panel,), dict(base, bl_idname=main_id, bl_label="CameraMatch",
                                                   draw=draw_main))]
    for key, label, fn in (("solve", "Advanced: Solve", draw_solve), ("cg", "Advanced: Set & Characters", draw_cg),
                           ("render", "Advanced: Render", draw_render), ("tools", "More Tools", draw_tools)):
        pid = f"CAMERAMATCH_PT_{key}_{suffix}"
        out.append(type(pid, (bpy.types.Panel,), dict(base, bl_idname=pid, bl_label=label, bl_parent_id=main_id,
                                                       bl_options={"DEFAULT_CLOSED"}, draw=fn)))
    return out


PANELS = _make_panels("CLIP_EDITOR", "clip") + _make_panels("VIEW_3D", "view3d")


classes = (CAMERAMATCH_Prefs, CAMERAMATCH_Settings, CAMERAMATCH_OT_check_server,
           CAMERAMATCH_OT_solve, CAMERAMATCH_OT_reimport, CAMERAMATCH_OT_refine_matte, CAMERAMATCH_OT_show_log,
           CAMERAMATCH_OT_clear_log, CAMERAMATCH_OT_build_set, CAMERAMATCH_OT_place_on_floor,
           CAMERAMATCH_OT_add_robot, CAMERAMATCH_OT_render_comp, CAMERAMATCH_OT_set_scale,
           CAMERAMATCH_OT_export_tracks, CAMERAMATCH_OT_resolve, CAMERAMATCH_OT_move_along_path,
           CAMERAMATCH_OT_start_engine, CAMERAMATCH_OT_stop_engine, CAMERAMATCH_OT_cancel,
           CAMERAMATCH_OT_load_footage, CAMERAMATCH_OT_show_footage, CAMERAMATCH_OT_check_overlay,
           CAMERAMATCH_OT_apply_fix, CAMERAMATCH_OT_add_character, CAMERAMATCH_OT_new_path,
           CAMERAMATCH_OT_walk_to_cursor, CAMERAMATCH_OT_open_docs, CAMERAMATCH_OT_download_sample,
           CAMERAMATCH_OT_export_scene, *PANELS)


def _statusbar_progress(self, context):
    """Compact progress bar in Blender's status bar, visible from every editor while a job runs."""
    if not _STATUS["running"]:
        return
    pct = max(0.0, min(1.0, _STATUS["pct"]))
    stage = _STAGE_LABELS.get(_STATUS["stage"], _STATUS["stage"])
    row = self.layout.row(align=True)
    row.label(text=f"CameraMatch {_STATUS['op']}: {stage}", icon="SORTTIME")
    sub = row.row()
    sub.ui_units_x = 8
    sub.progress(factor=pct, type="BAR", text=f"{pct * 100:.0f}%")


def _engine_tick():
    """Every second: point the health poller at the configured URLs, redraw the panels when the state changes."""
    try:
        p = _prefs()
        engine.POLLER.urls = (p.server_url, p.matte_server_url)
        engine.POLLER.start()
        if engine.POLLER.changed or engine.STATE["action"]:
            engine.POLLER.changed = False
            _tag_redraw(bpy.context.window_manager)
    except Exception:
        pass
    return 1.0


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.STATUSBAR_HT_header.append(_statusbar_progress)
    bpy.types.Scene.cameramatch = PointerProperty(type=CAMERAMATCH_Settings)
    if not bpy.app.background:
        bpy.app.timers.register(_engine_tick, first_interval=0.5, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_engine_tick):
        bpy.app.timers.unregister(_engine_tick)
    engine.POLLER.stop()
    _render_handlers(False)
    bpy.types.STATUSBAR_HT_header.remove(_statusbar_progress)
    del bpy.types.Scene.cameramatch
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
