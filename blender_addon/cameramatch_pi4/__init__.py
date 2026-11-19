# SPDX-License-Identifier: GPL-3.0-or-later
"""CameraMatch Pi4: send a movie clip to a local Pi4 server, import the solved camera.

Workflow: Movie Clip Editor > Sidebar > Pi4 tab > Solve Camera.
The heavy lifting runs in a Docker container (see repository README); this add-on
only copies the clip to the shared work folder, calls the server over HTTP and
imports result.npz.
"""
import json
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
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from mathutils import Matrix

from . import robot, set_builder

# The work folder is the repo's ./work, mounted as /work by docker-compose. Set it in the add-on preferences,
# or with the CAMERAMATCH_WORK environment variable.
_WORK_ENV = "CAMERAMATCH_WORK"
# Shared between the UI (main thread) and worker/poller threads. Only plain values are written from threads.
_STATUS = {"text": "", "running": False, "pct": 0.0, "stage": "", "op": "", "t0": 0.0, "t_end": 0.0,
           "log": [], "synced": 0}
_LOG_TEXT = "CameraMatch_Log"
_STAGE_LABELS = {
    "copy": "Copying clip", "decode": "Decoding frames", "pi4": "Pi4 inference", "post": "Post-processing",
    "klt": "KLT tracking", "ba": "Bundle adjustment", "write": "Writing result", "import": "Importing into Blender",
    "load": "Loading frames", "sam_seeds": "SAM 2.1 keyframes", "sam_video": "SAM 2.1 propagation",
    "matanyone": "MatAnyone 2", "fuse": "Fusing mattes", "export": "Exporting mattes", "done": "Done",
    "error": "Failed", "mesh": "Set mesh (Poisson)", "envmap": "Plate environment", "set": "Building set in Blender",
    "render": "Rendering (Cycles)", "comp": "Compositing", "encode": "Encoding video",
}


def _log(line):
    stamp = time.strftime("%H:%M:%S")
    _STATUS["log"].append(f"{stamp} {line}")
    print(f"[CameraMatch] {line}")


def _start_progress(op, server_url, job):
    """Begin a tracked operation; a poller thread mirrors the server's /progress into _STATUS."""
    _STATUS.update(running=True, pct=0.0, stage="copy" if op == "Solve" else "load", op=op,
                   t0=time.time(), t_end=0.0, text=f"{op} starting...")
    _log(f"===== {op} | job {job} =====")
    url = f"{server_url.rstrip('/')}/progress?job={urllib.parse.quote(job)}"

    def poll():
        since, live = 0, False
        while _STATUS["running"]:
            try:
                p = _http(f"{url}&since={since}", timeout=2)
                # the server may still hold this job's previous run (finished): wait until the new one starts
                live = live or p.get("state") == "running"
                if live:
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
    _STATUS.update(running=False, pct=1.0 if ok else _STATUS["pct"], stage="done" if ok else "error",
                   text=text, t_end=time.time())
    _log(("DONE: " if ok else "FAILED: ") + text)
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
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type in {"CLIP_EDITOR", "TEXT_EDITOR", "STATUSBAR", "VIEW_3D"}:
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

    def draw(self, context):
        self.layout.prop(self, "server_url")
        self.layout.prop(self, "matte_server_url")
        self.layout.prop(self, "work_dir")
        if not _work_dir():
            self.layout.label(text="Set the Work folder to <your clone>/work (docker-compose mounts it as /work)",
                              icon="ERROR")


class CAMERAMATCH_Settings(bpy.types.PropertyGroup):
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
    robot_look: BoolProperty(name="Look at Camera", default=True,
                             description="The robot turns its head to the camera in the last third of the shot")
    render_samples: IntProperty(name="Samples", default=192, min=1, max=4096)
    render_motion_blur: BoolProperty(name="Motion Blur", default=True)
    comp_soft: FloatProperty(name="CG Softness (px)", default=0.6, min=0.0, max=3.0,
                             description="Gaussian blur on the CG to match the plate's sharpness")
    comp_wrap: FloatProperty(name="Light Wrap", default=0.25, min=0.0, max=1.0)


def _prefs():
    return bpy.context.preferences.addons[__package__].preferences


def _work_dir():
    """Work folder from the preferences, else $CAMERAMATCH_WORK, else '' (not configured)."""
    w = _prefs().work_dir.strip() or os.environ.get(_WORK_ENV, "").strip()
    return bpy.path.abspath(w) if w else ""


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
    if refined:
        # Same model as libmv's Polynomial: x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6), centred principal point.
        tcam.distortion_model = "POLYNOMIAL"
        tcam.k1, tcam.k2, tcam.k3 = float(d["k1"]), float(d["k2"]), 0.0
        tcam.principal_point = (0.0, 0.0)
    return cam, lens_med, float(np.std(lens))


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
        return context.edit_movieclip is not None and not _STATUS["running"]

    def invoke(self, context, event):
        clip = context.edit_movieclip
        prefs = _prefs()
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
                       conf_threshold=s.conf_threshold, num_points=s.num_points,
                       refine=s.refine, max_tracks=s.max_tracks, solve_distortion=s.solve_distortion,
                       focal_sigma=0.002 if s.lock_focal else 0.3, zoom=s.animate_focal, solver="lm")
        url = prefs.server_url.rstrip("/") + "/solve"

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
            self.report({"ERROR"}, "Pi4 solve failed: " + r.get("error", "?"))
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
        _end_progress(True, text)
        self.report({"INFO"}, "Pi4 solve imported: " + text)
        if scene.cameramatch.auto_matte:
            bpy.ops.cameramatch.refine_matte("INVOKE_DEFAULT")
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
        return (context.edit_movieclip is not None and not _STATUS["running"] and j
                and os.path.isfile(os.path.join(j, "pi4_cache.npz")))

    def invoke(self, context, event):
        self.job_dir = context.scene.cameramatch.last_job
        self.clip_name = context.edit_movieclip.name
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
        return context.edit_movieclip is not None and j and os.path.isfile(os.path.join(j, "result.npz"))

    def execute(self, context):
        s = context.scene.cameramatch
        remove_imported(context.scene, context.edit_movieclip)
        import_result(context.scene, context.edit_movieclip, s.last_job, s)
        return {"FINISHED"}


# --------------------------------------------------------------------------- CG integration
def _job_ok(context, fname="result.npz"):
    j = context.scene.cameramatch.last_job
    return bool(j) and os.path.isfile(os.path.join(j, fname))


def _main_clip(context):
    clip = getattr(context, "edit_movieclip", None)
    if clip is None:
        cands = [c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte", "_comp"))]
        clip = cands[0] if cands else None
    return clip


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
        text = (f"Set built: floor fit {context.scene['cm_floor_sigma'] * 100:.1f} cm, collection "
                f"'{set_builder.SET_COLL}'. Place CG with Place on Floor / Add Walking Robot")
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
    bl_description = ("Procedural robot walking from the 3D cursor toward the camera, feet planted on the fitted "
                      "floor, over the scene frame range")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return "cm_floor_coef" in context.scene and context.scene.camera is not None and not _STATUS["running"]

    def execute(self, context):
        sc = context.scene
        s = sc.cameramatch
        _log(f"===== Add Walking Robot from cursor {tuple(round(x, 2) for x in sc.cursor.location)} =====")
        try:
            robot.add_walking_robot(sc, sc.cursor.location.copy(), lambda x, y: set_builder.floor_z(sc, x, y),
                                    _log, speed=s.robot_speed, look_at_camera=s.robot_look)
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
        self.job_dir = s.last_job
        self.job = os.path.basename(os.path.normpath(self.job_dir))
        clip = _main_clip(context)
        self.clip_name = clip.name if clip else "plate"
        render_dir = os.path.join(self.job_dir, "render")
        set_builder.setup_render(sc, render_dir, s.render_samples, s.render_motion_blur)
        self.frames = list(range(sc.frame_start, sc.frame_end + 1))
        self.fps = sc.render.fps / sc.render.fps_base
        self.phase = "render"
        self.result = None
        _STATUS.update(running=True, pct=0.0, stage="render", op="Render & Comp", t0=time.time(), t_end=0.0,
                       text="warm-up render")
        _log(f"===== Render & Comp | job {self.job} | frames {self.frames[0]}-{self.frames[-1]}, "
             f"{s.render_samples} samples =====")
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
        payload = dict(job=self.job, render_dir="render", out_dir="comp", frames=self.frames, fps=self.fps,
                       soft=s.comp_soft, wrap=s.comp_wrap)

        def worker():
            try:
                self.result = _http(url, payload, timeout=None)
            except Exception as e:
                self.result = {"ok": False, "error": f"{type(e).__name__}: {e} (is matte-server running?)"}

        t0 = _STATUS["t0"]
        _start_progress("Render & Comp", _prefs().matte_server_url, self.job)
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
            if _RENDER["cancel"]:
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
        name = f"{self.clip_name}_comp"
        old = bpy.data.movieclips.get(name)
        if old:
            bpy.data.movieclips.remove(old)
        mc = bpy.data.movieclips.load(video)
        mc.name = name
        mc.frame_start = self.frames[0]
        text = f"{r['frames']} frames composited -> {video} (clip '{name}'; side by side: {r['side_by_side']})"
        _end_progress(True, text)
        self.report({"INFO"}, text)
        return {"FINISHED"}


# --------------------------------------------------------------------------- UI
def draw_cg_box(layout, s):
    box = layout.box()
    box.label(text="CG Integration", icon="SCENE_DATA")
    col = box.column(align=True)
    col.prop(s, "set_street_walls")
    if s.set_street_walls:
        col.prop(s, "set_wall_height")
    col.prop(s, "set_rebuild")
    box.operator("cameramatch.build_set", icon="MESH_GRID")
    box.operator("cameramatch.place_on_floor", icon="SNAP_FACE")
    box.operator("cameramatch.add_robot", icon="ARMATURE_DATA")
    col = box.column(align=True)
    col.prop(s, "robot_speed")
    col.prop(s, "robot_look")
    col = box.column(align=True)
    col.prop(s, "render_samples")
    col.prop(s, "render_motion_blur")
    col.prop(s, "comp_soft")
    col.prop(s, "comp_wrap")
    box.operator("cameramatch.render_comp", icon="RENDER_ANIMATION")


class CAMERAMATCH_PT_view3d(bpy.types.Panel):
    """The CG-integration tools where CG is placed: 3D Viewport > Sidebar > Pi4."""
    bl_label = "Pi4 CG Integration"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pi4"

    def draw(self, context):
        draw_cg_box(self.layout, context.scene.cameramatch)
        CAMERAMATCH_PT_panel.draw_progress(self.layout)


class CAMERAMATCH_PT_panel(bpy.types.Panel):
    bl_label = "Pi4 Camera Match"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Pi4"

    def draw(self, context):
        layout = self.layout
        s = context.scene.cameramatch
        col = layout.column(align=True)
        col.prop(s, "stride")
        col.prop(s, "max_frames")
        col.prop(s, "max_size")
        layout.prop(s, "track")
        layout.prop(s, "refine")
        if s.refine:
            layout.prop(s, "max_tracks")
            layout.prop(s, "solve_distortion")
            layout.prop(s, "lock_focal")
        layout.prop(s, "animate_focal")
        col = layout.column(align=True)
        col.prop(s, "conf_threshold")
        col.prop(s, "num_points")
        row = layout.row(align=True)
        row.operator("cameramatch.check_server", icon="URL")
        layout.operator("cameramatch.solve", icon="CAMERA_DATA")
        layout.operator("cameramatch.reimport", icon="FILE_REFRESH")
        box = layout.box()
        box.label(text=f"Scene Scale: x{job_scale(s.last_job):.4f}" if s.last_job else "Scene Scale",
                  icon="DRIVER_DISTANCE")
        box.operator("cameramatch.set_scale", icon="FIXED_SIZE")
        box.operator("cameramatch.export_tracks", icon="TRACKER")
        box = layout.box()
        box.label(text="Foreground Matte", icon="MOD_MASK")
        box.operator("cameramatch.refine_matte", icon="MATSHADERBALL")
        box.prop(s, "auto_matte")
        draw_cg_box(layout, s)
        self.draw_progress(layout)
        layout.label(text="Pi4 / MatAnyone 2: non-commercial licenses", icon="ERROR")

    @staticmethod
    def draw_progress(layout):
        if not _STATUS["op"]:
            return
        box = layout.box()
        running = _STATUS["running"]
        pct = max(0.0, min(1.0, _STATUS["pct"]))
        stage = _STAGE_LABELS.get(_STATUS["stage"], _STATUS["stage"])
        head = box.row()
        head.label(text=f"{_STATUS['op']}: {stage}", icon="SORTTIME" if running else
                   ("CHECKMARK" if _STATUS["stage"] == "done" else "ERROR"))
        box.progress(factor=pct, type="BAR", text=f"{pct * 100:.0f}%")
        el = (time.time() if running else _STATUS["t_end"]) - _STATUS["t0"]
        eta = f" | ETA {el * (1 - pct) / pct:.0f}s" if running and pct > 0.03 else ""
        box.label(text=f"elapsed {el:.0f}s{eta}")
        if _STATUS["text"]:
            for chunk in [_STATUS["text"][i:i + 60] for i in range(0, min(len(_STATUS["text"]), 180), 60)]:
                box.label(text=chunk)
        logcol = box.column(align=True)
        logcol.scale_y = 0.75
        for line in _STATUS["log"][-7:]:
            logcol.label(text=line[:80])
        row = box.row(align=True)
        row.operator("cameramatch.show_log", icon="TEXT")
        row.operator("cameramatch.clear_log", icon="TRASH", text="")


classes = (CAMERAMATCH_Prefs, CAMERAMATCH_Settings, CAMERAMATCH_OT_check_server,
           CAMERAMATCH_OT_solve, CAMERAMATCH_OT_reimport, CAMERAMATCH_OT_refine_matte, CAMERAMATCH_OT_show_log,
           CAMERAMATCH_OT_clear_log, CAMERAMATCH_OT_build_set, CAMERAMATCH_OT_place_on_floor,
           CAMERAMATCH_OT_add_robot, CAMERAMATCH_OT_render_comp, CAMERAMATCH_OT_set_scale,
           CAMERAMATCH_OT_export_tracks, CAMERAMATCH_PT_panel, CAMERAMATCH_PT_view3d)


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


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.STATUSBAR_HT_header.append(_statusbar_progress)
    bpy.types.Scene.cameramatch = PointerProperty(type=CAMERAMATCH_Settings)


def unregister():
    _render_handlers(False)
    bpy.types.STATUSBAR_HT_header.remove(_statusbar_progress)
    del bpy.types.Scene.cameramatch
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
