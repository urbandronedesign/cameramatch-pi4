"""Drive the INSTALLED add-on through the real UI operator (modal, in a Blender window).

blender --python tests/gui_solve.py -- <video> <out.blend> [sensor_mm]
Uses user preferences (so the installed extension is active), opens a Movie Clip Editor,
invokes cameramatch.solve exactly like the button, waits, saves, quits.
"""
import importlib
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
video, out_blend = os.path.abspath(argv[0]), os.path.abspath(argv[1])
sensor = float(argv[2]) if len(argv) > 2 else 36.0
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
state = {"phase": "start", "t0": time.time()}


def log(*a):
    print("[gui_solve]", *a, flush=True)


def step():
    try:
        return _step()
    except Exception as e:  # never leave a hung Blender
        import traceback
        traceback.print_exc()
        log("ERROR", e)
        bpy.ops.wm.quit_blender()
        return None


def _step():
    win = bpy.context.window_manager.windows[0]
    if state["phase"] == "start":
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        clip = bpy.data.movieclips.load(video)
        clip.tracking.camera.sensor_width = sensor
        area = max(win.screen.areas, key=lambda a: a.width * a.height)
        area.type = "CLIP_EDITOR"
        area.spaces.active.clip = clip
        s = bpy.context.scene.cameramatch
        s.refine, s.track, s.max_frames, s.stride = True, True, int(os.environ.get("CM_MAX_FRAMES", 300)), 1
        if "CM_SOLVE_DISTORTION" in os.environ:
            s.solve_distortion = os.environ["CM_SOLVE_DISTORTION"] == "1"
        if "CM_LOCK_FOCAL" in os.environ:
            s.lock_focal = os.environ["CM_LOCK_FOCAL"] == "1"
        state.update(phase="invoke", area=area, clip=clip.name)
        return 1.0
    if state["phase"] == "invoke":
        area = state["area"]
        region = next(r for r in area.regions if r.type == "WINDOW")
        with bpy.context.temp_override(window=win, area=area, region=region):
            log("poll", bpy.ops.cameramatch.solve.poll())
            res = bpy.ops.cameramatch.solve("INVOKE_DEFAULT")
        log("invoke ->", res)
        state["phase"] = "wait"
        return 2.0
    if state["phase"] == "wait":
        if cm._STATUS["running"]:
            log("running:", cm._STATUS["text"], f"{time.time() - state['t0']:.0f}s")
            return 5.0
        log("STATUS:", cm._STATUS["text"])
        cams = [o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("Pi4_")]
        log("imported cameras:", [c.name for c in cams])
        tc = bpy.data.movieclips[state["clip"]].tracking.camera
        log("clip camera:", tc.focal_length, tc.distortion_model, tc.k1, tc.k2)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        log("SAVED", out_blend)
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(step, first_interval=2.0)
