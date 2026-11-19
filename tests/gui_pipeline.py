"""Drive the INSTALLED add-on's CG-integration buttons through the real UI (modal operators, render window).

blender <solved.blend> --python tests/gui_pipeline.py -- <out.blend> <cursor_x> <cursor_y> [f_from f_to samples]
Build Set + Lighting -> 3D cursor -> Add Walking Robot (full range) -> Render & Comp on [f_from, f_to] -> save, quit.
"""
import importlib
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out_blend = os.path.abspath(argv[0])
cx, cy = float(argv[1]), float(argv[2])
f_from, f_to, samples = (int(argv[3]), int(argv[4]), int(argv[5])) if len(argv) > 5 else (None, None, 64)
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
state = {"phase": "build", "t0": time.time()}


def log(*a):
    print("[gui_pipeline]", *a, flush=True)


def ctx():
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    if area.type != "CLIP_EDITOR":
        area.type = "CLIP_EDITOR"
        area.spaces.active.clip = cm._main_clip(bpy.context)
    region = next(r for r in area.regions if r.type == "WINDOW")
    return dict(window=win, area=area, region=region)


def wait_done(next_phase):
    if cm._STATUS["running"]:
        log(f"  {cm._STATUS['op']} | {cm._STATUS['stage']} {cm._STATUS['pct'] * 100:.0f}% | {cm._STATUS['text'][:80]}")
        return False
    log("STATUS:", cm._STATUS["text"])
    if cm._STATUS["stage"] == "error":
        raise RuntimeError(cm._STATUS["text"])
    state["phase"] = next_phase
    return True


def step():
    try:
        return _step()
    except Exception as e:
        import traceback
        traceback.print_exc()
        log("ERROR", e)
        bpy.ops.wm.quit_blender()
        return None


def _step():
    sc = bpy.context.scene
    ph = state["phase"]
    if ph == "build":
        with bpy.context.temp_override(**ctx()):
            log("Build Set ->", bpy.ops.cameramatch.build_set("INVOKE_DEFAULT"))
        state["phase"] = "build_wait"
        return 2.0
    if ph == "build_wait":
        wait_done("robot")
        return 3.0
    if ph == "robot":
        sc.cursor.location = (cx, cy, cm.set_builder.floor_z(sc, cx, cy))
        with bpy.context.temp_override(**ctx()):
            log("Add Walking Robot ->", bpy.ops.cameramatch.add_robot())
        log("   ", cm._STATUS["log"][-1])
        if f_from is not None:
            sc.frame_start, sc.frame_end = f_from, f_to
        sc.cameramatch.render_samples = samples
        state["phase"] = "render"
        return 1.0
    if ph == "render":
        with bpy.context.temp_override(**ctx()):
            log("Render & Comp ->", bpy.ops.cameramatch.render_comp("INVOKE_DEFAULT"))
        state["phase"] = "render_wait"
        return 3.0
    if ph == "render_wait":
        if wait_done("save"):
            return 0.5
        return 5.0
    if ph == "save":
        for line in cm._STATUS["log"][-25:]:
            log("LOG", line)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        log("SAVED", out_blend, f"total {time.time() - state['t0']:.0f}s")
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(step, first_interval=3.0)
