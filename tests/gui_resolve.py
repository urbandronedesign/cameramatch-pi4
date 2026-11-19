"""Drive Re-solve (refine only) through the real UI on an already-solved .blend, with changed settings.

blender <solved.blend> --python tests/gui_resolve.py -- <out.blend>
Turns Rolling Shutter on and Active Tracks to 800, invokes cameramatch.resolve, waits, checks there is exactly
one imported solve, saves, quits.
"""
import importlib
import os
import sys
import time

import bpy

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
st = {"phase": "go", "t0": time.time()}


def log(*a):
    print("[gui_resolve]", *a, flush=True)


def step():
    try:
        win = bpy.context.window_manager.windows[0]
        area = max(win.screen.areas, key=lambda a: a.width * a.height)
        if st["phase"] == "go":
            area.type = "CLIP_EDITOR"
            area.spaces.active.clip = cm._main_clip(bpy.context)
            s = bpy.context.scene.cameramatch
            s.rolling_shutter, s.max_tracks = True, 800
            region = next(r for r in area.regions if r.type == "WINDOW")
            with bpy.context.temp_override(window=win, area=area, region=region):
                log("poll", bpy.ops.cameramatch.resolve.poll(), "->", bpy.ops.cameramatch.resolve("INVOKE_DEFAULT"))
            st["phase"] = "wait"
            return 2.0
        if cm._STATUS["running"]:
            log(f"  {cm._STATUS['stage']} {cm._STATUS['pct'] * 100:.0f}% {cm._STATUS['text'][:70]}")
            return 5.0
        log("STATUS:", cm._STATUS["text"])
        n = sum(c.name.startswith("Pi4_") for c in bpy.data.collections)
        log("imported solves:", n, "| scene readout", bpy.context.scene.get("cm_rolling_shutter"),
            f"| {time.time() - st['t0']:.0f}s")
        bpy.ops.wm.save_as_mainfile(filepath=out)
        log("SAVED", out)
        bpy.ops.wm.quit_blender()
    except Exception as e:
        import traceback
        traceback.print_exc()
        log("ERROR", e)
        bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(step, first_interval=3.0)
