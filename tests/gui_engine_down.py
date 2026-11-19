"""What a user sees when Docker is closed (real UI). Quit Docker Desktop first (`docker desktop stop`).

blender --python tests/gui_engine_down.py -- <footage.mp4> <out_dir>
Solve with the engine down -> plain message + Start Engine button (screenshot) -> Start Engine (starts Docker
Desktop and the containers) -> engine light green (screenshot) -> a short Solve succeeds.
"""
import importlib
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
footage, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
os.makedirs(out, exist_ok=True)
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
st = {"ph": "wait_down", "t": time.time(), "ok": True}


def log(*a):
    print("[enginedown]", *a, flush=True)


def check(name, cond, info=""):
    st["ok"] &= bool(cond)
    log(f"{'PASS' if cond else 'FAIL'} {name} {info}")


def ctx():
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    if area.type != "CLIP_EDITOR":
        area.type = "CLIP_EDITOR"
        area.spaces.active.show_region_ui = True
    return dict(window=win, area=area, region=next(r for r in area.regions if r.type == "WINDOW"), screen=win.screen)


def shot(name):
    win = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=win, screen=win.screen):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        bpy.ops.screen.screenshot(filepath=os.path.join(out, name + ".png"))


def go(ph):
    st["ph"], st["t"] = ph, time.time()


def step():
    try:
        return _step()
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("no exception", False, str(e))
        bpy.ops.wm.quit_blender()


def _step():
    s = bpy.context.scene.cameramatch
    ph = st["ph"]
    if ph == "wait_down":
        ctx()
        for a in bpy.context.window_manager.windows[0].screen.areas:
            for r in a.regions:
                if r.type == "UI":
                    try:
                        r.active_panel_category = "Pi4"
                    except Exception:
                        pass
        if cm.engine.STATE["pi4"] == "unknown":
            return 1.0
        check("engine light says not running", cm.engine.summary()[1] == "Engine off", cm.engine.summary()[1])
        cm._load_footage(type("O", (), {"report": lambda self, t, m: None})(), bpy.context, footage)
        s.max_frames = 30
        with bpy.context.temp_override(**ctx()):
            try:
                r = bpy.ops.cameramatch.solve("INVOKE_DEFAULT")
            except RuntimeError as e:
                r = str(e)
        log("Solve with Docker closed ->", r)
        check("Solve refused with a plain message", cm._STATUS["need_engine"] and "Start Engine" in cm._STATUS["text"],
              cm._STATUS["text"])
        go("shot_down")
        return 1.5
    if ph == "shot_down":
        shot("1_engine_down")
        with bpy.context.temp_override(**ctx()):
            log("Start Engine ->", bpy.ops.cameramatch.start_engine())
        go("starting")
        return 1.0
    if ph == "starting":
        if st.get("shot_starting") is None and cm.engine.STATE["action"]:
            shot("2_starting")
            st["shot_starting"] = 1
        log("  ", cm.engine.summary()[1])
        if cm.engine.ready() and not cm.engine.STATE["action"]:
            check("engine started from Blender with Docker closed", True, f"{time.time() - st['t']:.0f}s")
            go("solve")
            return 1.0
        if cm.engine.STATE["error"] or time.time() - st["t"] > 600:
            check("engine started", False, cm.engine.STATE["error"])
            bpy.ops.wm.quit_blender()
            return None
        return 2.0
    if ph == "solve":
        shot("3_engine_ready")
        with bpy.context.temp_override(**ctx()):
            log("Solve ->", bpy.ops.cameramatch.solve("INVOKE_DEFAULT"))
        go("solve_wait")
        return 3.0
    if ph == "solve_wait":
        if cm._STATUS["running"]:
            return 3.0
        check("Solve works after Start Engine", cm._STATUS["stage"] == "done", cm._STATUS["text"][:120])
        shot("4_solved")
        log("ALL PASS" if st["ok"] else "SOME FAILED")
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(step, first_interval=3.0)
