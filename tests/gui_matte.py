"""Drive the installed add-on's Refine Matte button in a Blender window on a solved .blend.

blender <solved.blend> --python tests/gui_matte.py -- <out.blend>
"""
import importlib
import os
import sys
import time

import bpy

out_blend = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
st = {"phase": "start", "t0": time.time()}


def log(*a):
    print("[gui_matte]", *a, flush=True)


def step():
    try:
        return _step()
    except Exception:
        import traceback
        traceback.print_exc()
        bpy.ops.wm.quit_blender()


def _step():
    win = bpy.context.window_manager.windows[0]
    if st["phase"] == "start":
        clip = next(c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte")))
        area = max(win.screen.areas, key=lambda a: a.width * a.height)
        area.type = "CLIP_EDITOR"
        area.spaces.active.clip = clip
        area.spaces.active.show_region_ui = True
        for rg in area.regions:
            if rg.type == "UI":
                try:
                    rg.active_panel_category = "Pi4"
                except Exception as e:
                    log("could not switch sidebar tab:", e)
        log("last_job", bpy.context.scene.cameramatch.last_job)
        st.update(phase="invoke", area=area, clip=clip.name)
        return 1.0
    if st["phase"] == "invoke":
        area = st["area"]
        region = next(r for r in area.regions if r.type == "WINDOW")
        with bpy.context.temp_override(window=win, area=area, region=region):
            log("poll", bpy.ops.cameramatch.refine_matte.poll())
            log("invoke ->", bpy.ops.cameramatch.refine_matte("INVOKE_DEFAULT"))
        st["phase"] = "wait"
        return 2.0
    if cm._STATUS["running"]:
        log("running:", f"{cm._STATUS['pct'] * 100:.0f}%", cm._STATUS["stage"], "|", cm._STATUS["text"],
            f"{time.time() - st['t0']:.0f}s")
        shot_dir = os.path.dirname(out_blend)
        for thr in (0.15, 0.5, 0.8):
            if cm._STATUS["pct"] >= thr and thr not in st.setdefault("shots", set()):
                st["shots"].add(thr)
                path = os.path.join(shot_dir, f"progress_ui_{int(thr * 100)}.png")
                with bpy.context.temp_override(window=win):
                    bpy.ops.screen.screenshot(filepath=path)
                log("screenshot", path)
        return 2.0
    log("STATUS:", cm._STATUS["text"])
    m = bpy.data.movieclips.get(st["clip"] + "_matte")
    log("matte clip:", m and (m.name, m.source, m.frame_start, m.frame_duration, tuple(m.size)))
    img = bpy.data.images.get(st["clip"] + "_pi4_invdepth")
    log("invdepth image:", img and (img.name, img.source, img.colorspace_settings.name))
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    log("SAVED", out_blend)
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(step, first_interval=2.0)
