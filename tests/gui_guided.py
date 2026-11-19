"""Drive the guided workflow (v0.12) through the real UI, from an empty scene, and screenshot every step.

blender --python tests/gui_guided.py -- <footage.mp4> <character.glb> <out_dir> [max_frames]
(uses the installed add-on, enabled in the user preferences)
Engine ready -> Load Footage -> Solve (Auto preset) -> Cancel test (Re-solve cancelled) -> Check Overlay ->
grade + suggestions -> a one-click fix (re-solve) -> Build Set -> Add Character + Walk to Cursor -> New Path +
robot -> Preview render -> save. Writes <out_dir>/shots/NN_step.png and <out_dir>/guided.blend, prints PASS/FAIL.
"""
import importlib
import json
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
footage, char_file, out_dir = (os.path.abspath(a) for a in argv[:3])
max_frames = int(argv[3]) if len(argv) > 3 else 120
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
os.makedirs(os.path.join(out_dir, "shots"), exist_ok=True)
state = {"phase": "layout", "t0": time.time(), "n": 0, "ok": True, "t_phase": time.time()}


def log(*a):
    print("[guided]", *a, flush=True)


def check(name, cond, info=""):
    state["ok"] &= bool(cond)
    log(f"{'PASS' if cond else 'FAIL'} {name} {info}")


def areas():
    win = bpy.context.window_manager.windows[0]
    return win, win.screen.areas


def ctx(kind="CLIP_EDITOR"):
    win, ars = areas()
    area = next(a for a in ars if a.type == kind)
    region = next(r for r in area.regions if r.type == "WINDOW")
    return dict(window=win, area=area, region=region, screen=win.screen)


def frame_views():
    """Clip fitted in the Clip Editor, 3D Viewport looking through the solved camera."""
    win, ars = areas()
    for a in ars:
        reg = next((r for r in a.regions if r.type == "WINDOW"), None)
        if reg is None:
            continue
        with bpy.context.temp_override(window=win, screen=win.screen, area=a, region=reg,
                                       space_data=a.spaces.active):
            try:
                if a.type == "CLIP_EDITOR" and a.spaces.active.clip is not None:
                    bpy.ops.clip.view_all(fit_view=True)
                elif a.type == "VIEW_3D" and bpy.context.scene.camera is not None:
                    if a.spaces.active.region_3d.view_perspective != "CAMERA":
                        bpy.ops.view3d.view_camera()
                    bpy.ops.view3d.view_center_camera()
            except Exception as e:
                log("frame_views:", e)


def shot(name):
    state["n"] += 1
    path = os.path.join(out_dir, "shots", f"{state['n']:02d}_{name}.png")
    frame_views()
    win, _ = areas()
    with bpy.context.temp_override(window=win, screen=win.screen):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        bpy.ops.screen.screenshot(filepath=path)
    log("screenshot", path)


def wait(next_phase, timeout=1800):
    if cm._STATUS["running"] or cm.engine.STATE["action"]:
        if time.time() - state["t_phase"] > timeout:
            raise RuntimeError("timeout")
        log(f"  {cm._STATUS['op']} | {cm._STATUS['stage']} {cm._STATUS['pct'] * 100:.0f}% | "
            f"{cm._STATUS['text'][:90]}")
        return False
    log("STATUS:", cm._STATUS["stage"], "|", cm._STATUS["text"])
    state["phase"], state["t_phase"] = next_phase, time.time()
    return True


def go(phase):
    state["phase"], state["t_phase"] = phase, time.time()


def step():
    try:
        return _step()
    except Exception as e:
        import traceback
        traceback.print_exc()
        log("ERROR", e)
        state["ok"] = False
        finish()
        return None


def finish():
    try:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "guided.blend"))
    except Exception as e:
        log("save failed", e)
    log("ALL PASS" if state["ok"] else "SOME FAILED", f"total {time.time() - state['t0']:.0f}s")
    bpy.ops.wm.quit_blender()


def _step():
    sc = bpy.context.scene
    s = sc.cameramatch
    ph = state["phase"]
    if ph == "layout":
        # left: Movie Clip Editor with the Pi4 sidebar; right: 3D Viewport
        win, ars = areas()
        big = max(ars, key=lambda a: a.width * a.height)
        with bpy.context.temp_override(window=win, screen=win.screen, area=big):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
        win, ars = areas()
        v3d = sorted([a for a in ars if a.type == "VIEW_3D"], key=lambda a: a.x)
        v3d[0].type = "CLIP_EDITOR"
        for a in ars:
            if a.type in ("CLIP_EDITOR", "VIEW_3D"):
                a.spaces.active.show_region_ui = True
        go("engine")
        return 2.0
    if ph == "engine":
        for a in areas()[1]:
            for r in a.regions:
                if r.type == "UI" and a.type in ("CLIP_EDITOR", "VIEW_3D"):
                    try:
                        r.active_panel_category = "Pi4"
                    except Exception as e:
                        log("cannot select the Pi4 tab:", e)
        if not cm.engine.ready():
            if time.time() - state["t_phase"] > 5 and not cm.engine.STATE["action"]:
                log("engine not ready:", cm.engine.summary())
                with bpy.context.temp_override(**ctx()):
                    bpy.ops.cameramatch.start_engine()
            if time.time() - state["t_phase"] > 900:
                raise RuntimeError("engine never became ready")
            return 2.0
        check("engine ready", True, cm.engine.summary()[1])
        shot("empty_scene")
        go("footage")
        return 1.0
    if ph == "footage":
        r = cm._load_footage(type("Op", (), {"report": lambda self, t, m: log("report", m)})(), bpy.context, footage)
        check("load footage", r == {"FINISHED"} and s.clip is not None, f"{s.clip.size[:]}")
        s.max_frames = max_frames
        s.preset = "AUTO"
        go("shot_footage")
        return 1.5
    if ph == "shot_footage":
        shot("footage_loaded")
        with bpy.context.temp_override(**ctx()):
            log("Solve ->", bpy.ops.cameramatch.solve("INVOKE_DEFAULT"))
        go("solve_wait")
        return 3.0
    if ph == "solve_wait":
        if state.get("solve_shot") is None and cm._STATUS["pct"] > 0.3:
            shot("solving")
            state["solve_shot"] = 1
        if wait("solved"):
            check("solve", cm._STATUS["stage"] == "done", cm._STATUS["text"][:120])
            cube = bpy.data.objects.get("Cube")
            if cube is not None:
                check("startup Cube (camera inside it) hidden from render", cube.hide_render)
        return 3.0
    if ph == "solved":
        st = cm.guide.load_stats(s.last_job)
        check("zoom auto-detected as fixed focal", st and st.get("zoom_auto", {}).get("zoom") is False,
              str(st.get("zoom_auto") if st else None))
        check("diagnostics present", st and "diag" in st, str((st or {}).get("diag", {}))[:200])
        shot("solved_grade")
        with bpy.context.temp_override(**ctx()):
            log("Re-solve (to cancel) ->", bpy.ops.cameramatch.resolve("INVOKE_DEFAULT"))
        go("cancel")
        return 6.0
    if ph == "cancel":
        with bpy.context.temp_override(**ctx()):
            log("Cancel ->", bpy.ops.cameramatch.cancel())
        go("cancel_wait")
        return 2.0
    if ph == "cancel_wait":
        if wait("check"):
            check("cancel stops the re-solve", cm._STATUS["text"] == "Cancelled.", cm._STATUS["text"])
            # the cancelled re-solve must leave the previous solve in place
            check("previous solve kept", cm._job_ok(bpy.context))
        return 2.0
    if ph == "check":
        with bpy.context.temp_override(**ctx()):
            log("Check Overlay ->", bpy.ops.cameramatch.check_overlay("INVOKE_DEFAULT"))
        go("check_wait")
        return 3.0
    if ph == "check_wait":
        if wait("checked"):
            m = cm.guide.load_check(s.last_job)
            check("overlay check", cm._STATUS["stage"] == "done" and m is not None
                  and bpy.data.movieclips.get(f"{s.clip.name}_check") is not None, cm._STATUS["text"][:160])
        return 2.0
    if ph == "checked":
        bpy.context.scene.frame_set(sc.frame_start + 40)
        st = cm.guide.load_stats(s.last_job)
        lvl, head, tips = cm.guide.grade(st, s, cm.guide.load_check(s.last_job), tuple(s.clip.size))
        log("GRADE", lvl, head)
        for t in tips:
            log("  TIP", t)
        check("grade computed", lvl in ("GOOD", "FAIR", "BAD"), head)
        shot("overlay_check")
        with bpy.context.temp_override(**ctx()):
            bpy.ops.cameramatch.show_footage()
            log("Fix (rolling shutter) ->", bpy.ops.cameramatch.apply_fix("INVOKE_DEFAULT", fix="rs"))
        go("fix_wait")
        return 3.0
    if ph == "fix_wait":
        if wait("set"):
            check("one-click fix re-solves", cm._STATUS["stage"] == "done" and s.rs_mode == "ON",
                  cm._STATUS["text"][:140])
        return 3.0
    if ph == "set":
        with bpy.context.temp_override(**ctx()):
            log("Build Set ->", bpy.ops.cameramatch.build_set("INVOKE_DEFAULT"))
        go("set_wait")
        return 3.0
    if ph == "set_wait":
        if wait("chars"):
            check("build set", "cm_floor_coef" in sc, cm._STATUS["text"][:120])
        return 3.0
    if ph == "chars":
        shot("set_built")
        cam = sc.camera
        sc.frame_set(sc.frame_start)
        c = cam.matrix_world.translation
        fwd = (cam.matrix_world.to_3x3() @ cm.Vector((0, 0, -1))).xy.normalized()
        a = c.xy + fwd * 6.0
        sc.cursor.location = (a.x, a.y, cm.set_builder.floor_z(sc, a.x, a.y))
        with bpy.context.temp_override(**ctx("VIEW_3D")):
            r = cm.add_character(type("Op", (), {"report": lambda self, t, m: log("report", m)})(), bpy.context,
                                 char_file)
        check("add character", r == {"FINISHED"})
        b = c.xy + fwd * 14.0
        sc.cursor.location = (b.x, b.y, cm.set_builder.floor_z(sc, b.x, b.y))
        with bpy.context.temp_override(**ctx("VIEW_3D")):
            r = bpy.ops.cameramatch.walk_to_cursor()
        check("walk to cursor", r == {"FINISHED"}, cm._STATUS["log"][-1][9:160])
        slide = [ln for ln in cm._STATUS["log"][-6:] if "slide" in ln]
        if slide:
            v = float(slide[-1].split("slide ")[1].split(" cm/s")[0])
            check("character feet planted", v < 10.0, f"{v} cm/s median")
        # a second mover: robot on a New Path coming toward the camera
        d = c.xy + fwd * 12.0 + cm.Vector((-fwd.y, fwd.x)) * 0.6
        sc.cursor.location = (d.x, d.y, cm.set_builder.floor_z(sc, d.x, d.y))
        with bpy.context.temp_override(**ctx("VIEW_3D")):
            bpy.ops.cameramatch.new_path()
            bpy.ops.object.mode_set(mode="OBJECT")
        crv = bpy.context.active_object
        # reverse it: toward the camera
        sp = crv.data.splines[0]
        cos = [p.co.copy() for p in sp.bezier_points]
        for p, q in zip(sp.bezier_points, reversed(cos)):
            p.co = q
        for p, k in zip(sp.bezier_points, range(len(cos))):
            t = k / (len(cos) - 1)
            p.co = (d.x - fwd.x * 6 * t, d.y - fwd.y * 6 * t, p.co.z)
        with bpy.context.temp_override(**ctx("VIEW_3D"), active_object=crv, selected_objects=[crv]):
            r = bpy.ops.cameramatch.add_robot()
        check("robot on a new path", r == {"FINISHED"}, cm._STATUS["log"][-1][9:160])
        wl = [ln for ln in cm._STATUS["log"][-8:] if "closest wall" in ln]
        if wl:
            dw = float(wl[-1].split("closest wall ")[1].split(" m")[0])
            check("robot path drawn into the wall ends up on the street", dw >= 0.3, f"closest wall {dw:.2f} m")
        go("shot_chars")
        return 1.0
    if ph == "shot_chars":
        sc.frame_set(sc.frame_start + 60)
        shot("characters")
        with bpy.context.temp_override(**ctx()):
            log("Preview ->", bpy.ops.cameramatch.render_comp("INVOKE_DEFAULT", preview=True))
        go("render_wait")
        return 5.0
    if ph == "render_wait":
        state.setdefault("t_render", time.time())
        if wait("done"):
            pv = bpy.data.movieclips.get(f"{s.clip.name}_preview")
            check("preview render + comp", cm._STATUS["stage"] == "done" and pv is not None
                  and pv.size[0] == s.clip.size[0] // 2, cm._STATUS["text"][:160])
            log(f"preview render + comp took {time.time() - state['t_render']:.0f}s")
            # pixels: the comp must look like the plate (not black / covered), with CG in it
            jd = s.last_job
            means = []
            for f in (sc.frame_start, sc.frame_start + 60, sc.frame_end - 1):
                im = bpy.data.images.load(os.path.join(jd, "comp_preview", f"comp_{f:04d}.png"))
                px = [0.0] * (im.size[0] * im.size[1] * 4)
                im.pixels.foreach_get(px)
                means.append(sum(px[0::4]) / (len(px) / 4))
                bpy.data.images.remove(im)
            check("preview comp is not black", min(means) > 0.12, f"red mean per frame {[round(m, 3) for m in means]}")
        return 5.0
    if ph == "done":
        sc.frame_set(sc.frame_start + 60)
        shot("preview_done")
        for line in cm._STATUS["log"][-40:]:
            log("LOG", line)
        with open(os.path.join(out_dir, "guided_status.json"), "w") as fh:
            json.dump(dict(ok=state["ok"], seconds=time.time() - state["t0"]), fh)
        finish()
        return None


bpy.app.timers.register(step, first_interval=3.0)
