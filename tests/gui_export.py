"""Export Camera + Scene through the real UI (modal: plate / masks / depth come from the matte server).

blender <solved scene>.blend --python tests/gui_export.py -- <out_dir>
Checks: every file present, sequences complete and numbered like the scene, Nuke script reads them, README.
"""
import importlib
import os
import sys
import time

import bpy

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1:][0])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
st = {"ph": "go", "t": time.time(), "ok": True}


def check(name, cond, info=""):
    st["ok"] &= bool(cond)
    print(f"[guiexport] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)


def step():
    try:
        return _step()
    except Exception:
        import traceback
        traceback.print_exc()
        st["ok"] = False
        bpy.ops.wm.quit_blender()


def _step():
    sc = bpy.context.scene
    if st["ph"] == "go":
        if cm.engine.STATE["pi4"] == "unknown":
            return 1.0
        win = bpy.context.window_manager.windows[0]
        area = max(win.screen.areas, key=lambda a: a.width * a.height)
        with bpy.context.temp_override(window=win, area=area, screen=win.screen):
            r = bpy.ops.cameramatch.export_scene("EXEC_DEFAULT", directory=out + os.sep, include_cg=True)
        print("[guiexport] export ->", r, flush=True)
        st["ph"], st["t"] = "wait", time.time()
        return 2.0
    if cm._STATUS["running"]:
        print(f"[guiexport]   {cm._STATUS['stage']} {cm._STATUS['pct'] * 100:.0f}% {cm._STATUS['text'][:80]}", flush=True)
        return 3.0
    print("[guiexport] STATUS", cm._STATUS["stage"], cm._STATUS["text"], flush=True)
    check("export finished", cm._STATUS["stage"] == "done", f"{time.time() - st['t']:.0f}s")
    n = sc.frame_end - sc.frame_start + 1
    files = set(os.listdir(out))
    for f in ("camera.json", "camera.csv", "README.txt", "stmap_undistort.exr", "stmap_redistort.exr"):
        check(f"{f} present", f in files)
    for ext in (".abc", ".fbx", ".usdc", ".nk"):
        check(f"{ext} present", any(f.endswith(ext) for f in files))
    seq = {d: sorted(os.listdir(os.path.join(out, d))) if os.path.isdir(os.path.join(out, d)) else []
           for d in ("plate", "masks", "depth")}
    check("plate: every frame", len(seq["plate"]) == n and seq["plate"][0] == f"plate_{sc.frame_start:04d}.png",
          f"{len(seq['plate'])} files, {seq['plate'][:1]}")
    mv = [f for f in seq["masks"] if f.startswith("moving_")]
    check("masks: moving objects", len(mv) == n, f"{len(mv)} files; kinds {sorted({f.rsplit('_', 1)[0] for f in seq['masks']})}")
    check("depth: every frame", len(seq["depth"]) == n, f"{len(seq['depth'])} files")
    nk = open(os.path.join(out, next(f for f in files if f.endswith(".nk")))).read()
    for node in ("Plate_sequence", "Mask_moving", "Depth", "CameraMatch_cam", "Undistort_plate"):
        check(f"Nuke script has {node}", f"name {node}" in nk)
    print("[guiexport]", "ALL PASS" if st["ok"] else "SOME FAILED", flush=True)
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(step, first_interval=3.0)
