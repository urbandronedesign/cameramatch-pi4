# SPDX-License-Identifier: GPL-3.0-or-later
"""The 'engine': the two Docker services (Pi4 solver :8765, matte/set/comp server :8766).

Health is polled in a background thread (never blocks the UI). Start / Stop run `docker compose` in the
engine folder (the repository clone holding docker-compose.yml) without a console window, starting Docker
Desktop first if its daemon is not running.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request

STATE = {"pi4": "unknown", "matte": "unknown", "gpu": "", "docker": "unknown", "action": "", "error": "",
         "t": 0.0, "pi4_detail": ""}
_lock = threading.Lock()
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW
SERVICES = ("pi4", "matte-server")
IMAGES = ("cameramatch-pi4:latest", "cameramatch-matte:latest")
DOCKER_DESKTOP = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"


def _get(url, timeout=1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def probe(pi4_url, matte_url):
    """One health check of both servers (blocking, short timeouts). Updates STATE."""
    st = {}
    try:
        h = _get(pi4_url.rstrip("/") + "/health")
        st["pi4"] = h.get("status", "?")          # starting | loading | ready | error
        st["gpu"] = h.get("gpu") or ""
        st["pi4_detail"] = (h.get("error") or "")[-300:]
        if h.get("busy"):
            st["pi4"] = "busy"
    except Exception:
        st["pi4"] = "down"
    try:
        h = _get(matte_url.rstrip("/") + "/health")
        st["matte"] = "busy" if h.get("busy") else h.get("status", "?")
        st["gpu"] = st.get("gpu") or h.get("gpu") or ""
    except Exception:
        st["matte"] = "down"
    st["t"] = time.time()
    with _lock:
        STATE.update(st)
    return st


def ready():
    return STATE["pi4"] in ("ready", "busy") and STATE["matte"] in ("ready", "busy")


def summary():
    """(icon, text) for the status light."""
    a = STATE["action"]
    if a:
        return "SORTTIME", a
    p, m = STATE["pi4"], STATE["matte"]
    if p == "unknown":
        return "QUESTION", "Engine: checking..."
    if p in ("ready", "busy") and m in ("ready", "busy"):
        gpu = STATE["gpu"].replace("NVIDIA GeForce ", "")
        return "CHECKMARK", f"Engine ready ({gpu})" + (" - working" if "busy" in (p, m) else "")
    if p == "down" and m == "down":
        return "CANCEL", "Engine off"
    if p in ("starting", "loading"):
        return "SORTTIME", "Engine starting: loading the Pi4 model..."
    if p == "error":
        return "ERROR", "Engine error: the Pi4 model failed to load (see Show Log)"
    return "ERROR", f"Engine partly running (solver: {p}, matte: {m})"


def find_engine_dir(pref_dir, work_dir):
    """The folder holding docker-compose.yml: the preference, else the work folder's parent."""
    for d in (pref_dir, os.path.dirname(os.path.normpath(work_dir)) if work_dir else ""):
        if d and os.path.isfile(os.path.join(d, "docker-compose.yml")):
            return os.path.normpath(d)
    return ""


def _docker():
    exe = shutil.which("docker")
    if exe:
        return exe
    for p in (r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",):
        if os.path.isfile(p):
            return p
    return ""


def _run(args, cwd=None, timeout=60):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       creationflags=_NO_WINDOW)
    return r.returncode, (r.stdout + r.stderr).strip()


def _set(**kw):
    with _lock:
        STATE.update(kw)


def start(engine_dir, pi4_url, matte_url, log=print, wait_s=900):
    """Start Docker (if needed) and both services, then wait until they answer. Runs in a thread."""
    def job():
        try:
            docker = _docker()
            if not docker:
                raise RuntimeError("Docker is not installed. Run setup.ps1 from the CameraMatch folder "
                                   "(it checks WSL2, Docker and the GPU and installs the engine).")
            _set(action="Starting Docker...", error="")
            rc, out = _run([docker, "info", "--format", "{{.ServerVersion}}"], timeout=20)
            if rc != 0:
                if os.path.isfile(DOCKER_DESKTOP):
                    log("Docker is not running: starting Docker Desktop")
                    subprocess.Popen([DOCKER_DESKTOP], creationflags=_NO_WINDOW)
                t0 = time.time()
                while rc != 0 and time.time() - t0 < 180:
                    time.sleep(3)
                    rc, out = _run([docker, "info", "--format", "{{.ServerVersion}}"], timeout=20)
                if rc != 0:
                    raise RuntimeError("Docker did not start. Open Docker Desktop and try again.")
            missing = [im for im in IMAGES if _run([docker, "image", "inspect", im], timeout=30)[0] != 0]
            if missing:
                raise RuntimeError("The engine is not installed yet (missing " + ", ".join(missing) +
                                   "). Run setup.ps1 from the CameraMatch folder once.")
            _set(action="Starting the engine...")
            log(f"docker compose up -d ({engine_dir})")
            rc, out = _run([docker, "compose", "up", "-d", *SERVICES], cwd=engine_dir, timeout=300)
            if rc != 0:
                raise RuntimeError("docker compose failed: " + out[-400:])
            t0 = time.time()
            while time.time() - t0 < wait_s:
                st = probe(pi4_url, matte_url)
                if st["pi4"] in ("ready", "busy") and st["matte"] in ("ready", "busy"):
                    log(f"engine ready in {time.time() - t0:.0f}s ({st.get('gpu', '')})")
                    _set(action="")
                    return
                if st["pi4"] == "error":
                    raise RuntimeError("the Pi4 model failed to load: " + STATE["pi4_detail"][-200:])
                _set(action="Loading models (first start downloads several GB)..." if st["pi4"] == "loading"
                     else f"Waiting for the engine ({time.time() - t0:.0f}s)...")
                time.sleep(2)
            raise RuntimeError("the engine did not answer in time (see `docker compose logs pi4`)")
        except Exception as e:
            _set(action="", error=str(e))
            log("Start Engine failed: " + str(e))

    threading.Thread(target=job, daemon=True).start()


def stop(engine_dir, log=print):
    def job():
        try:
            docker = _docker()
            _set(action="Stopping the engine...")
            rc, out = _run([docker, "compose", "stop", *SERVICES], cwd=engine_dir, timeout=180)
            log("engine stopped" if rc == 0 else "docker compose stop failed: " + out[-300:])
        except Exception as e:
            log("Stop Engine failed: " + str(e))
        finally:
            _set(action="", pi4="down", matte="down")

    threading.Thread(target=job, daemon=True).start()


class Poller:
    """Background health polling every `period` s; `changed` is set when the summary text changes."""

    def __init__(self):
        self.urls = ("http://127.0.0.1:8765", "http://127.0.0.1:8766")
        self.changed = False
        self._stop = threading.Event()
        self._th = None

    def start(self, period=4.0):
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def loop():
            last = None
            while not self._stop.is_set():
                try:
                    probe(*self.urls)
                except Exception:
                    pass
                cur = (summary(), STATE["error"])
                if cur != last:
                    self.changed, last = True, cur
                self._stop.wait(period)

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()


POLLER = Poller()
