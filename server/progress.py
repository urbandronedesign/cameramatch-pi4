"""Thread-safe per-job progress + log, served at GET /progress?job=<name>."""
import threading
import time


class Cancelled(Exception):
    """Raised inside a job (at its next progress update) after POST /cancel."""


class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}
        self._cancel = set()

    def cancel(self, job):
        """Ask a running job to stop: its next update() raises Cancelled."""
        with self._lock:
            j = self._jobs.get(job)
            if not j or j["state"] != "running":
                return False
            self._cancel.add(job)
            j["msg"] = "cancelling..."
            return True

    def cancelled(self, job):
        with self._lock:
            return job in self._cancel

    def start(self, job, stages):
        """stages: list of (name, weight). Resets the job."""
        with self._lock:
            self._jobs[job] = dict(stages=list(stages), done={}, stage=stages[0][0], frac=0.0, msg="",
                                   log=[], t0=time.time(), state="running", error=None)
            self._cancel.discard(job)

    def add_stages(self, job, stages):
        with self._lock:
            j = self._jobs.get(job)
            if j:
                j["stages"] += [s for s in stages if s[0] not in dict(j["stages"])]

    def update(self, job, stage, frac, msg=None):
        with self._lock:
            if job in self._cancel:
                raise Cancelled("cancelled by the user")
            j = self._jobs.get(job)
            if not j:
                return
            if stage != j["stage"]:
                j["done"][j["stage"]] = 1.0
            j["stage"], j["frac"] = stage, max(0.0, min(1.0, float(frac)))
            if msg:
                j["msg"] = msg

    def log(self, job, line):
        with self._lock:
            j = self._jobs.get(job)
            if j:
                j["log"].append(f"[{time.time() - j['t0']:6.1f}s] {line}")
                j["log"] = j["log"][-400:]

    def finish(self, job, ok=True, error=None):
        with self._lock:
            j = self._jobs.get(job)
            self._cancel.discard(job)
            if j:
                j["state"] = "done" if ok else "error"
                j["error"] = error
                if ok:
                    j["done"] = {n: 1.0 for n, _ in j["stages"]}
                    j["frac"] = 1.0

    def get(self, job, since=0):
        with self._lock:
            j = self._jobs.get(job)
            if not j:
                return {"state": "unknown"}
            tot = sum(w for _, w in j["stages"]) or 1.0
            acc = sum(w * (1.0 if n in j["done"] else (j["frac"] if n == j["stage"] else 0.0))
                      for n, w in j["stages"])
            return dict(state=j["state"], stage=j["stage"], stage_frac=j["frac"], msg=j["msg"],
                        pct=acc / tot, elapsed=time.time() - j["t0"], error=j["error"],
                        log=j["log"][since:], log_len=len(j["log"]))


PROGRESS = Progress()
