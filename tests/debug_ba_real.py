"""Instrument BA on a cached real job (run in container): python debug_ba_real.py <job>"""
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/app")
import refine as rf  # noqa: E402

job = sys.argv[1]
captured = {}
orig_solve = rf.BA.solve


def spy(self, x, max_nfev, log, loss="huber", f_scale=1.5):
    if "ba" not in captured:
        captured.update(ba=self, x=x.copy(), loss=loss, f_scale=f_scale)
    return orig_solve(self, x, max_nfev, log, loss, f_scale)


rf.BA.solve = spy
result = dict(np.load(f"/work/{job}/result.npz"))
cache = np.load(f"/work/{job}/pi4_cache.npz")
rf.refine(f"/work/{job}/" + str(cache["src"]), result, cache, log=lambda s: None)

ba, x0 = captured["ba"], captured["x"]
# captured ba has stage-2 obs now; rebuild residual set is fine for diagnosis
r0 = ba.fun(x0)
J = ba.jac(x0)
g = J.T @ r0
print("nobs", ba.nobs, "cost0", 0.5 * r0 @ r0, "|g|", np.linalg.norm(g))
names = [("intr", slice(0, 3)), ("w", slice(ba.o_w, ba.o_c)), ("C", slice(ba.o_c, ba.o_x)), ("X", slice(ba.o_x, None))]
for n, s in names:
    print(f"  grad {n}: max {np.abs(g[s]).max():.3e}")
print("  grad intr", g[:3])
for xs in ["jac", 1.0]:
    for loss in ["linear", "huber"]:
        res = least_squares(ba.fun, x0, jac=ba.jac, method="trf", tr_solver="lsmr", loss=loss, f_scale=1.0,
                            x_scale=xs, max_nfev=50, verbose=0)
        e = ba.obs_errors(res.x)
        print(f"x_scale={xs} loss={loss}: nfev {res.nfev} status {res.status} cost {res.cost:.4e} "
              f"median {np.median(e):.3f} f {ba.f0*np.exp(res.x[0]):.2f} k {res.x[1]:.4f} {res.x[2]:.4f} "
              f"|dx| {np.linalg.norm(res.x - x0):.3e}")
