"""Find slow, steady, trackable shots in a long video (cut detection + optical-flow motion stats).

python /tools/find_shots.py <video> <out_dir> [--min_sec 6] [--top 8]
Writes shots.json (all shots with stats) and cand_XX.jpg contact sheets for the top candidates.
"""
import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--min_sec", type=float, default=6.0)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_h, prev_g = None, None
    cuts, flow_med, flow_spread, texture, small = [0], [], [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        sm = cv2.resize(fr, (480, int(480 * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        h = cv2.normalize(h, None).flatten()
        if prev_h is not None:
            if cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA) > 0.35:
                cuts.append(i)
                flow_med.append(np.nan); flow_spread.append(np.nan)
            else:
                fl = dis.calc(prev_g, g, None)
                mag = np.hypot(fl[..., 0], fl[..., 1])
                flow_med.append(float(np.median(mag)))
                flow_spread.append(float(np.percentile(mag, 90) - np.percentile(mag, 10)))
        else:
            flow_med.append(np.nan); flow_spread.append(np.nan)
        texture.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        if i % 6 == 0:
            small.append((i, sm))
        prev_h, prev_g = h, g
        i += 1
    cuts.append(i)
    n = i
    flow_med, flow_spread, texture = np.array(flow_med), np.array(flow_spread), np.array(texture)
    shots = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s < 3:
            continue
        fm = flow_med[s + 1:e]
        shots.append(dict(start=int(s), end=int(e), sec=round((e - s) / fps, 2),
                          flow_median=float(np.nanmedian(fm)) if len(fm) else 0.0,
                          flow_p90=float(np.nanpercentile(fm, 90)) if len(fm) else 0.0,
                          parallax=float(np.nanmedian(flow_spread[s + 1:e])) if len(fm) else 0.0,
                          texture=float(np.median(texture[s:e]))))
    json.dump(dict(fps=fps, frames=n, shots=shots), open(os.path.join(a.out, "shots.json"), "w"), indent=1)
    # rank: long enough, slow-but-moving, steady (p90 not much above median), textured, with parallax
    cand = [s for s in shots if s["sec"] >= a.min_sec and 0.4 <= s["flow_median"] <= 3.0
            and s["flow_p90"] < 3 * max(s["flow_median"], 0.5)]
    cand.sort(key=lambda s: -(min(s["sec"], 15) * np.log1p(s["texture"]) * min(s["parallax"], 4)))
    print(f"{n} frames @ {fps:.2f} fps, {len(shots)} shots, {len(cand)} candidates")
    lut = dict(small)
    for k, s in enumerate(cand[:a.top]):
        picks = np.linspace(s["start"], s["end"] - 1, 5).astype(int)
        tiles = [lut[min(lut, key=lambda q: abs(q - p))] for p in picks]
        sheet = np.hstack(tiles)
        cv2.putText(sheet, f"#{k} frames {s['start']}-{s['end']} {s['sec']}s flow {s['flow_median']:.2f}px "
                    f"parallax {s['parallax']:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(a.out, f"cand_{k:02d}.jpg"), sheet)
        print(f"  #{k}: frames {s['start']}-{s['end']} ({s['sec']} s) flow {s['flow_median']:.2f} px/f, "
              f"p90 {s['flow_p90']:.2f}, parallax {s['parallax']:.2f}, texture {s['texture']:.0f}")


if __name__ == "__main__":
    main()
