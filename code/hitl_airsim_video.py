#!/usr/bin/env python3
"""Video of the ThinkFly AirSim PX4-HITL bench runs (Pixhawk 6C in HITL, AirSim computes physics and sensors).
Chase-camera frames (frames.csv: airsim_t, file, ..., wall time) and true poses (pose.csv: wall, x, y, z NED) are
matched to the mission logs: flight segments in pose.csv are taken in mission order, and the time offset inside each
segment is found by matching the logged track (ENU) to the true path.
  hitl_airsim_video.py RUN_DIR MISSION_DIR OUT.mp4 --first-seg 9 [--speed 6] [--fps 15]"""
import argparse
import bisect
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_site import rescore  # noqa: E402

W, H = 1280, 720
NAME = {'thinkfly': 'ThinkFly: event-triggered, small onboard model + rule checker, large model only on alarm',
        'event': 'Baseline: event-triggered, large model only',
        'step': 'Baseline: step-by-step, large model at every step'}
ORDER = ['thinkfly', 'event', 'step']
SHORT = {'thinkfly': 'ThinkFly', 'event': 'Event-triggered (baseline)', 'step': 'Step-by-step (baseline)'}
BG = (32, 23, 15)


def text(im, s, xy, sc=0.8, col=(255, 255, 255), th=2):
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def card(title, lines, sc=0.72, dy=44):
    im = np.full((H, W, 3), BG, np.uint8)
    text(im, title, (60, 130), 1.3, (255, 255, 255), 3)
    for k, l in enumerate(lines):
        text(im, l, (60, 200 + dy * k), sc, (232, 222, 215), 2)
    return im


def segments(P):
    segs, on = [], False
    for i, (w, x, y, z) in enumerate(P):
        if z < -3 and not on:
            on, s = True, i
        if z > -1 and on:
            on = False
            segs.append((s, i))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('missions')
    ap.add_argument('out')
    ap.add_argument('--first-seg', type=int, required=True)
    ap.add_argument('--speed', type=float, default=6.0)
    ap.add_argument('--fps', type=int, default=15)
    a = ap.parse_args()
    F = [r for r in csv.reader(open(os.path.join(a.run, 'frames.csv')))]
    fw = [float(r[5]) for r in F]
    P = np.array([[float(v) for v in r] for r in csv.reader(open(os.path.join(a.run, 'pose.csv')))])
    segs = segments(P)
    tmp = tempfile.mkdtemp()
    k = 0

    def put(im, n=1):
        nonlocal k
        for _ in range(n):
            cv2.imwrite(os.path.join(tmp, f'f{k:06d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            k += 1
    put(card('ThinkFly hardware in the loop (PX4 HITL + AirSim)', [
        'Pixhawk 6C runs real PX4 1.15.1 in HITL mode: AirSim computes the physics and sensors',
        'Jetson Orin Nano (25 W) runs ThinkFly + the small model (Qwen3.5-4B); energy = VDD_IN rail',
        'Large model (Qwen3.5-27B) on a ground-station GPU, energy metered per call (NVML)',
        'AirSim 1.8.1 AirSimNH neighbourhood; the same paired scenarios as the simulation study',
        'Bench test, propellers off; 4 November 2026']), a.fps * 7)
    summary = []
    seg_i = a.first_seg
    for s in range(3):
        for ag in ORDER:
            r = json.load(open(os.path.join(a.missions, f'mission_{s}_{ag}.json')))
            v, n, safe = rescore(r)
            i0, i1 = segs[seg_i]
            seg_i += 1
            tr = np.array(r['track'])
            # true path in ENU: east = y_ned, north = x_ned
            pw, pe, pn = P[i0:i1, 0], P[i0:i1, 2], P[i0:i1, 1]
            best = None
            for off in np.arange(pw[0] - 30, pw[0] + 60, 0.25):        # wall time of mission t = 0
                tw = off + tr[:, 0]
                m = (tw >= pw[0]) & (tw <= pw[-1])
                if m.sum() < 20:
                    continue
                e = np.interp(tw[m], pw, pe) - tr[m, 1]
                nn = np.interp(tw[m], pw, pn) - tr[m, 2]
                err = float(np.mean(np.hypot(e, nn)))
                if best is None or err < best[0]:
                    best = (err, off)
            err, off = best
            print(f'scenario {s} {ag}: segment {seg_i - 1}, path match error {err:.1f} m', flush=True)
            evs = []
            for res in r['calls']:
                s0 = res['t']
                for c in res['calls']:
                    evs.append((s0, s0 + c['latency_s'], c))
                    s0 += c['latency_s']
            put(card(f'Scenario {s + 1} of 3: {SHORT[ag]}', [NAME[ag], ''] + [w for nt in r['notes'][:3] for w in textwrap.wrap(nt, 88)]), a.fps * 4)
            T = tr[-1, 0]
            for tm in np.arange(-8, T + 4, a.speed / a.fps):
                j = min(bisect.bisect_left(fw, off + tm), len(F) - 1)
                im = cv2.imread(os.path.join(a.run, 'frames', F[j][1]))
                im = cv2.resize(im, (W, H)) if im is not None else np.zeros((H, W, 3), np.uint8)
                row = tr[min(max(int(np.searchsorted(tr[:, 0], tm)), 0), len(tr) - 1)]
                done = [c for (s0, s1, c) in evs if s1 <= tm]
                now = [c for (s0, s1, c) in evs if s0 <= tm < s1]
                cv2.rectangle(im, (0, 0), (W, 80), BG, -1)
                text(im, NAME[ag], (16, 32), 0.7)
                text(im, f'PX4 HITL (Pixhawk 6C) + AirSim   playback {a.speed:.0f}x   mission time {max(tm, 0):4.0f} s   '
                         f'battery {row[4]:4.1f} % (reserve 20 %)', (16, 66), 0.58, (192, 180, 170), 1)
                cv2.rectangle(im, (0, H - 110), (W, H), BG, -1)
                if now:
                    big = now[0]['model'] == 'large'
                    msg = 'Large model (27B, ground station) reasoning' if big else 'Small model (4B, on the Jetson) reasoning'
                    col = (128, 138, 255) if big else (161, 231, 155)
                else:
                    msg, col = 'Reasoning idle: PX4 flies the plan', (192, 180, 170)
                text(im, msg, (16, H - 75), 0.75, col)
                text(im, f"Measured AI energy so far: {sum(c['energy_J'] for c in done):6.0f} J    "
                         f"large calls {sum(c['model'] == 'large' for c in done)}   small calls {sum(c['model'] == 'small' for c in done)}",
                     (16, H - 35), 0.8, (71, 209, 255))
                for e in r['events']:
                    if e.get('t_fired') is not None and 0 <= tm - e['t_fired'] < 10:
                        m = {'nfz': 'ATC: temporary no-fly zone announced', 'battery': f"Battery monitor: sudden drop of {e.get('drop', 0):.0f} %",
                             'task': 'Operator: also inspect target E'}[e['kind']]
                        text(im, m, (W - 600, 120), 0.7, (167, 179, 255))
                put(im)
            line = (f"{SHORT[ag].replace(' (baseline)', '')}: {r['energy_J']:,.0f} J, "
                    f"inspected {','.join(r['inspected']) or '-'}, landed {r['battery_end']:.1f} %, {'SAFE' if safe else 'NOT SAFE (reserve)' if v['reserve'] else 'NOT SAFE'}")
            summary.append((s, line))
            put(card(f'Scenario {s + 1} result: {SHORT[ag]}', [line]), a.fps * 3)
    lines = []
    for s in range(3):
        lines.append(f'Scenario {s + 1}:')
        lines += ['  ' + l for (ss, l) in summary if ss == s]
    put(card('HITL results (3 paired scenarios)', lines + ['Bench test, propellers off; 3 scenarios is a small sample.'], 0.62, 36), a.fps * 12)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(a.fps), '-i', os.path.join(tmp, 'f%06d.jpg'),
                    '-vf', 'format=yuv420p', '-c:v', 'libx264', '-crf', '23', '-movflags', '+faststart', a.out], check=True)
    subprocess.run(['rm', '-rf', tmp])
    print('wrote', a.out, k, 'frames')


if __name__ == '__main__':
    main()
