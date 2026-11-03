#!/usr/bin/env python3
"""Video of the AirSim (AirSimNH) ThinkFly runs: chase-camera frames + a status overlay per mission.
  airsim_video.py RUN_DIR OUT.mp4 [--speed 4] [--fps 15]
RUN_DIR: output of thinkfly_airsim.py --record (frames/, frames.csv, mission_*.json)."""
import argparse
import bisect
import glob
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np

NAME = {'thinkfly': 'ThinkFly: event-triggered, small onboard model + rule checker, large model only on alarm',
        'step': 'Baseline: step-by-step agent, large model at every step',
        'periodic': 'Baseline: large model re-plans every 30 s',
        'event': 'Ablation: event-triggered, large model only'}
ORDER = ['thinkfly', 'event', 'periodic', 'step']
W, H = 1280, 720


def text(im, s, xy, scale=0.8, col=(255, 255, 255), th=2):
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def card(lines, title):
    im = np.full((H, W, 3), (32, 23, 15), np.uint8)
    text(im, title, (60, 140), 1.4, (255, 255, 255), 3)
    for k, l in enumerate(lines):
        text(im, l, (60, 230 + 48 * k), 0.8, (232, 222, 215), 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--speed', type=float, default=4.0)
    ap.add_argument('--fps', type=int, default=15)
    a = ap.parse_args()
    rows = [l.strip().split(',') for l in open(os.path.join(a.run, 'frames.csv')) if l.strip()]
    ft = [float(r[0]) for r in rows]
    fn = [r[1] for r in rows]
    runs = {json.load(open(f))['agent']: json.load(open(f)) for f in glob.glob(os.path.join(a.run, 'mission_*.json'))}
    tmp = tempfile.mkdtemp()
    k = 0

    def put(im, n=1):
        nonlocal k
        for _ in range(n):
            cv2.imwrite(os.path.join(tmp, f'f{k:06d}.jpg'), im)
            k += 1
    put(card(['Proof of concept for UKRI APP125712 (UWL + University of Glasgow)',
              'Second simulator: Microsoft AirSim 1.8.1, AirSimNH neighbourhood, SimpleFlight quadrotor',
              'Same mission code, models, rule checker and energy meter as the Isaac Sim and HIL runs',
              'Qwen3.5-4B (onboard role) and Qwen3.5-27B (ground station) on a metered GPU',
              'Rooftop inspection; no-fly zone, battery drop and extra target during the flight',
              'Simulation, 3 November 2026'], 'ThinkFly in AirSim'), a.fps * 6)
    for ag in [x for x in ORDER if x in runs]:
        r = runs[ag]
        t0 = r.get('t0_sim')
        tr = np.array(r['track'])
        T = tr[-1, 0]
        sp = a.speed if ag == 'thinkfly' else a.speed * 2
        evs = []
        for res in r['calls']:
            s0 = res['t']
            for c in res['calls']:
                evs.append((s0, s0 + c['latency_s'], c))
                s0 += c['latency_s']
        for tm in np.arange(0, T + 2, sp / a.fps):
            i = min(bisect.bisect_left(ft, t0 + tm), len(fn) - 1)
            im = cv2.imread(os.path.join(a.run, 'frames', fn[i]))
            im = cv2.resize(im, (W, H)) if im is not None else np.zeros((H, W, 3), np.uint8)
            row = tr[min(np.searchsorted(tr[:, 0], tm), len(tr) - 1)]
            done = [c for (s0, s1, c) in evs if s1 <= tm]
            now = [c for (s0, s1, c) in evs if s0 <= tm < s1]
            E = sum(c['energy_J'] for c in done)
            nL = sum(c['model'] == 'large' for c in done)
            nS = sum(c['model'] == 'small' for c in done)
            cv2.rectangle(im, (0, 0), (W, 80), (32, 23, 15), -1)
            text(im, NAME[ag], (16, 32), 0.75)
            text(im, f'AirSim (AirSimNH), playback {sp:.0f}x   mission time {tm:4.0f} s   battery {row[4]:4.1f} % (reserve 20 %)',
                 (16, 66), 0.6, (192, 180, 170), 1)
            cv2.rectangle(im, (0, H - 110), (W, H), (32, 23, 15), -1)
            if now:
                msg = 'Large model (27B) reasoning' if now[0]['model'] == 'large' else 'Small onboard model (4B) reasoning'
                col = (128, 138, 255) if now[0]['model'] == 'large' else (161, 231, 155)
            else:
                msg, col = 'Reasoning idle: flight controller flies the plan', (192, 180, 170)
            text(im, msg, (16, H - 75), 0.75, col)
            text(im, f'Measured AI energy so far: {E:6.0f} J    large calls {nL}   small calls {nS}', (16, H - 35), 0.8, (71, 209, 255))
            for e in r['events']:
                if e.get('t_fired') is not None and tm >= e['t_fired'] and tm - e['t_fired'] < 12:
                    m = {'nfz': 'ATC: temporary no-fly zone announced', 'battery': f"Battery monitor: sudden drop of {e.get('drop', 0):.0f} %",
                         'task': 'Operator: also inspect target E'}[e['kind']]
                    text(im, m, (W - 620, 120), 0.7, (167, 179, 255))
            put(im)
        put(card([f"Inspected {', '.join(r['inspected']) or 'none'}; skipped {', '.join(r['skipped']) or 'none'}",
                  f"Collisions {len(r.get('airsim_collisions', []))}; rule violations {r['n_violations']}",
                  f"Measured AI energy {r['energy_J']:.0f} J; large calls {r['n_large']}, small calls {r['n_small']}"],
                 NAME[ag].split(':')[0] + ': result'), a.fps * 3)
    lines = [f"{ag:>9}: {runs[ag]['energy_J']:6.0f} J, large {runs[ag]['n_large']}, small {runs[ag]['n_small']}, "
             f"inspected {','.join(runs[ag]['inspected']) or '-'}, {'safe' if runs[ag]['safe_completion'] else 'NOT safe'}"
             for ag in ORDER if ag in runs]
    put(card(lines + ['One scenario in a second simulator; a first result, not a benchmark.'], 'Results (AirSim)'), a.fps * 8)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(a.fps), '-i', os.path.join(tmp, 'f%06d.jpg'),
                    '-vf', 'format=yuv420p', '-c:v', 'libx264', '-crf', '22', a.out], check=True)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
