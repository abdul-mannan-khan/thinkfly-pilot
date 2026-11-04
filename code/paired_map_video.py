#!/usr/bin/env python3
"""Top-down replay video of the nine paired AirSim scenarios, drawn from the mission logs (no camera footage).
Each scenario shows the four agents side by side on the AirSimNH house map, same events for every agent.
  paired_map_video.py RUN_DIR OUT.mp4 [--speed 10] [--fps 15]
Safety verdicts use the same scoring as the reviewer page (code/build_site.py, no-fly-zone grace rule)."""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thinkfly-pilot-public', 'code'))
from build_site import rescore, summary  # noqa: E402

W, H = 1280, 720
ORDER = [('thinkfly', 'ThinkFly (small model + checker, large on alarm)'), ('event', 'Event-triggered, large model only'),
         ('periodic', 'Periodic, large model every 30 s'), ('step', 'Step-by-step, large model every step')]
BG, PANEL, INK, INK2 = (32, 23, 15), (48, 38, 28), (240, 236, 232), (190, 178, 168)
GREEN, RED, AMBER, BLUE = (120, 200, 90), (80, 80, 235), (60, 170, 240), (240, 190, 90)


def text(im, s, xy, sc=0.5, col=INK, th=1):
    cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def card(title, lines, sub=None):
    im = np.full((H, W, 3), BG, np.uint8)
    text(im, title, (60, 120), 1.2, INK, 2)
    if sub:
        text(im, sub, (60, 160), 0.6, INK2)
    for k, l in enumerate(lines):
        text(im, l, (60, 220 + 36 * k), 0.62, INK if not l.startswith('  ') else INK2)
    return im


def timeline(r):
    ev = []
    for res in r['calls']:
        s0 = res['t']
        for c in res['calls']:
            ev.append((s0, s0 + c['latency_s'], c))
            s0 += c['latency_s']
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--speed', type=float, default=10.0)
    ap.add_argument('--fps', type=int, default=15)
    a = ap.parse_args()
    scene = json.load(open(os.path.join(a.run, 'airsimnh_scene.json')))
    L = scene['extent'][1]
    runs = []
    for f in sorted(glob.glob(os.path.join(a.run, 'mission_*.json'))):
        r = json.load(open(f))
        r['_v'], r['_n'], r['_safe'] = rescore(r)
        runs.append(r)
    full, S, R = summary(runs)
    tmp = tempfile.mkdtemp()
    k = 0

    def put(im, n=1):
        nonlocal k
        for _ in range(n):
            cv2.imwrite(os.path.join(tmp, f'f{k:06d}.png'), im)
            k += 1

    put(card('ThinkFly: nine paired AirSim scenarios', [
        'UKRI APP125712 proof of concept (University of West London + University of Glasgow)',
        'AirSim 1.8.1, AirSimNH neighbourhood; mission replayed from the flight logs (top-down map)',
        'Task: inspect four rooftops, return home with at least 20 % battery',
        'Each scenario: no-fly zone over one target, sudden battery drop, extra target E',
        'Every agent meets exactly the same events (paired comparison)',
        'Energy = measured GPU energy of every model call (NVML), idle subtracted',
        'Models: Qwen3.5-4B (small, onboard role), Qwen3.5-27B (large, ground station)'],
        'Simulation run 3 November 2026'), a.fps * 7)

    PW, PH = 640, 325
    MS = 300                                 # map square in each panel
    sc = MS / (2 * L)
    for seed in full:
        scen = json.load(open(os.path.join(a.run, f'scenario_{seed}.json')))
        rs = {r['agent']: r for r in runs if r['seed'] == seed}
        T = max(r['track'][-1][0] for r in rs.values())
        tl = {ag: timeline(r) for ag, r in rs.items()}
        tracks = {ag: np.array(r['track']) for ag, r in rs.items()}
        put(card(f'Scenario {seed + 1} of {len(full)} (seed {seed})', [
            f"Start battery {scen['battery0']:.1f} %; reserve {scen['reserve']:.0f} %",
            *[('  ' + n) for n in rs['thinkfly']['notes'][:3]]]), a.fps * 3)
        for tm in np.arange(0, T + 3, a.speed / a.fps):
            im = np.full((H, W, 3), BG, np.uint8)
            text(im, f'Scenario {seed + 1}/{len(full)}   mission time {tm:5.0f} s   replay {a.speed:.0f}x   (top-down map from flight logs)',
                 (16, 34), 0.62, INK, 1)
            for p, (ag, lab) in enumerate(ORDER):
                r, tr = rs[ag], tracks[ag]
                ox, oy = (p % 2) * PW, 60 + (p // 2) * (PH + 5)
                cv2.rectangle(im, (ox + 4, oy), (ox + PW - 4, oy + PH), PANEL, -1)
                mx, my = ox + 12, oy + 14

                def px(x, y):
                    return int(mx + (x + L) * sc), int(my + (L - y) * sc)
                cv2.rectangle(im, (mx, my), (mx + MS, my + MS), (60, 50, 40), -1)
                for b in scene['buildings']:
                    (cx, cy), (sx, sy) = b['centre'], b['size']
                    cv2.rectangle(im, px(cx - sx / 2, cy + sy / 2), px(cx + sx / 2, cy - sy / 2), (110, 100, 92), -1)
                for e in r['events']:
                    if e['kind'] == 'nfz' and e.get('t_fired') is not None and tm >= e['t_fired']:
                        ov = im.copy()
                        cv2.circle(ov, px(*e['centre']), int(e['r'] * sc), RED, -1)
                        cv2.addWeighted(ov, 0.35, im, 0.65, 0, im)
                        cv2.circle(im, px(*e['centre']), int(e['r'] * sc), RED, 1, cv2.LINE_AA)
                i = min(int(np.searchsorted(tr[:, 0], tm)), len(tr) - 1)
                done_t = tm >= tr[-1, 0]
                tg = dict(scen['targets'])
                for e in r['events']:
                    if e['kind'] == 'task' and e.get('t_fired') is not None and tm >= e['t_fired']:
                        tg[e['id']] = e['xy']
                for name, (x, y) in tg.items():
                    seen = name in r['inspected'] and (np.hypot(tr[:i + 1, 1] - x, tr[:i + 1, 2] - y) < 15).any()
                    cv2.circle(im, px(x, y), 7, GREEN if seen else AMBER, -1 if seen else 2, cv2.LINE_AA)
                    text(im, name, (px(x, y)[0] + 8, px(x, y)[1] - 6), 0.45, INK)
                cv2.drawMarker(im, px(0, 0), INK, cv2.MARKER_TRIANGLE_UP, 12, 2)
                pts = np.array([px(row[1], row[2]) for row in tr[:i + 1]], np.int32)
                if len(pts) > 1:
                    cv2.polylines(im, [pts], False, BLUE, 2, cv2.LINE_AA)
                cv2.circle(im, px(tr[i, 1], tr[i, 2]), 5, (255, 255, 255), -1, cv2.LINE_AA)
                # info column
                ix = mx + MS + 14
                text(im, lab.split(' (')[0].split(',')[0], (ix, oy + 30), 0.62, INK, 2)
                text(im, lab[len(lab.split(' (')[0].split(',')[0]):].strip(' ,()'), (ix, oy + 50), 0.42, INK2)
                bat = tr[i, 4]
                cv2.rectangle(im, (ix, oy + 64), (ix + 260, oy + 80), (70, 60, 50), -1)
                cv2.rectangle(im, (ix, oy + 64), (ix + int(260 * max(bat, 0) / 100), oy + 80),
                              GREEN if bat >= 25 else (AMBER if bat >= 20 else RED), -1)
                x20 = ix + int(260 * 0.2)
                cv2.line(im, (x20, oy + 60), (x20, oy + 84), INK, 1)
                text(im, f'battery {bat:4.1f} %  (line = 20 % reserve)', (ix, oy + 100), 0.45, INK2)
                done = [c for (s0, s1, c) in tl[ag] if s1 <= tm]
                now = [c for (s0, s1, c) in tl[ag] if s0 <= tm < s1]
                E = sum(c['energy_J'] for c in done)
                text(im, f'AI energy {E:6.0f} J', (ix, oy + 132), 0.7, (90, 210, 255), 2)
                text(im, f"large calls {sum(c['model'] == 'large' for c in done)}   small calls {sum(c['model'] == 'small' for c in done)}",
                     (ix, oy + 156), 0.45, INK2)
                if done_t:
                    ok = r['_safe']
                    why = [w for w, c in (('below reserve', r['_v']['reserve']), ('no-fly zone', r['_v']['nfz_s'] > 0),
                                          ('collision', r['_v']['collision'] or r.get('airsim_collisions')),
                                          ('geofence', r['_v']['fence']), ('battery empty', r['_v']['empty'])) if c]
                    cv2.rectangle(im, (ix, oy + 176), (ix + 270, oy + 212), GREEN if ok else RED, -1)
                    text(im, 'SAFE COMPLETION' if ok else 'NOT SAFE: ' + ', '.join(why), (ix + 8, oy + 200), 0.5 if ok else 0.42, BG, 2 if ok else 1)
                    text(im, f"inspected {','.join(r['inspected']) or '-'}; landed {r['battery_end']:.1f} %", (ix, oy + 236), 0.45, INK2)
                elif now:
                    big = now[0]['model'] == 'large'
                    text(im, 'Large model (27B) reasoning' if big else 'Small model (4B) reasoning', (ix, oy + 196), 0.5, RED if big else GREEN, 1)
                else:
                    text(im, 'flying the plan, no reasoning', (ix, oy + 196), 0.5, INK2)
                for e in r['events']:
                    if e.get('t_fired') is not None and 0 <= tm - e['t_fired'] < 10 and not done_t:
                        m = {'nfz': 'no-fly zone announced', 'battery': f"battery drop {e.get('drop', 0):.0f} %",
                             'task': f"operator: add target {e.get('id', 'E')}"}[e['kind']]
                        text(im, m, (ix, oy + 270), 0.5, AMBER, 1)
            put(im)
        lines = []
        for ag, lab in ORDER:
            r = rs[ag]
            lines.append(f"{lab.split(' (')[0].split(',')[0]:<16} {r['energy_J']:7,.0f} J   "
                         f"{'safe' if r['_safe'] else 'NOT safe'}   inspected {','.join(r['inspected']) or '-'}")
        put(card(f'Scenario {seed + 1}: result', lines), a.fps * 4)

    names = dict(ORDER)
    lines = []
    for ag, lab in ORDER:
        s = S[ag]
        rt = '' if ag == 'thinkfly' else f"   {R[ag][0]:.1f}x ThinkFly's energy (paired median)"
        lines.append(f"{lab.split(' (')[0].split(',')[0]:<16} safe {s['safe']}/{s['n']} "
                     f"(95 % CI {100 * s['ci'][0]:.0f}-{100 * s['ci'][1]:.0f} %)   median {s['med']:,.0f} J{rt}")
    lines += ['', '  Baselines have no rule checker; most of their failures were landing below the 20 % reserve.',
              '  In 2 of 9 scenarios a baseline used less energy than ThinkFly. Simulation only; HIL bench test separate.',
              '  Logs, code and scoring: github.com/abdul-mannan-khan/thinkfly-pilot']
    put(card('Nine paired scenarios: summary', lines), a.fps * 10)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(a.fps), '-i', os.path.join(tmp, 'f%06d.png'),
                    '-vf', 'format=yuv420p', '-c:v', 'libx264', '-crf', '23', '-movflags', '+faststart', a.out], check=True)
    subprocess.run(['rm', '-rf', tmp])
    print('wrote', a.out, k, 'frames')


if __name__ == '__main__':
    main()
