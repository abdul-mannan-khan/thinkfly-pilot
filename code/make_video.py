#!/usr/bin/env python3
"""Proof-of-concept video: Isaac Sim camera (left) + live mission map and AI-energy panel (right) for each recorded
mission, with title and results cards.
  make_video.py REC_DIR RESULTS_DIR OUT.mp4 [--speed 4] [--fps 20]
REC_DIR: output of `run_batch.sh REC_DIR <seed> <agents> record` (flight_frames/, sim/states.csv, mission_*.json).
RESULTS_DIR: batch results (summary.json, fig_energy.png) for the results card.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

NAME = {'thinkfly': 'ThinkFly: event-triggered, small onboard model + rule checker, large model only on alarm',
        'step': 'Baseline: step-by-step agent, large model at every step',
        'periodic': 'Baseline: large model re-plans every 30 s',
        'event': 'Ablation: event-triggered, large model only'}
COL = {'step': '#8a8f98', 'periodic': '#c07a1c', 'event': '#2f5d9a', 'thinkfly': '#2e8b57'}
SCENE = os.environ.get('THINKFLY_SCENE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scene', 'urban_isaac.json'))


def card(path, title, lines, img=None):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#0f1720')
    fig.text(0.5, 0.86, title, ha='center', color='white', fontsize=38, weight='bold')
    for k, l in enumerate(lines):
        fig.text(0.5, 0.74 - k * 0.055, l, ha='center', color='#d7dee8', fontsize=21)
    if img:
        ax = fig.add_axes([0.2, 0.04, 0.6, 0.45])
        ax.imshow(Image.open(img))
        ax.axis('off')
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_states(p):
    t, xyz = [], []
    with open(p) as f:
        r = csv.reader(f)
        hdr = next(r)
        for row in r:
            try:
                t.append(float(row[0]))
                xyz.append([float(row[2]), float(row[3]), float(row[4])])
            except (ValueError, IndexError):
                pass
    return np.array(t), np.array(xyz)


def offset(track, st_t, st_x):
    """Sim-time offset so that mission time + offset = Isaac sim time (match positions)."""
    tr = np.array(track)
    best, bo = 1e18, 0.0
    for o in np.arange(0, st_t[-1], 0.5):
        idx = np.searchsorted(st_t, tr[::10, 0] + o)
        ok = idx < len(st_t)
        if ok.sum() < 5:
            continue
        e = np.linalg.norm(st_x[idx[ok]] - tr[::10][ok, 1:4], axis=1).mean()
        if e < best:
            best, bo = e, o
    return bo, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('res')
    ap.add_argument('out')
    ap.add_argument('--speed', type=float, default=4.0)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--rec-fps', type=float, default=10.0)
    ap.add_argument('--mp4', default=None, help='read Isaac frames from the recorded flight.mp4 (one frame per recorded frame)')
    ap.add_argument('--tmp', default='/root')
    ap.add_argument('--mode', choices=['sitl', 'hil', 'airsim'], default='sitl')
    ap.add_argument('--offsets', default='', help='agent=seconds,... : sim-time offsets found from the video (overrides matching)')
    a = ap.parse_args()
    meta = json.load(open(SCENE))
    st_t, st_x = load_states(os.path.join(a.rec, 'sim', 'states.csv'))
    frames = sorted(glob.glob(os.path.join(a.rec, 'flight_frames', 'f*.png')))
    t_frame0 = st_t[0]
    tmp = tempfile.mkdtemp(dir=a.tmp)
    cap = None
    if a.mp4:
        import cv2
        cap = cv2.VideoCapture(a.mp4)
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = list(range(nfr))
    def get_frame(i):
        if cap is None:
            return Image.open(frames[i])
        import cv2
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        return im[:, :, ::-1] if ok else np.zeros((720, 1280, 3), np.uint8)
    segs = []
    S = json.load(open(os.path.join(a.res, 'summary.json')))
    A = S['agents']
    red = S.get('reduction_vs_step_x')
    TITLE = {
        'sitl': ['PX4 X500 drone in an Isaac Sim city (322 buildings), inspection mission given in plain language',
                 'Disturbances in flight: temporary no-fly zone, battery drop, extra target from the operator',
                 'Real open models on a metered GPU: Qwen3.5-4B (onboard role) and Qwen3.5-27B (ground station)',
                 'Software-in-the-loop simulation, 3 November 2026'],
        'hil': ['HARDWARE IN THE LOOP: real PX4 1.15 on a Pixhawk 6C flight controller (SIH physics on the board)',
                'ThinkFly + Qwen3.5-4B run ON the Jetson Orin Nano companion computer; energy from its power rail',
                'Qwen3.5-27B at the ground-station GPU, energy metered there; Isaac Sim mirrors the flight',
                'Same mission and disturbances as the simulation runs. Bench test, props off, 3 November 2026']}
    card(os.path.join(tmp, 'title.png'), 'ThinkFly: think in proportion to need',
         ['Proof of concept for UKRI APP125712 (UWL + University of Glasgow)'] + TITLE[a.mode])
    segs.append(('title.png', 6))
    missions = sorted(glob.glob(os.path.join(a.rec, 'mission_*.json')),
                      key=lambda f: 0 if 'thinkfly' in f else 1)
    nseg = 0
    for mf in missions:
        r = json.load(open(mf))
        sc = json.load(open(glob.glob(os.path.join(a.rec, 'scenario_*.json'))[0]))
        o, err = offset(r['track'], st_t, st_x)
        ov = dict(x.split('=') for x in a.offsets.split(',') if x)
        if r['agent'] in ov:
            o, err = float(ov[r['agent']]), float('nan')
        print(os.path.basename(mf), 'offset', o, 'match error m', round(err, 2))
        tr = np.array(r['track'])
        T = tr[-1, 0]
        calls = [(c['w0'], c['w1'], c) for res in r['calls'] for c in res['calls']]
        # wall->mission time for calls: use the res['t'] (mission time when reasoning started) + latency
        evs = []
        for res in r['calls']:
            t0 = res['t']
            for c in res['calls']:
                evs.append((t0, t0 + c['latency_s'], c))
                t0 += c['latency_s']
        tg = dict(sc['targets'])
        tg['E'] = sc['events'][2]['xy']
        speed = a.speed if r['agent'] == 'thinkfly' else a.speed * 2
        times = np.arange(0, T + 3, speed / a.fps)
        fdir = os.path.join(tmp, f'm{nseg}')
        os.makedirs(fdir)
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        for k, tm in enumerate(times):
            fig.clf()
            fig.patch.set_facecolor('#0f1720')
            sim_t = tm + o
            fi = int(round((sim_t - t_frame0) * a.rec_fps))
            ax0 = fig.add_axes([0.0, 0.25, 0.667, 0.667 * 1.0])
            if frames:
                ax0.imshow(get_frame(min(max(fi, 0), len(frames) - 1)))
            ax0.axis('off')
            fig.text(0.01, 0.955, NAME[r['agent']], color='white', fontsize=19, weight='bold')
            fig.text(0.01, 0.925, (f'HIL: Pixhawk 6C (PX4 SIH) + Jetson Orin Nano, mirrored in Isaac Sim. Playback {speed:.0f}x' if a.mode == 'hil' else f'Isaac Sim + PX4 SITL (X500), urban scene. Playback {speed:.0f}x'), color='#aab4c0', fontsize=13)
            # map
            ax = fig.add_axes([0.675, 0.40, 0.315, 0.56])
            ax.set_facecolor('#f4f4f4')
            for b in meta['buildings']:
                (cx, cy), (sx, sy) = b['centre'], b['size']
                ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color='#888' if b['height'] > 26 else '#ccc', lw=0))
            for e in r['events']:
                if e['kind'] == 'nfz' and e.get('t_fired') is not None and tm >= e['t_fired']:
                    c = e['centre']
                    ax.add_patch(Circle(c, e['r'], color='red', alpha=0.28, lw=0))
                    ax.text(c[0], c[1], 'NO-FLY', color='darkred', fontsize=8, ha='center')
            m = tr[:, 0] <= tm
            ax.plot(tr[m, 1], tr[m, 2], color=COL[r['agent']], lw=2)
            if m.any():
                ax.plot(tr[m][-1, 1], tr[m][-1, 2], marker='o', color='k', ms=7)
            ins_t = {}
            for nline in r['notes']:
                pass
            for kk, (x_, y_) in tg.items():
                if kk == 'E' and not any(e['kind'] == 'task' and e.get('t_fired') is not None and tm >= e['t_fired'] for e in r['events']):
                    continue
                ax.plot(x_, y_, marker='D', ms=8, color='white', mec='k')
                ax.text(x_ + 12, y_ + 12, kk, fontsize=10, weight='bold')
            ax.plot(*sc['home'], marker='s', color='k', ms=7)
            ax.set_xlim(-360, 360)
            ax.set_ylim(-360, 360)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            # status
            row = tr[min(np.searchsorted(tr[:, 0], tm), len(tr) - 1)]
            done = [c for (s0, s1, c) in evs if s1 <= tm]
            now = [c for (s0, s1, c) in evs if s0 <= tm < s1]
            Eacc = sum(c['energy_J'] for c in done)
            nL = sum(c['model'] == 'large' for c in done)
            nS = sum(c['model'] == 'small' for c in done)
            if now:
                c = now[0]
                think = ('ALARM: large model (27B, ground station) analysing' if c['model'] == 'large' and c.get('escalated')
                         else ('Large model (27B) reasoning' if c['model'] == 'large' else 'Small onboard model (4B) reasoning'))
                tcol = '#ff8a80' if c['model'] == 'large' else '#9be7a1'
            else:
                think, tcol = 'Reasoning idle: low-level controllers fly the plan', '#aab4c0'
            fig.text(0.68, 0.36, f'Mission time {tm:5.0f} s     Battery {row[4]:5.1f} %  (reserve 20 %)', color='white', fontsize=15)
            fig.text(0.68, 0.32, think, color=tcol, fontsize=15, weight='bold')
            fig.text(0.68, 0.27, f'Measured AI energy so far: {Eacc:7.0f} J', color='#ffd166', fontsize=19, weight='bold')
            fig.text(0.68, 0.235, f'Large-model calls {nL}    Small-model calls {nS}', color='white', fontsize=15)
            msgs = [e for e in r['events'] if e.get('t_fired') is not None and tm >= e['t_fired']]
            y = 0.19
            for e in msgs[-3:]:
                txt = {'nfz': 'ATC: temporary no-fly zone announced', 'battery': f"Battery monitor: sudden drop of {e.get('drop', 0):.0f} %",
                       'task': 'Operator: also inspect target E'}[e['kind']]
                fig.text(0.68, y, f"{e['t_fired']:.0f} s  {txt}", color='#ffb3a7', fontsize=13)
                y -= 0.035
            ax0b = fig.add_axes([0.01, 0.03, 0.65, 0.2])
            ax0b.set_facecolor('#0f1720')
            ax0b.axis('off')
            ins = ', '.join(r['inspected']) or 'none'
            ax0b.text(0, 0.8, f"Final result: inspected {ins}; skipped {', '.join(r['skipped']) or 'none'}; "
                              f"{'no rule violations' if r['n_violations'] == 0 else 'RULE VIOLATION'}; "
                              f"total measured AI energy {r['energy_J']:.0f} J", color='#aab4c0', fontsize=13, transform=ax0b.transAxes)
            fig.savefig(os.path.join(fdir, f'f{k:05d}.png'), facecolor=fig.get_facecolor())
        plt.close(fig)
        segs.append((f'm{nseg}', None))
        nseg += 1
    # results card
    lines = []
    for ag in ('step', 'periodic', 'event', 'thinkfly'):
        if ag in A:
            s = A[ag]
            lines.append(f"{ag:>9}: {s['energy_J_mean']:7.0f} J per mission, {s['large_calls_mean']:.1f} large calls, "
                         f"safe {s['safe_completions']}/{s['n']}")
    if red:
        rv = S.get('reduction_vs_event_large_x', 0)
        lines.append(f'ThinkFly used {red:.1f}x less measured AI energy than step-by-step')
        lines.append(f'and {rv:.1f}x less than event-triggered (large only)' if rv >= 1 else
                     f'but {1/rv:.1f}x more than event-triggered (large only): 3 small-model plans were escalated')
        lines.append(f'Pilot evidence: {len(S["scenarios"])} paired city mission(s); all flights safe; a first result, not a benchmark')
    card(os.path.join(tmp, 'results.png'), 'Results (HIL bench test)' if a.mode == 'hil' else 'Results (simulation pilot)', lines,
         os.path.join(a.res, 'fig_energy.png'))
    segs.append(('results.png', 10))
    card(os.path.join(tmp, 'end.png'), 'Next: from simulation to flight',
         ['UKRI project: control-theory triggers (WP1), distilled onboard model + temporal-logic checker (WP2),',
          'Jetson Orin NX energy measurement (WP3), indoor flights with small drones (WP4).',
          ('Limits: bench HIL (props off, SIH physics), one scenario, hand-set triggers.' if a.mode == 'hil' else 'Limits of this pilot: simulated flights, hand-set triggers, energy measured on an RTX 4090.'),
          'Code, logs and data will be released openly.'])
    segs.append(('end.png', 6))
    parts = []
    for k, (name, dur) in enumerate(segs):
        p = os.path.join(tmp, f'part{k}.mp4')
        if dur:
            cmd = ['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-t', str(dur), '-i', os.path.join(tmp, name),
                   '-r', str(a.fps), '-vf', 'scale=1920:1080,format=yuv420p', '-c:v', 'libx264', '-crf', '20', p]
        else:
            cmd = ['ffmpeg', '-v', 'error', '-y', '-framerate', str(a.fps), '-i', os.path.join(tmp, name, 'f%05d.png'),
                   '-vf', 'scale=1920:1080,format=yuv420p', '-c:v', 'libx264', '-crf', '20', p]
        subprocess.run(cmd, check=True)
        parts.append(p)
    lst = os.path.join(tmp, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', a.out], check=True)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
