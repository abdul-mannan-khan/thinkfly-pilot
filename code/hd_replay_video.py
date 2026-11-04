#!/usr/bin/env python3
"""Compose the HD replay video of the ThinkFly PX4-HITL bench flights: AirSim frames (hd_replay_render.py) + overlay
(agent, battery, current plan, which model is reasoning, measured AI energy, event banners, top-down mini-map) +
title / scenario / result cards. 1920x1080, 30 fps.

  hd_replay_video.py PLAN.json FRAMES_DIR MISSION_DIR SCENE.json OUT.mp4 [--workers 8]
Safety verdicts use the reviewer page's scoring (build_site.rescore)."""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_site import rescore  # noqa: E402

W, H, FPS = 1920, 1080, 30
FD = os.environ.get('FONT_DIR', '/usr/share/fonts/truetype/dejavu')


def font(sz, bold=False):
    return ImageFont.truetype(os.path.join(FD, 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), sz)


F = {k: font(*v) for k, v in {'h1': (58, True), 'h2': (40, True), 'b': (30, True), 'm': (28, False), 's': (24, False),
                              'xs': (20, False), 'mb': (24, True)}.items()}
INK, INK2, BG = (240, 236, 232), (178, 186, 196), (15, 23, 32)
GREEN, PURPLE, AMBER, RED, BLUE, GREY = (110, 220, 140), (190, 150, 255), (255, 190, 70), (255, 95, 90), (100, 180, 255), (170, 176, 184)
SHORT = {'thinkfly': 'ThinkFly', 'event': 'Event-triggered (baseline)', 'step': 'Step-by-step (baseline)'}
NAME = {'thinkfly': 'Small model on the drone plans; a rule checker approves; the large model is asked only on alarm',
        'event': 'Large model (ground station) re-plans at every event; no onboard model, no checker',
        'step': 'Large model (ground station) re-plans at every step of the mission; no checker'}
FOOT = ('Replay of the recorded PX4-HITL flight path in AirSim (position logged at 4 Hz, attitude derived from it). '
        'Decisions and energy measured on the bench, 4 Nov 2026.')


def card(title, lines, sub=None):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((120, 150), title, font=F['h1'], fill=INK)
    y = 250
    if sub:
        d.text((120, 235), sub, font=F['s'], fill=INK2)
        y = 310
    for ln in lines:
        if ln == '':
            y += 22
            continue
        f, col = (F['mb'], INK) if ln.startswith('#') else (F['m'], (225, 225, 225))
        ln = ln.lstrip('#')
        d.text((120, y), ln, font=f, fill=col)
        y += 46
    return im


def timeline(r):
    ev = []
    for res in r['calls']:
        s0 = res['t']
        for c in res['calls']:
            ev.append((s0, s0 + c['latency_s'], c))
            s0 += c['latency_s']
        res['_end'] = s0
    return ev


def plan_text(p):
    if not p:
        return 'waiting for the first plan'
    if p.get('return_home_now'):
        return 'return home now'
    def name(x):
        if isinstance(x, dict):
            rs = (x.get('reason') or '').replace('-', ' ')
            return f"{x.get('id', '?')} ({rs})" if rs else str(x.get('id', '?'))
        return str(x)
    order = ' > '.join(name(x) for x in (p.get('order') or [])) or '(none)'
    sk = ', '.join(name(x) for x in (p.get('skip') or []))
    return f'visit {order}, then home' + (f'   (skip {sk})' if sk else '')


EVTXT = {'nfz': lambda e: f"Air traffic control: temporary no-fly zone over target {e.get('over', '?')} (radius {e['r']:.0f} m)",
         'battery': lambda e: f"Battery monitor: sudden drop of {e.get('drop', 0):.0f} % (cell imbalance)",
         'task': lambda e: f"Operator: please also inspect new target {e.get('id', 'E')}"}


def panel(im, box, alpha=170):
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(box, 14, fill=(12, 18, 26, alpha))
    im.alpha_composite(ov)


CTX = {}


def init(ctx):
    CTX.update(ctx)


def overlay(job):
    out, src, mi, k = job
    m, ctx = CTX[mi]
    r, scen, scene = ctx['r'], ctx['scen'], ctx['scene']
    f = m['frames'][k]
    tm = f['tm']
    try:
        im = Image.open(src).convert('RGBA')
    except Exception:  # noqa: BLE001
        im = Image.new('RGBA', (W, H), (40, 40, 40, 255))
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    tr = np.array(r['track'])
    i = int(np.clip(np.searchsorted(tr[:, 0], tm), 0, len(tr) - 1))
    bat = tr[i, 4] if tm >= 0 else tr[0, 4]
    dtk = (m['frames'][k + 1]['tm'] - tm) if k + 1 < len(m['frames']) else (tm - m['frames'][k - 1]['tm'])
    speed = dtk * FPS
    # top bar
    panel(im, (0, 0, W, 112), 190)
    d = ImageDraw.Draw(im)
    d.text((28, 14), f"{SHORT[m['agent']]}  ·  Scenario {m['scenario'] + 1} of 3", font=F['h2'], fill=INK)
    d.text((28, 68), NAME[m['agent']], font=F['s'], fill=INK2)
    d.text((W - 640, 18), f'flight time {max(tm, 0):5.0f} s    replay {speed:.1f}×', font=F['m'], fill=INK)
    # battery bar
    bx, by, bw = W - 640, 68, 600
    d.rounded_rectangle((bx, by, bx + bw, by + 28), 6, fill=(45, 52, 62))
    col = GREEN if bat >= scen['reserve'] + 5 else AMBER if bat >= scen['reserve'] else RED
    d.rounded_rectangle((bx, by, bx + int(bw * min(bat, 100) / 50), by + 28), 6, fill=col)
    rx = bx + int(bw * scen['reserve'] / 50)
    d.line((rx, by - 4, rx, by + 32), fill=INK, width=3)
    d.text((bx + 8, by + 1), f'battery {bat:4.1f} %   (reserve {scen["reserve"]:.0f} %, white line)', font=F['xs'], fill=(10, 10, 10))
    # event banner
    for e in r['events']:
        if e.get('t_fired') is not None and 0 <= tm - e['t_fired'] < 9:
            s = EVTXT[e['kind']](e)
            tw = d.textlength(s, font=F['b'])
            panel(im, (W / 2 - tw / 2 - 30, 132, W / 2 + tw / 2 + 30, 192), 215)
            d = ImageDraw.Draw(im)
            d.text((W / 2 - tw / 2, 144), s, font=F['b'], fill=AMBER)
    # reasoning / plan / energy panel
    ev = ctx['ev']
    done = [c for (s0, s1, c) in ev if s1 <= tm]
    now = [c for (s0, s1, c) in ev if s0 <= tm < s1]
    last_small_rejected = bool(done) and done[-1]['model'] == 'small' and done[-1].get('check') is False
    if now:
        c = now[0]
        if c['model'] == 'large':
            msg, mc = ('Checker rejected the onboard plan: asking the large model (27B, ground station)' if last_small_rejected
                       else 'Large model (27B, ground station) is planning…'), PURPLE
        else:
            msg, mc = 'Small model (4B, on the Jetson) is planning…', GREEN
    else:
        msg, mc = 'No model running: PX4 is flying the current plan', GREY
    plan = None
    for res in r['calls']:
        if res['_end'] <= tm:
            plan = res.get('plan')
    E = sum(c['energy_J'] for c in done)
    panel(im, (24, H - 236, 1360, H - 56), 185)
    d = ImageDraw.Draw(im)
    d.ellipse((48, H - 208, 72, H - 184), fill=mc)
    d.text((86, H - 214), msg, font=F['b'], fill=mc)
    d.text((48, H - 166), 'Current plan: ' + plan_text(plan), font=F['m'], fill=INK)
    if plan and plan.get('note'):
        d.text((48, H - 128), '“' + textwrap.shorten(plan['note'], 95, placeholder='…') + '”', font=F['s'], fill=INK2)
    d.text((48, H - 94), f'AI energy so far {E:6,.0f} J    ·    large-model calls {sum(c["model"] == "large" for c in done)}'
                         f'    ·    small-model calls {sum(c["model"] == "small" for c in done)}', font=F['mb'], fill=(255, 215, 90))
    # mini-map
    MS, L = 400, scene['extent'][1]
    mx, my = W - MS - 24, H - MS - 52
    panel(im, (mx - 8, my - 46, mx + MS + 8, my + MS + 8), 200)
    d = ImageDraw.Draw(im)
    sc = MS / (2 * L)

    def px(x, y):
        return mx + (x + L) * sc, my + (L - y) * sc
    for b in scene['buildings']:
        (cx, cy), (sx, sy) = b['centre'], b['size']
        x0, y0 = px(cx - sx / 2, cy + sy / 2)
        x1, y1 = px(cx + sx / 2, cy - sy / 2)
        d.rectangle((x0, y0, x1, y1), fill=(85, 92, 100))
    for e in r['events']:
        if e['kind'] == 'nfz' and e.get('t_fired') is not None and tm >= e['t_fired']:
            ox, oy = px(*e['centre'])
            rr = e['r'] * sc
            ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).ellipse((ox - rr, oy - rr, ox + rr, oy + rr), fill=(255, 80, 80, 80), outline=(255, 95, 90, 255), width=2)
            im.alpha_composite(ov)
            d = ImageDraw.Draw(im)
    tg = dict(scen['targets'])
    for e in r['events']:
        if e['kind'] == 'task' and e.get('t_fired') is not None and tm >= e['t_fired']:
            tg[e['id']] = e['xy']
    for name, (x, y) in tg.items():
        seen = name in r['inspected'] and (np.hypot(tr[:i + 1, 1] - x, tr[:i + 1, 2] - y) < 15).any() and tm >= 0
        qx, qy = px(x, y)
        d.ellipse((qx - 9, qy - 9, qx + 9, qy + 9), fill=GREEN if seen else None, outline=GREEN if seen else AMBER, width=3)
        d.text((qx + 11, qy - 24), name, font=F['mb'], fill=INK)
    hx, hy = px(0, 0)
    d.polygon([(hx, hy - 10), (hx - 9, hy + 7), (hx + 9, hy + 7)], fill=INK)
    if tm >= 0 and i > 0:
        d.line([px(row[1], row[2]) for row in tr[:i + 1]], fill=BLUE, width=3)
    # drone marker at the replay position (ENU: east = y_ned, north = x_ned)
    xd, yd = f['drone'][1], f['drone'][0]
    qx, qy = px(xd, yd)
    yaw = f['drone'][5]
    d.ellipse((qx - 7, qy - 7, qx + 7, qy + 7), fill=(255, 255, 255))
    d.line((qx, qy, qx + 18 * math.sin(yaw), qy - 18 * math.cos(yaw)), fill=(255, 255, 255), width=3)
    d.text((mx + 4, my - 38), 'Map, north up: targets, no-fly zone, path', font=F['xs'], fill=INK2)
    # footer
    panel(im, (0, H - 44, W, H), 200)
    d = ImageDraw.Draw(im)
    d.text((28, H - 36), FOOT, font=F['xs'], fill=(200, 205, 212))
    im.convert('RGB').save(out, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('frames')
    ap.add_argument('missions')
    ap.add_argument('scene')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only', type=int, default=None)
    ap.add_argument('--max', type=int, default=None)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    scene = json.load(open(a.scene))
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    seq = []                     # ('card', image) or ('frame', job)
    n = 0

    def add_card(img, secs):
        nonlocal n
        p = os.path.join(tmp, f'card{n}.jpg')
        img.save(p, quality=95)
        seq.append((p, int(secs * FPS)))
        n += 1
    if a.only is None:
        add_card(card('ThinkFly on real PX4 hardware in the loop', [
            'The Pixhawk 6C flight controller runs real PX4 1.15.1 in HITL mode; AirSim computes physics and sensors.',
            'ThinkFly and the small model (Qwen3.5-4B) run on a Jetson Orin Nano; energy from its VDD_IN rail.',
            'The large model (Qwen3.5-27B) runs on a ground-station GPU; energy measured per call.',
            '',
            '#Task: inspect the rooftop targets and land home with at least 20 % battery.',
            'Every scenario brings three surprises, the same for every agent:',
            '   a temporary no-fly zone over a target, a sudden battery drop, and a new target E.',
            '',
            'Each agent is shown in turn. Watch the plan, which model is thinking, and the energy it costs.',
        ], 'Bench test, propellers off, 4 November 2026  ·  UKRI APP125712 proof of concept'), 9)
    jobs = []
    sums = []
    allctx = {}
    for mi, m in enumerate(plan):
        if a.only is not None and mi != a.only:
            continue
        r = json.load(open(os.path.join(a.missions, f"mission_{m['scenario']}_{m['agent']}.json")))
        scen = json.load(open(os.path.join(a.missions, f"scenario_{m['scenario']}.json")))
        v, nv, safe = rescore(r)
        ev = timeline(r)
        ctx = {'r': r, 'scen': scen, 'scene': scene, 'ev': ev}
        allctx[mi] = (m, ctx)
        evl = [f"{e['t_fired']:5.0f} s   " + EVTXT[e['kind']](e) for e in r['events'] if e.get('t_fired') is not None]
        add_card(card(f"Scenario {m['scenario'] + 1} of 3  ·  {SHORT[m['agent']]}", [
            NAME[m['agent']] + '.', '', f"Start battery {scen['battery0']:.1f} %, reserve {scen['reserve']:.0f} %. "
            f"Targets {', '.join(sorted(scen['targets']))}.", '', '#What happens during the flight:', *evl]), 5)
        start = len(seq)
        fr = m['frames'][:a.max]
        for k in range(len(fr)):
            out = os.path.join(tmp, f'm{mi}_{k:05d}.jpg')
            jobs.append((out, os.path.join(a.frames, f'm{mi}', f'{k:05d}.jpg'), mi, k))
            seq.append((out, 1))
        verdict = 'SAFE' if safe else ('NOT SAFE: landed below the 20 % reserve' if v.get('reserve') else 'NOT SAFE')
        line = (f"{SHORT[m['agent']].replace(' (baseline)', '')}: {r['energy_J']:,.0f} J of AI energy "
                f"({r['n_large']} large / {r['n_small']} small calls), inspected {', '.join(r['inspected']) or 'none'}, "
                f"landed with {r['battery_end']:.1f} %")
        sums.append((m['scenario'], line, verdict))
        add_card(card(f"Result  ·  Scenario {m['scenario'] + 1}  ·  {SHORT[m['agent']]}", [line, '', '#' + verdict]), 4)
    if a.only is None:
        L = []
        for s in range(3):
            L.append(f'#Scenario {s + 1}')
            L += [f'   {ln}  —  {vd}' for (ss, ln, vd) in sums if ss == s]
        L += ['', 'ThinkFly: safe in 3 of 3; each baseline safe in 2 of 3. Three scenarios is a small sample.']
        im = card('Results on the hardware bench', [])
        d = ImageDraw.Draw(im)
        y = 250
        for ln in L:
            f = F['mb'] if ln.startswith('#') else F['s']
            d.text((120, y), ln.lstrip('#'), font=f, fill=INK if ln.startswith('#') else (225, 225, 225))
            y += 44 if ln else 20
        add_card(im, 14)
    with Pool(a.workers, initializer=init, initargs=(allctx,)) as p:
        for i, _ in enumerate(p.imap(overlay, jobs, chunksize=16)):
            if i % 1000 == 0:
                print('overlay', i, '/', len(jobs), flush=True)
    lst = os.path.join(tmp, 'list.txt')
    with open(lst, 'w') as fh:
        for pth, nfr in seq:
            for _ in range(nfr):
                fh.write(f"file '{pth}'\nduration {1 / FPS:.6f}\n")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-r', str(FPS),
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out],
                   check=True)
    print('wrote', a.out, sum(nfr for _, nfr in seq), 'frames', flush=True)


if __name__ == '__main__':
    main()
