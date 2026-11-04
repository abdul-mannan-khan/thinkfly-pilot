#!/usr/bin/env python3
"""Fig. 2 of the Vision and Approach: AirSim proof of concept.
(a) AirSim simulation still, (b) PX4-HITL bench (HD replay still), (c) measured AI energy per mission
(median and range) with safe-mission counts, for the nine paired AirSim scenarios and the three HITL scenarios.
  make_poc_figure_airsim.py SIM_STILL.jpg HIL_STILL.jpg OUT.png"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, '..', 'results', 'airsim_paired')
HIL = os.path.join(HERE, '..', 'results', 'hitl_airsim')
sys.path.insert(0, HERE)
from build_site import rescore  # noqa: E402

AGENTS = [('step', 'Step-by-step'), ('periodic', 'Periodic'), ('event', 'Event\n(large only)'), ('thinkfly', 'ThinkFly')]
BLUE, ORANGE = '#2f5f9e', '#c47a1e'


def stats(d):
    out = {}
    for f in glob.glob(os.path.join(d, 'mission_*.json')):
        r = json.load(open(f))
        v, n, safe = rescore(r)
        out.setdefault(r['agent'], []).append((r['energy_J'], safe))
    return {a: (np.median([e for e, _ in L]), min(e for e, _ in L), max(e for e, _ in L), sum(s for _, s in L), len(L))
            for a, L in out.items()}


def crop(im, aspect):
    h, w = im.shape[:2]
    if w / h > aspect:
        nw = int(h * aspect)
        x0 = (w - nw) // 2
        return im[:, x0:x0 + nw]
    nh = int(w / aspect)
    y0 = (h - nh) // 2
    return im[y0:y0 + nh]


def main():
    sim_im, hil_im, out = sys.argv[1:4]
    S, Hh = stats(SIM), stats(HIL)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
    fig = plt.figure(figsize=(20.1, 4.86), dpi=100)
    axa = fig.add_axes([0.005, 0.05, 0.325, 0.80])
    axb = fig.add_axes([0.338, 0.05, 0.325, 0.80])
    axc = fig.add_axes([0.715, 0.17, 0.28, 0.68])
    for ax, path, title in [(axa, sim_im, '(a) Simulation: AirSim, 9 paired scenarios'),
                            (axb, hil_im, '(b) HIL: real PX4 on Pixhawk 6C + Jetson')]:
        im = plt.imread(path)
        h, w = im.shape[:2]
        zw, zh = int(w / 1.6), int(h / 1.6)                 # centre crop: the follow camera keeps the drone central
        im = im[(h - zh) // 2:(h + zh) // 2, (w - zw) // 2:(w + zw) // 2]
        ax.imshow(crop(im, 1.78))
        ax.set_axis_off()
        ax.set_title(title, loc='left', fontsize=18)
    x = np.arange(len(AGENTS))
    wdt = 0.38
    for k, (a, lab) in enumerate(AGENTS):
        for off, D, col in [(-wdt / 2, S, BLUE), (wdt / 2, Hh, ORANGE)]:
            if a not in D:
                axc.text(x[k] + off, 150, 'n/a', ha='center', va='bottom', fontsize=10, color='#666')
                continue
            med, lo, hi, safe, n = D[a]
            axc.bar(x[k] + off, med, wdt, color=col)
            axc.errorbar(x[k] + off, med, yerr=[[med - lo], [hi - med]], fmt='none', ecolor='#333', capsize=3, lw=1)
            axc.text(x[k] + off, hi + 120, f'{safe}/{n}', ha='center', va='bottom', fontsize=12.5,
                     color='#1b7f3b' if safe == n else '#b3261e', fontweight='bold')
    axc.set_xticks(x)
    axc.set_xticklabels([lab for _, lab in AGENTS], fontsize=13.5)
    axc.set_ylabel('AI energy per mission (J)', fontsize=14)
    axc.tick_params(axis='y', labelsize=13)
    axc.set_ylim(0, 9300)
    axc.spines[['top', 'right']].set_visible(False)
    axc.set_title('(c) Measured AI energy and safe missions', loc='left', fontsize=18)
    from matplotlib.patches import Patch
    axc.legend(handles=[Patch(color=BLUE, label='Simulation (9 scenarios)'), Patch(color=ORANGE, label='HIL bench (3 scenarios)')],
               frameon=False, fontsize=12.5, loc='upper right')
    fig.savefig(out, dpi=100)
    print('wrote', out)


if __name__ == '__main__':
    main()
