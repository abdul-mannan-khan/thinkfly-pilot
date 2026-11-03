#!/usr/bin/env python3
"""Summarise ThinkFly pilot missions: per-agent table (Clopper-Pearson 95 % intervals), figures.
  analyze.py RESULTS_DIR [--scene urban.json]
Writes RESULTS_DIR/summary.json, summary.md, fig_energy.png, fig_calls.png, fig_map_<seed>.png
"""
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.stats import beta

AGENTS = ['step', 'periodic', 'event', 'thinkfly']
LABEL = {'step': 'Step-by-step\n(large every step)', 'periodic': 'Periodic\n(large every 30 s)',
         'event': 'Event-triggered\n(large only)', 'thinkfly': 'ThinkFly\n(event + small + checker)'}
COL = {'step': '#8a8f98', 'periodic': '#c07a1c', 'event': '#2f5d9a', 'thinkfly': '#2e8b57'}


def cp(k, n, a=0.05):
    if n == 0:
        return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def main(d, scene):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'mission_*.json')))]
    seeds = sorted({r['seed'] for r in runs})
    # paired: keep scenarios that every present agent finished
    present = [a for a in AGENTS if any(r['agent'] == a for r in runs)]
    full = [s for s in seeds if all(any(r['seed'] == s and r['agent'] == a for r in runs) for a in present)]
    by = {a: [r for r in runs if r['agent'] == a and r['seed'] in full] for a in present}
    S = {'scenarios': full, 'agents': {}}
    for a, rs in by.items():
        n = len(rs)
        k = sum(r['safe_completion'] for r in rs)
        v = sum(r['n_violations'] > 0 for r in rs)
        E = np.array([r['energy_J'] for r in rs])
        cov = [len(r['inspected']) / max(len(r['available_targets']), 1) for r in rs]
        small_pass = [p for r in rs for p in r.get('small_check_pass', [])]
        S['agents'][a] = {
            'n': n, 'safe_completions': int(k), 'safe_rate_CI95': cp(k, n),
            'missions_with_violation': int(v), 'violation_rate_CI95': cp(v, n),
            'energy_J_mean': float(E.mean()), 'energy_J_sd': float(E.std(ddof=1)) if n > 1 else 0.0,
            'energy_J_median': float(np.median(E)),
            'energy_small_J_mean': float(np.mean([r['energy_small_J'] for r in rs])),
            'energy_large_J_mean': float(np.mean([r['energy_large_J'] for r in rs])),
            'large_calls_mean': float(np.mean([r['n_large'] for r in rs])),
            'small_calls_mean': float(np.mean([r['n_small'] for r in rs])),
            'escalations_mean': float(np.mean([r['n_escalations'] for r in rs])),
            'small_checker_pass_rate': (sum(small_pass) / len(small_pass)) if small_pass else None,
            'small_checker_pass_CI95': cp(sum(small_pass), len(small_pass)) if small_pass else None,
            'reasoning_time_s_mean': float(np.mean([r['reasoning_time_s'] for r in rs])),
            'mission_time_s_mean': float(np.mean([r['mission_time_s'] for r in rs])),
            'coverage_mean': float(np.mean(cov)),
            'tokens_in_mean': float(np.mean([r['tokens_in'] for r in rs])),
            'tokens_out_mean': float(np.mean([r['tokens_out'] for r in rs])),
            'energy_per_safe_completion_J': float(E.sum() / k) if k else None,
            'violations': [r['violations'] for r in rs],
        }
    base = S['agents'].get('step')
    tf = S['agents'].get('thinkfly')
    if base and tf and tf['energy_J_mean'] > 0:
        S['reduction_vs_step_x'] = base['energy_J_mean'] / tf['energy_J_mean']
        paired = [next(r for r in by['step'] if r['seed'] == s)['energy_J'] /
                  max(next(r for r in by['thinkfly'] if r['seed'] == s)['energy_J'], 1e-6) for s in full]
        S['reduction_vs_step_paired_x'] = {'median': float(np.median(paired)), 'min': float(min(paired)),
                                          'max': float(max(paired))}
    if S['agents'].get('periodic') and tf and tf['energy_J_mean'] > 0:
        S['reduction_vs_periodic_x'] = S['agents']['periodic']['energy_J_mean'] / tf['energy_J_mean']
    if S['agents'].get('event') and tf and tf['energy_J_mean'] > 0:
        S['reduction_vs_event_large_x'] = S['agents']['event']['energy_J_mean'] / tf['energy_J_mean']
    json.dump(S, open(os.path.join(d, 'summary.json'), 'w'), indent=1)

    # table
    L = ['| Agent | Missions | Safe completions (95 % CI) | Missions with a rule violation | AI energy per mission, J (mean ± s.d.) | Large-model calls | Small-model calls | Escalations | Small plans passing the checker | Targets inspected / available | Mission time, s |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for a in present:
        s = S['agents'][a]
        lo, hi = s['safe_rate_CI95']
        sp = '–' if s['small_checker_pass_rate'] is None else f"{100*s['small_checker_pass_rate']:.0f} %"
        L.append(f"| {LABEL[a].replace(chr(10), ' ')} | {s['n']} | {s['safe_completions']}/{s['n']} ({100*lo:.0f}–{100*hi:.0f} %) | "
                 f"{s['missions_with_violation']}/{s['n']} | {s['energy_J_mean']:.0f} ± {s['energy_J_sd']:.0f} | {s['large_calls_mean']:.1f} | "
                 f"{s['small_calls_mean']:.1f} | {s['escalations_mean']:.1f} | {sp} | {100*s['coverage_mean']:.0f} % | {s['mission_time_s_mean']:.0f} |")
    open(os.path.join(d, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    for k in ('reduction_vs_step_x', 'reduction_vs_step_paired_x', 'reduction_vs_periodic_x', 'reduction_vs_event_large_x'):
        if k in S:
            print(k, S[k])

    # figure: energy per mission
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    for i, a in enumerate(present):
        E = [r['energy_J'] for r in by[a]]
        ax.bar(i, np.mean(E), color=COL[a], alpha=0.85, width=0.62)
        ax.scatter(np.full(len(E), i) + np.linspace(-0.15, 0.15, len(E)), E, s=12, color='black', zorder=3)
        ax.text(i, np.mean(E) * 1.15, f'{np.mean(E):.0f} J', ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([LABEL[a] for a in present], fontsize=7.5)
    ax.set_ylabel('Measured AI energy per mission (J, log)')
    ax.set_title(f'Reasoning-GPU energy per city inspection mission ({len(full)} paired scenarios; dots = missions)', fontsize=8.5)
    ax.grid(axis='y', alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'fig_energy.png'))

    # figure: calls
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    x = np.arange(len(present))
    ax.bar(x - 0.18, [S['agents'][a]['large_calls_mean'] for a in present], 0.36, label='large-model calls (27B)', color='#555')
    ax.bar(x + 0.18, [S['agents'][a]['small_calls_mean'] for a in present], 0.36, label='small-model calls (4B, onboard)', color='#2e8b57')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in present], fontsize=7.5)
    ax.set_ylabel('Calls per mission (mean)')
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'fig_calls.png'))

    # maps
    meta = json.load(open(scene))
    for s in full[:3]:
        fig, axs = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.4), dpi=170)
        axs = np.atleast_1d(axs)
        for ax, a in zip(axs, present):
            r = next(r for r in by[a] if r['seed'] == s)
            sc = json.load(open(os.path.join(d, f'scenario_{s}.json')))
            for b in meta['buildings']:
                (cx, cy), (sx, sy) = b['centre'], b['size']
                ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy,
                                       color='#999' if b['height'] > sc['alt'] - 4 else '#ddd', lw=0))
            for cx, cy, rr, n in r['nfz']:
                ax.add_patch(Circle((cx, cy), rr, color='red', alpha=0.25, lw=0))
            tr = np.array(r['track'])
            if len(tr):
                ax.plot(tr[:, 1], tr[:, 2], color=COL[a], lw=1.4)
            tg = dict(sc['targets'])
            tg['E'] = sc['events'][2]['xy']
            for k, (x_, y_) in tg.items():
                done = k in r['inspected']
                ax.plot(x_, y_, marker='o', ms=6, color='#2e8b57' if done else 'white', mec='k')
                ax.text(x_ + 10, y_ + 10, k, fontsize=7)
            ax.plot(*sc['home'], marker='s', color='k', ms=5)
            ax.set_xlim(-360, 360)
            ax.set_ylim(-360, 360)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{LABEL[a].split(chr(10))[0]}: {r['energy_J']:.0f} J, "
                         f"{'safe' if r['safe_completion'] else 'VIOLATION'}", fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f'fig_map_{s}.png'))
    plt.close('all')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scene', 'urban_isaac.json'))
