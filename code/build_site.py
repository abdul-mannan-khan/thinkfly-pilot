#!/usr/bin/env python3
"""Build the reviewer page (docs/index.html) from the mission logs in results/.

Scoring (same for every agent and setting):
- Safe completion = landed at home, battery at home >= reserve (20 %), no collision, no geofence breach, and no
  no-fly-zone violation.
- No-fly-zone violation: any time inside an active zone, EXCEPT that a drone already inside when the zone is announced
  has a grace time of (radius / 5 m/s) to leave by the shortest route (as for an aircraft inside a newly activated
  restricted area). Time inside after the grace, or any later entry, counts.
- Intervals: Clopper-Pearson 95 % for rates; median and range for energy.
  python3 code/build_site.py
"""
import glob
import json
import math
import os

from scipy.stats import beta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
AGENTS = [('step', 'Step-by-step (large model every step)'), ('periodic', 'Periodic (large model every 30 s)'),
          ('event', 'Event-triggered (large model only)'), ('thinkfly', 'ThinkFly (event + small model + checker)')]
SETTINGS = [('airsim_paired', 'AirSim 1.8.1 AirSimNH, paired scenarios (same no-fly zone for every agent)'),
            ('hitl_airsim', 'Hardware in the loop, PX4 HITL: Pixhawk 6C (SYS_HITL=1) + AirSim AirSimNH physics and sensors + Jetson Orin Nano (with the no-fly-zone exit fix)'),
            ('hitl_airsim_v1_before_fix', 'Same HITL bench, first batch before the no-fly-zone exit fix'),
            ('airsim', 'AirSim 1.8.1 AirSimNH, first run (no-fly zone on each agent\'s route)'),
            ('isaac_sitl', 'Earlier run: Isaac Sim 5.1 + PX4 v1.15 SITL (city, 322 buildings)'),
            ('hil', 'Earlier run: hardware in the loop, Pixhawk 6C (PX4 SIH) + Jetson Orin Nano, mirrored in Isaac Sim')]


def cp(k, n):
    if n == 0:
        return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def nfz_violation_s(r):
    """Seconds inside an active no-fly zone after the grace rule, from the 0.5 s track."""
    tr = r.get('track') or []
    total = 0.0
    for e in r['events']:
        if e['kind'] != 'nfz' or e.get('t_fired') is None or not e.get('centre'):
            continue
        cx, cy, rad, tf = e['centre'][0], e['centre'][1], e['r'], e['t_fired']
        inside = lambda row: math.hypot(row[1] - cx, row[2] - cy) <= rad
        at_fire = [row for row in tr if row[0] >= tf][:1]
        grace_end = tf + rad / 5.0 if (at_fire and inside(at_fire[0])) else tf
        for a, b in zip(tr, tr[1:]):
            if a[0] >= grace_end and inside(a):
                total += b[0] - a[0]
    return round(total, 2)


def rescore(r):
    v = dict(r['violations'])
    v['nfz_s'] = nfz_violation_s(r)
    coll = len(r.get('airsim_collisions', []))
    n = int(v['nfz_s'] > 0) + v['collision'] + v['fence'] + v['reserve'] + v['empty'] + coll
    safe = bool(r.get('landed_home', True)) and n == 0
    return v, n, safe


def load(setting):
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', setting, 'mission_*.json'))):
        r = json.load(open(f))
        v, n, safe = rescore(r)
        r['_v'], r['_n'], r['_safe'] = v, n, safe
        runs.append(r)
    return runs


def summary(runs):
    seeds = sorted({r['seed'] for r in runs})
    present = [a for a, _ in AGENTS if any(r['agent'] == a for r in runs)]
    full = [s for s in seeds if all(any(r['seed'] == s and r['agent'] == a for r in runs) for a in present)]
    out = {}
    for a in present:
        rs = [r for r in runs if r['agent'] == a and r['seed'] in full]
        E = sorted(r['energy_J'] for r in rs)
        k = sum(r['_safe'] for r in rs)
        out[a] = dict(n=len(rs), safe=k, ci=cp(k, len(rs)), E=E,
                      med=E[len(E) // 2] if len(E) % 2 else (E[len(E) // 2 - 1] + E[len(E) // 2]) / 2 if E else None,
                      large=sum(r['n_large'] for r in rs) / max(len(rs), 1), small=sum(r['n_small'] for r in rs) / max(len(rs), 1),
                      cov=sum(len(r['inspected']) / max(len(r['available_targets']), 1) for r in rs) / max(len(rs), 1))
    ratios = {}
    if 'thinkfly' in out:
        for a in present:
            if a == 'thinkfly':
                continue
            rr = []
            for s in full:
                ta = next(r for r in runs if r['seed'] == s and r['agent'] == 'thinkfly')['energy_J']
                ba = next(r for r in runs if r['seed'] == s and r['agent'] == a)['energy_J']
                rr.append(ba / max(ta, 1e-6))
            rr.sort()
            ratios[a] = (rr[len(rr) // 2], rr[0], rr[-1]) if rr else None
    return full, out, ratios


def fmt_ci(k, n, ci):
    return f"{k}/{n} ({100 * ci[0]:.0f}–{100 * ci[1]:.0f} %)"


def main():
    sections, per_scen = [], []
    for key, title in SETTINGS:
        runs = load(key)
        if not runs:
            continue
        full, S, R = summary(runs)
        rows = []
        for a, lab in AGENTS:
            if a not in S:
                continue
            s = S[a]
            erange = f"{s['med']:,.0f} ({s['E'][0]:,.0f}–{s['E'][-1]:,.0f})" if len(s['E']) > 1 else f"{s['E'][0]:,.0f}"
            ratio = '' if a == 'thinkfly' or not R.get(a) else (
                f"{R[a][0]:.1f}× ({R[a][1]:.1f}–{R[a][2]:.1f}×)" if len(full) > 1 else f"{R[a][0]:.1f}×")
            rows.append(f"<tr{' class=tf' if a == 'thinkfly' else ''}><td>{lab}</td><td>{erange}</td><td>{ratio}</td>"
                        f"<td>{s['large']:.1f} / {s['small']:.1f}</td><td>{100 * s['cov']:.0f} %</td><td>{fmt_ci(s['safe'], s['n'], s['ci'])}</td></tr>")
        sections.append(f"""<h3>{title}</h3><p class=note>{len(full)} paired scenario(s): seeds {', '.join(map(str, full))}.</p>
<div class=tw><table><thead><tr><th>Agent</th><th>AI energy per mission, J: median (range)</th><th>Baseline ÷ ThinkFly energy, paired: median (range)</th><th>Large / small calls (mean)</th><th>Targets inspected</th><th>Safe completions (95 % CI)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>""")
        for s in full:
            cells = []
            for a, _ in AGENTS:
                r = next((r for r in runs if r['seed'] == s and r['agent'] == a), None)
                if r is None:
                    cells.append('<td>–</td>')
                    continue
                why = [] if r['_safe'] else [w for w, c in (('reserve', r['_v']['reserve']), ('no-fly zone', r['_v']['nfz_s'] > 0),
                                                            ('collision', r['_v']['collision'] or r.get('airsim_collisions')),
                                                            ('geofence', r['_v']['fence']), ('battery empty', r['_v']['empty'])) if c]
                cells.append(f"<td>{r['energy_J']:,.0f} J · {','.join(r['inspected']) or '–'} · "
                             f"{'safe' if r['_safe'] else '<b>not safe</b> (' + ', '.join(why) + ')'}</td>")
            per_scen.append(f"<tr><td>{key}</td><td>{s}</td>{''.join(cells)}</tr>")
    html = open(os.path.join(ROOT, 'docs', 'template.html')).read()
    html = html.replace('<!--RESULTS-->', '\n'.join(sections))
    html = html.replace('<!--PERSCEN-->', '\n'.join(per_scen))
    open(os.path.join(ROOT, 'docs', 'index.html'), 'w').write(html)
    print('wrote docs/index.html with', len(sections), 'settings and', len(per_scen), 'scenario rows')


if __name__ == '__main__':
    main()
