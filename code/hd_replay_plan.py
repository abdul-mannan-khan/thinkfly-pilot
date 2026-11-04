#!/usr/bin/env python3
"""Plan an HD replay of the ThinkFly PX4-HITL bench flights in AirSim (no hardware needed).

Input: pose.csv (wall time, true NED position of the HITL drone at 4 Hz, logged with simGetVehiclePose during the bench
run) and the mission logs. Each mission is matched to its flight segment (same method as hitl_airsim_video.py), the path
is smoothed with a cubic spline, and for every output frame we compute
  - the drone pose: position from the spline, heading = direction of travel (turn rate <= 90 deg/s), tilt from the
    path's acceleration (<= 20 deg);
  - a follow camera: 12 m behind and 5 m above the drone, heading follows the drone slowly (critically damped, tau 4 s,
    <= 15 deg/s of flight time), always looking at the drone.
Frames are spaced 0.1 s of flight time (3x at 30 fps) and 0.05 s for 8 s after each mission event (1.5x).

  hd_replay_plan.py POSE_CSV MISSION_DIR OUT_PLAN.json --first-seg 9
"""
import argparse
import csv
import json
import math
import os

import numpy as np
from scipy.interpolate import CubicSpline

ORDER = ['thinkfly', 'event', 'step']
G = 9.81
CAM_BACK, CAM_UP = 12.0, 5.0
DT_FAST, DT_SLOW, SLOW_AFTER_EVENT = 0.1, 0.05, 8.0


def segments(P):
    segs, on = [], False
    for i, (w, x, y, z) in enumerate(P):
        if z < -3 and not on:
            on, s = True, i
        if z > -1 and on:
            on = False
            segs.append((s, i))
    return segs


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def match_offset(P, i0, i1, tr):
    pw, pe, pn = P[i0:i1, 0], P[i0:i1, 2], P[i0:i1, 1]
    best = None
    for off in np.arange(pw[0] - 30, pw[0] + 60, 0.05):
        tw = off + tr[:, 0]
        m = (tw >= pw[0]) & (tw <= pw[-1])
        if m.sum() < 20:
            continue
        err = float(np.mean(np.hypot(np.interp(tw[m], pw, pe) - tr[m, 1], np.interp(tw[m], pw, pn) - tr[m, 2])))
        if best is None or err < best[0]:
            best = (err, off)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pose')
    ap.add_argument('missions')
    ap.add_argument('out')
    ap.add_argument('--first-seg', type=int, required=True)
    a = ap.parse_args()
    P = np.array([[float(v) for v in r] for r in csv.reader(open(a.pose))])
    segs = segments(P)
    plan, seg_i = [], a.first_seg
    for s in range(3):
        for ag in ORDER:
            r = json.load(open(os.path.join(a.missions, f'mission_{s}_{ag}.json')))
            tr = np.array(r['track'])
            i0, i1 = segs[seg_i]
            seg_i += 1
            err, off = match_offset(P, i0, i1, tr)
            T = float(tr[-1, 0])
            # pose window: some ground time before take-off and after landing
            j0, j1 = max(i0 - 60, 0), min(i1 + 40, len(P))
            w = P[j0:j1, 0]
            keep = np.concatenate([[True], np.diff(w) > 1e-3])
            w, X = w[keep], P[j0:j1, 1:4][keep]
            sp = CubicSpline(w, X, bc_type='natural')
            # frame times (mission time)
            ev_t = [e['t_fired'] for e in r['events'] if e.get('t_fired') is not None]
            t, ts = -6.0, []
            while t <= T + 3.0:
                ts.append(t)
                slow = any(0 <= t - te < SLOW_AFTER_EVENT for te in ev_t)
                t += DT_SLOW if slow else DT_FAST
            ts = np.array(ts)
            tw = np.clip(off + ts, w[0], w[-1])
            pos, vel, acc = sp(tw), sp(tw, 1), sp(tw, 2)
            # heading of travel, rate-limited drone yaw, slow camera yaw
            yaw = cam = None
            frames = []
            for k in range(len(ts)):
                dt = ts[k] - ts[k - 1] if k else DT_FAST
                vh = math.hypot(vel[k, 0], vel[k, 1])
                want = math.atan2(vel[k, 1], vel[k, 0]) if vh > 1.5 else (yaw if yaw is not None else 0.0)
                if yaw is None:
                    yaw = want
                    # first frames: face the first direction of real travel
                    fut = np.where(np.hypot(vel[:, 0], vel[:, 1]) > 1.5)[0]
                    if len(fut):
                        yaw = want = math.atan2(vel[fut[0], 1], vel[fut[0], 0])
                    cam = yaw
                yaw = wrap(yaw + float(np.clip(wrap(want - yaw), -math.radians(90) * dt, math.radians(90) * dt)))
                step = wrap(yaw - cam) * (1 - math.exp(-dt / 4.0))
                cam = wrap(cam + float(np.clip(step, -math.radians(15) * dt, math.radians(15) * dt)))
                # tilt from acceleration in the body frame (NED: nose down = negative pitch)
                cy, sy = math.cos(yaw), math.sin(yaw)
                af = acc[k, 0] * cy + acc[k, 1] * sy
                al = -acc[k, 0] * sy + acc[k, 1] * cy
                lim = math.radians(20)
                pitch = float(np.clip(-math.atan2(af, G), -lim, lim)) if vh > 0.3 or abs(af) > 0.3 else 0.0
                roll = float(np.clip(math.atan2(al, G), -lim, lim)) if vh > 0.3 or abs(al) > 0.3 else 0.0
                x, y, z = (float(v) for v in pos[k])
                z = min(z, -0.15)                                  # stay above the ground
                cx, cyy = x - CAM_BACK * math.cos(cam), y - CAM_BACK * math.sin(cam)
                cz = z - CAM_UP
                d = math.hypot(x - cx, y - cyy)
                frames.append({'tm': round(float(ts[k]), 3), 'drone': [x, y, z, roll, pitch, yaw],
                               'cam': [cx, cyy, cz, -math.atan2(z - cz, d), math.atan2(y - cyy, x - cx)]})
            plan.append({'scenario': s, 'agent': ag, 'offset': off, 'match_err_m': err, 'T': T, 'frames': frames})
            print(f'scenario {s} {ag}: match {err:.1f} m, {len(frames)} frames', flush=True)
    json.dump(plan, open(a.out, 'w'))
    print('frames total', sum(len(p['frames']) for p in plan))


if __name__ == '__main__':
    main()
