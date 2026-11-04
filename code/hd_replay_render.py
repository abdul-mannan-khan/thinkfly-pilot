#!/usr/bin/env python3
"""Render the HD replay frames planned by hd_replay_plan.py in a running AirSimNH (ExternalPhysicsEngine, external
'follow' camera). For each frame: place the drone, place the camera, wait for a rendered frame, grab it.

  hd_replay_render.py PLAN.json OUT_DIR [--only 0] [--max N] [--settle 0.04]
Writes OUT_DIR/m<mission>/<k>.jpg and OUT_DIR/m<mission>/done when a mission is complete (resumable)."""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from airsim_rpc import AirSimRPC, image_request

V = 'Drone1'


def quat(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return {'w_val': cr * cp * cy + sr * sp * sy, 'x_val': sr * cp * cy - cr * sp * sy,
            'y_val': cr * sp * cy + sr * cp * sy, 'z_val': cr * cp * sy - sr * sp * cy}


def pose(x, y, z, r, p, yaw):
    return {'position': {'x_val': x, 'y_val': y, 'z_val': z}, 'orientation': quat(r, p, yaw)}


def grab(c):
    res = c.call('simGetImages', [image_request('follow', 0, False)], '', True)[0]
    h, w = res['height'], res['width']
    buf = np.frombuffer(res['image_data_uint8'], np.uint8)
    return buf.reshape(h, w, -1)[:, :, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('out')
    ap.add_argument('--only', type=int, default=None)
    ap.add_argument('--max', type=int, default=None)
    ap.add_argument('--settle', type=float, default=0.04)
    ap.add_argument('--step', type=int, default=1)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    c = AirSimRPC()
    print('ping', c.call('ping'), flush=True)
    for cmd in ['sg.ViewDistanceQuality 3', 'sg.AntiAliasingQuality 3', 'sg.ShadowQuality 3', 'sg.PostProcessQuality 3',
                'sg.TextureQuality 3', 'sg.EffectsQuality 3', 'sg.FoliageQuality 3', 'r.MotionBlurQuality 0',
                'r.EyeAdaptationQuality 0', 'foliage.LODDistanceScale 3', 'r.StaticMeshLODDistanceScale 0.5']:
        try:
            c.call('simRunConsoleCommand', cmd)
        except Exception as e:  # noqa: BLE001
            print('console', cmd, e, flush=True)
    t0, n = time.time(), 0
    for mi, m in enumerate(plan):
        if a.only is not None and mi != a.only:
            continue
        d = os.path.join(a.out, f'm{mi}')
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, 'done')):
            continue
        fr = m['frames'][::a.step][:a.max]
        for k, f in enumerate(fr):
            fn = os.path.join(d, f'{k * a.step:05d}.jpg')
            if os.path.exists(fn):
                continue
            c.call('simSetVehiclePose', pose(*f['drone']), True, V)
            cx, cy, cz, cp, cyaw = f['cam']
            c.call('simSetCameraPose', 'follow', pose(cx, cy, cz, 0.0, cp, cyaw), '', True)
            time.sleep(a.settle)
            grab(c)                      # discard: may be the frame from before the move
            im = grab(c)
            cv2.imwrite(fn, im, [cv2.IMWRITE_JPEG_QUALITY, 93])
            n += 1
            if n % 200 == 0:
                print(f'{n} frames, {n / (time.time() - t0):.1f}/s, mission {mi} k {k}', flush=True)
        if a.max is None and a.step == 1:
            open(os.path.join(d, 'done'), 'w').write('ok')
        print(f'mission {mi} ({m["agent"]} s{m["scenario"]}) rendered {len(fr)}', flush=True)
    print('RENDER_DONE', n, 'frames', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
