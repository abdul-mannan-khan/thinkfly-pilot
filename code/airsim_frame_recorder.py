#!/usr/bin/env python3
"""Chase-camera recorder for the AirSim PX4-HITL bench: frames + AirSim timestamp + vehicle NED position.
  airsim_frame_recorder.py OUT [--fps 4] [--vehicle PX4] [--stop FILE]"""
import argparse
import os
import time

import cv2
import numpy as np

from airsim_rpc import AirSimRPC, image_request

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--fps', type=float, default=4.0)
ap.add_argument('--vehicle', default='PX4')
ap.add_argument('--stop', default=None)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
c = None
while c is None:
    try:
        c = AirSimRPC()
    except OSError:
        time.sleep(3)
f = open(os.path.join(a.out, 'frames.csv'), 'a')
k = len(os.listdir(os.path.join(a.out, 'frames')))
last = 0
while not (a.stop and os.path.exists(a.stop)):
    t0 = time.time()
    try:
        s = c.call('getMultirotorState', a.vehicle)
        p = s['kinematics_estimated']['position']
        r = c.call('simGetImages', [image_request('chase')], a.vehicle, False)[0]
        if r['image_data_uint8']:
            im = cv2.imdecode(np.frombuffer(bytes(r['image_data_uint8']), np.uint8), cv2.IMREAD_COLOR)
            fn = f'{k:06d}.jpg'
            cv2.imwrite(os.path.join(a.out, 'frames', fn), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            f.write(f"{s['timestamp'] * 1e-9:.3f},{fn},{p['x_val']:.2f},{p['y_val']:.2f},{p['z_val']:.2f},{time.time():.3f}\n")
            f.flush()
            k += 1
    except Exception as e:
        print('recorder:', repr(e)[:150], flush=True)
        time.sleep(2)
    if time.time() - last > 60:
        print(time.strftime('%T'), 'frames', k, flush=True)
        last = time.time()
    time.sleep(max(0.0, 1.0 / a.fps - (time.time() - t0)))
