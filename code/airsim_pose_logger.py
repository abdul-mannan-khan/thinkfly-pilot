#!/usr/bin/env python3
"""Logs the AirSim vehicle's true pose (simGetVehiclePose) with wall time, 4 Hz: pose.csv (wall, x, y, z NED).
  airsim_pose_logger.py OUT.csv [--vehicle PX4] [--stop FILE]"""
import sys
import time

from airsim_rpc import AirSimRPC

out, veh = sys.argv[1], 'PX4'
stop = sys.argv[sys.argv.index('--stop') + 1] if '--stop' in sys.argv else None
c = AirSimRPC()
f = open(out, 'a')
import os
while not (stop and os.path.exists(stop)):
    try:
        p = c.call('simGetVehiclePose', veh)['position']
        f.write(f"{time.time():.3f},{p['x_val']:.2f},{p['y_val']:.2f},{p['z_val']:.2f}\n")
        f.flush()
    except Exception as e:
        print('pose:', repr(e)[:120], flush=True)
        time.sleep(2)
    time.sleep(0.25)
