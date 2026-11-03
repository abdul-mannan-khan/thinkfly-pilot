#!/usr/bin/env python3
"""ThinkFly pilot in a second simulator: Microsoft AirSim v1.8.1, 'AirSimNH' suburban neighbourhood, SimpleFlight
quadrotor. The mission code (agents, triggers, models, rule checker, energy meter, scenario generator) is the SAME
module as the Isaac Sim runs (thinkfly_mission.py); only the drone interface and the map differ:

- AirSimDrone: same methods as the PX4 drone (poll, setpoint, cmd, ...). Position setpoints become velocity commands
  (P-control, 8 m/s horizontal, 4 m/s vertical) to AirSim's built-in SimpleFlight controller. Physical collisions are
  read from AirSim every poll and reported.
- Map: house boxes read from the running scene (simListSceneObjects / simGetObjectPose); cruise altitude 35 m is above
  the houses and trees, so the A* map is open space except for no-fly zones. Targets are rooftops of houses.
- A chase camera records the flight (frames + sim timestamps) for the video.

  thinkfly_airsim.py --out OUT --scenarios 0-2 --agents thinkfly,event,periodic,step
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thinkfly_mission as tm                     # noqa: E402
from thinkfly_mission import Mission, make_scenario, OFFBOARD, parse_range   # noqa: E402
from citymap import CityMap                        # noqa: E402
from ai import PowerMeter, Model                   # noqa: E402
from airsim_rpc import AirSimRPC, YAW_FREE, DRIVE_MAX_DOF, image_request   # noqa: E402
from pymavlink import mavutil                      # noqa: E402

V = 'Drone1'


class AirSimDrone:
    """Drop-in for the PX4 drone used by Mission (world frame ENU, origin at the player start)."""

    def __init__(self, rpc, rpc_move):
        self.c, self.cm = rpc, rpc_move
        self.pad = np.zeros(3)
        self.pos = self.vel = self.t = None
        self.armed, self.mode, self.landed = False, None, None
        self.path, self.s = None, 0.0
        self.sp, self.last_cmd = None, 0.0
        self.collisions, self._last_coll = [], None
        self.landing = False
        self._still_since = None

    def poll(self):
        st = self.c.call('getMultirotorState', V)
        k = st['kinematics_estimated']
        p, v = k['position'], k['linear_velocity']
        self.pos = np.array([p['y_val'], p['x_val'], -p['z_val']])
        self.vel = np.array([v['y_val'], v['x_val'], -v['z_val']])
        self.t = st['timestamp'] * 1e-9
        self.landed = st['landed_state']           # 0 landed, 1 flying
        # touchdown: AirSim's landed_state is unreliable on this map, so also accept 'low and still for 2 s'
        still = self.pos[2] < 1.2 and np.linalg.norm(self.vel) < 0.3
        self._still_since = (self._still_since or time.time()) if still else None
        if self.landing and self.armed and (self.landed == 0 or (self._still_since and time.time() - self._still_since > 2.0)):
            self.c.call('armDisarm', False, V)
            self.armed, self.landing = False, False
        if self.armed and not self.landing:
            ci = self.c.call('simGetCollisionInfo', V)
            if ci.get('has_collided') and ci.get('time_stamp') != self._last_coll and self.pos[2] > 2.0:
                self._last_coll = ci.get('time_stamp')
                self.collisions.append({'t': self.t, 'object': ci.get('object_name')})
        if self.landing and self.armed and time.time() - self.last_cmd > 0.1:
            if self.pos[2] > 1.5:                      # fast controlled descent, then SimpleFlight's land for the last metre
                self.cm.post('moveByVelocity', 0.0, 0.0, 2.5, 0.5, DRIVE_MAX_DOF, YAW_FREE, V)
            elif not self._final:
                self.cm.post('land', 30.0, V)
                self._final = True
            self.last_cmd = time.time()
        if self.sp is not None and self.armed and not self.landing and time.time() - self.last_cmd > 0.1:
            e = self.sp - self.pos
            vh = 1.0 * e[:2]
            n = np.linalg.norm(vh)
            if n > 8.0:
                vh *= 8.0 / n
            vz = float(np.clip(1.0 * e[2], -3.0, 4.0))
            self.cm.post('moveByVelocity', float(vh[1]), float(vh[0]), float(-vz), 0.5, DRIVE_MAX_DOF, YAW_FREE, V)
            self.last_cmd = time.time()

    def gcs_heartbeat(self):
        pass

    def set_param(self, name, value):
        pass

    def cmd(self, c, *p):
        if c == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            self.mode = OFFBOARD
        elif c == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and p and p[0] == 1 and not self.armed:
            self.c.call('enableApiControl', True, V)
            self.c.call('armDisarm', True, V)
            self.c.call('takeoff', 20.0, V)          # SimpleFlight lift-off to ~3 m, then velocity control
            self.armed, self.landing = True, False
        elif c == mavutil.mavlink.MAV_CMD_NAV_LAND and not self.landing:
            self.sp = None
            self.landing, self._final = True, False

    def setpoint(self, world):
        self.sp = np.asarray(world, float)

    def carrot(self, lead):
        from px4_offboard import Drone
        return Drone.carrot(self, lead)

    def reset(self):
        """Back to the start pose between missions (same start for every agent)."""
        self.c.call('reset')
        time.sleep(2.0)
        self.c.call('enableApiControl', True, V)
        self.armed, self.landing, self.sp, self.mode = False, False, None, None
        self.collisions, self._last_coll = [], None
        self.poll()


def scene_from_airsim(rpc, path):
    """House boxes from the running AirSimNH scene -> scene json in the Isaac format (centre, size, height, extent)."""
    names = rpc.call('simListSceneObjects', '.*')
    houses = []
    for n in names:
        ln = n.lower()
        if ln.startswith('small_house'):
            pz = rpc.call('simGetObjectPose', n)['position']
            if any(math.isnan(pz[k]) for k in ('x_val', 'y_val')):
                continue
            x, y = pz['y_val'], pz['x_val']                  # NED -> ENU
            if all(math.hypot(x - h['centre'][0], y - h['centre'][1]) > 12 for h in houses):
                houses.append({'name': n, 'centre': [round(x, 1), round(y, 1)], 'size': [14.0, 14.0], 'height': 10.0})
    xs = [h['centre'][0] for h in houses] + [0.0]
    ys = [h['centre'][1] for h in houses] + [0.0]
    L = round(max(max(abs(v) for v in xs), max(abs(v) for v in ys)) + 70.0)
    ext = [-L, L]                                       # square, as the Isaac scene (geofence = extent - 30 m)
    json.dump({'buildings': houses, 'extent': ext, 'source': 'AirSimNH via simListSceneObjects; house size/height assumed 14x14x10 m'},
              open(path, 'w'), indent=1)
    return houses, ext


class Recorder(threading.Thread):
    """Chase-camera frames at ~fps with the sim timestamp (frames/NNNNNN.png + frames.csv)."""

    def __init__(self, out, fps=4.0, cam='chase'):
        super().__init__(daemon=True)
        self.c = AirSimRPC()
        self.out, self.dt, self.cam = out, 1.0 / fps, cam
        os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
        self.f = open(os.path.join(out, 'frames.csv'), 'a')
        self.k = len(os.listdir(os.path.join(out, 'frames')))
        self.on = True

    def run(self):
        while self.on:
            t0 = time.time()
            try:
                t = self.c.call('getMultirotorState', V)['timestamp'] * 1e-9
                r = self.c.call('simGetImages', [image_request(self.cam)], V, False)[0]
                data = r['image_data_uint8']
                if data:
                    import cv2
                    fn = f'{self.k:06d}.jpg'
                    im = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
                    cv2.imwrite(os.path.join(self.out, 'frames', fn), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.f.write(f'{t:.3f},{fn}\n')
                    self.f.flush()
                    self.k += 1
            except Exception as e:                       # keep recording through transient errors
                print('recorder:', repr(e)[:120], flush=True)
                time.sleep(1.0)
            time.sleep(max(0.0, self.dt - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenarios', default='0-2')
    ap.add_argument('--agents', default='thinkfly,event,periodic,step')
    ap.add_argument('--alt', type=float, default=35.0)
    ap.add_argument('--small-port', type=int, default=8081)
    ap.add_argument('--large-port', type=int, default=8082)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--dmin', type=float, default=110.0)
    ap.add_argument('--dmax', type=float, default=300.0)
    ap.add_argument('--spacing', type=float, default=90.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rpc, rpc_move = AirSimRPC(), AirSimRPC().start_drain()
    print('AirSim ping:', rpc.call('ping'), flush=True)
    scene = os.path.join(a.out, 'airsimnh_scene.json')
    houses, ext = scene_from_airsim(rpc, scene)
    print(f'{len(houses)} houses, extent {ext}', flush=True)
    cmap = CityMap(scene, a.alt)
    meter = PowerMeter(gpu=a.gpu)
    idle = meter.calibrate_idle(10.0, 3)
    print(f'reasoning GPU {a.gpu} idle power {idle:.1f} W', flush=True)
    small = Model('small', a.small_port, meter, thinking_flag=True)
    large = Model('large', a.large_port, meter, thinking_flag=True)
    json.dump({'simulator': 'AirSim 1.8.1 AirSimNH (SimpleFlight)', 'idle_W': idle, 'idle_windows_W': meter.idle_windows,
               'gpu': a.gpu, 'alt': a.alt, 'reserve': tm.RESERVE, 'step_s': tm.STEP_S, 'period_s': tm.PERIOD_S},
              open(os.path.join(a.out, 'setup.json'), 'w'), indent=1)
    d = AirSimDrone(rpc, rpc_move)
    d.reset()
    rec = None
    if a.record:
        rec = Recorder(a.out)
        rec.start()
    for seed in parse_range(a.scenarios):
        cmap.reset_dynamic()
        sc = make_scenario(seed, cmap, np.zeros(2), a.alt, a.dmin, a.dmax, a.spacing)
        if len(sc['targets']) < 4:
            print(f'scenario {seed}: only {len(sc["targets"])} targets found; skipped', flush=True)
            continue
        json.dump(sc, open(os.path.join(a.out, f'scenario_{seed}.json'), 'w'), indent=1)
        for agent in a.agents.split(','):
            f = os.path.join(a.out, f'mission_{seed}_{agent}.json')
            if os.path.exists(f):
                continue
            cmap.reset_dynamic()
            d.reset()
            print(f'--- AirSim scenario {seed} agent {agent} (battery {sc["battery0"]} %)', flush=True)
            m = Mission(d, agent, sc, cmap, small, large)

            def mon(m=m):
                while not getattr(m, '_done', False):
                    time.sleep(10)
                    try:
                        print(f'   [mon] t={m.now():.0f} pos={np.round(d.pos, 1)} leg_target={m.leg_target} order={m.order} '
                              f'remaining={m.remaining} pending={m.pending is not None} inspect={m.inspect_until} '
                              f'land={d.landing} trig={[x["trigger"] for x in m.triggers][-3:]} bat={m.battery:.1f}', flush=True)
                    except Exception as e:
                        print('   [mon]', repr(e)[:80], flush=True)
            threading.Thread(target=mon, daemon=True).start()
            res = m.run()
            m._done = True
            if res is None:
                print('    did not arm', flush=True)
                continue
            res['simulator'] = 'AirSim AirSimNH'
            res['t0_sim'] = m.t0 if m.t0 is not None else None
            res['airsim_collisions'] = d.collisions
            if d.collisions:
                res['safe_completion'] = False
                res['n_violations'] = res['n_violations'] + len(d.collisions)
            json.dump(res, open(f, 'w'))
            print(f'    {agent}: safe={res["safe_completion"]} inspected={res["inspected"]} skipped={list(res["skipped"])} '
                  f'collisions={len(d.collisions)} calls small/large={res["n_small"]}/{res["n_large"]} '
                  f'E={res["energy_J"]:.0f} J time={res["mission_time_s"]} s', flush=True)
    if rec:
        rec.on = False
        time.sleep(1.0)
    print('AIRSIM_DONE', flush=True)


if __name__ == '__main__':
    main()
