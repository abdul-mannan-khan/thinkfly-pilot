#!/usr/bin/env python3
"""ThinkFly hardware-in-the-loop run (runs ON the Jetson Orin Nano companion computer of the X500 bench).

  Pixhawk 6C (PX4 1.15.1, SIH mode: the real flight controller runs the real PX4 and computes the vehicle physics
  on the board) <-USB-> bridge.py <-UDP 14550-> this script (mission logic, event triggers, rule checker)
                                                   |-> small model Qwen3.5-4B on THIS Jetson (llama.cpp, CUDA),
                                                   |   energy = Jetson VDD_IN rail (INA3221), idle subtracted
                                                   '-> large model Qwen3.5-27B at the "ground station" (GPU server,
                                                       reached through the laptop), energy measured there per request
  Isaac Sim mirrors the board's pose in the urban scene (for the video); the scene map is the same urban.json.

Same Mission class, scenarios, prompts and checker as the simulation runs (thinkfly_mission.py). No PX4 parameter is
changed (shared bench; SIH needs none), so the board's own default speed limits apply.

  python3 thinkfly_hil.py --out OUT --scenarios 0 --agents thinkfly,event --pad X Y Z \
      --large-host 192.168.55.100 --large-port 8092
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thinkfly_mission as tm                      # noqa: E402
from thinkfly_mission import X500, Mission, make_scenario, parse_range   # noqa: E402
from citymap import CityMap                         # noqa: E402
from ai import JetsonMeter, Model                   # noqa: E402

_orig_nfz = Mission._nfz_centre


def _fixed_nfz(self, e):                            # paired scenarios: the no-fly zone is fixed per scenario
    return np.array(e['fixed'], float) if 'fixed' in e else _orig_nfz(self, e)


Mission._nfz_centre = _fixed_nfz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenarios', default='0')
    ap.add_argument('--agents', default='thinkfly,event')
    ap.add_argument('--pad', type=float, nargs=3, required=True, help='Isaac world position of the pad (from run.json)')
    ap.add_argument('--scene', default=os.path.join(HERE, 'urban.json'))
    ap.add_argument('--alt', type=float, default=30.0)
    ap.add_argument('--port', type=int, default=14550)
    ap.add_argument('--small-port', type=int, default=8081)
    ap.add_argument('--large-host', default='192.168.55.100')
    ap.add_argument('--large-port', type=int, default=8092)
    ap.add_argument('--scenario-dir', default=None, help='load scenario_<seed>.json from here (e.g. the paired AirSim run)')
    ap.add_argument('--label', default='HIL (Pixhawk 6C SIH + Jetson Orin Nano)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cmap = CityMap(a.scene, a.alt)
    meter = JetsonMeter()
    idle = meter.calibrate_idle(10.0, 3)
    print(f'Jetson VDD_IN idle (small model loaded) {idle:.2f} W', flush=True)
    small = Model('small', a.small_port, meter, thinking_flag=True)
    large = Model('large', a.large_port, meter, thinking_flag=True, host=a.large_host, remote_energy=True)
    json.dump({'mode': 'HIL: Pixhawk 6C PX4 1.15.1 SIH + Jetson Orin Nano Super (25 W mode) companion',
               'small_energy': 'Jetson VDD_IN (INA3221), idle subtracted', 'large_energy': 'ground-station GPU NVML via energy_proxy',
               'jetson_idle_W': idle, 'idle_windows_W': meter.idle_windows, 'alt': a.alt, 'reserve': tm.RESERVE,
               'pad': a.pad, 'params_changed': False, 'scene': os.path.basename(a.scene),
               'scenario_dir': a.scenario_dir, 'label': a.label}, open(os.path.join(a.out, 'setup.json'), 'w'), indent=1)
    d = X500(0, a.pad, a.port)
    while d.pos is None or d.t is None:
        d.poll()
        d.gcs_heartbeat()
        time.sleep(0.1)
    print('Pixhawk ready at', np.round(d.pos, 1), flush=True)
    for seed in parse_range(a.scenarios):
        cmap.reset_dynamic()
        if a.scenario_dir:
            sc = json.load(open(os.path.join(a.scenario_dir, f'scenario_{seed}.json')))
        else:
            sc = make_scenario(seed, cmap, np.array(a.pad[:2]), a.alt)
        json.dump(sc, open(os.path.join(a.out, f'scenario_{seed}.json'), 'w'), indent=1)
        for agent in a.agents.split(','):
            f = os.path.join(a.out, f'mission_{seed}_{agent}.json')
            if os.path.exists(f):
                continue
            cmap.reset_dynamic()
            print(f'--- HIL scenario {seed} agent {agent} (battery {sc["battery0"]} %)', flush=True)
            tw = time.time()
            while (d.armed or d.pos is None) and time.time() - tw < 120:
                d.poll(); d.gcs_heartbeat(); time.sleep(0.1)
            time.sleep(3.0)
            m = Mission(d, agent, sc, cmap, small, large)
            res = m.run()
            if res is None:
                print('    Pixhawk did not arm in 180 s', flush=True)
                sys.exit(3)
            res['mode'] = a.label
            res['t0_board'] = m.t0
            json.dump(res, open(f, 'w'))
            print(f'    {agent}: safe={res["safe_completion"]} inspected={res["inspected"]} skipped={list(res["skipped"])} '
                  f'calls small/large={res["n_small"]}/{res["n_large"]} E={res["energy_J"]:.0f} J '
                  f'(small {res["energy_small_J"]:.0f}, large {res["energy_large_J"]:.0f}) time={res["mission_time_s"]} s', flush=True)
    print('HIL_DONE', flush=True)


if __name__ == '__main__':
    main()
