"""Offline dry run of the mission loop: kinematic stand-in drone and stand-in models (no PX4, no Isaac, no LLM).
Only checks that the code runs end to end and the bookkeeping is sensible. Not a result."""
import json
import os
import random
import sys
import types

import numpy as np

sys.path.insert(0, '.')
pm = types.ModuleType('pymavlink')
mv = types.ModuleType('mavutil')
mv.mavlink = types.SimpleNamespace(MAV_CMD_DO_SET_MODE=176, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1,
                                   MAV_CMD_COMPONENT_ARM_DISARM=400, MAV_CMD_NAV_LAND=21,
                                   MAV_MODE_FLAG_SAFETY_ARMED=128)
pm.mavutil = mv
sys.modules['pymavlink'] = pm
sys.modules['pymavlink.mavutil'] = mv
import thinkfly_mission as tm  # noqa: E402
from citymap import CityMap  # noqa: E402
import ai  # noqa: E402

tm.time.sleep = lambda s: None


class MockDrone:
    carrot = tm.Drone.carrot

    def __init__(self):
        self.pad = np.array([0.0, 0.0, 0.15])
        self.pos, self.t, self.armed, self.mode = self.pad.copy(), 0.0, False, 0
        self.sp, self.landing, self.path, self.s = self.pad.copy(), False, None, 0.0

    def poll(self):
        dt = 0.1
        self.t += dt
        if self.armed:
            if self.landing:
                self.pos[2] = max(self.pos[2] - 3 * dt, self.pad[2])
                if self.pos[2] <= self.pad[2] + 0.01:
                    self.armed, self.landing = False, False
            else:
                v = self.sp - self.pos
                h = np.linalg.norm(v[:2])
                if h > 1e-6:
                    self.pos[:2] += v[:2] / h * min(h, 8 * dt)
                self.pos[2] += np.clip(v[2], -3 * dt, 4 * dt)

    def gcs_heartbeat(self):
        pass

    def cmd(self, c, *p):
        if c == 400:
            self.armed, self.mode = True, tm.OFFBOARD
        if c == 21:
            self.landing = True

    def setpoint(self, w):
        self.sp = np.array(w, float)


class MockModel:
    def __init__(self, name, err=0.0):
        self.name, self.err = name, err

    def ask(self, system, user, sch, max_tokens=320):
        s = user.split('Current state:\n')[-1]
        st = json.loads(s.split('\nALARM')[0])
        ids = [t['id'] for t in st['remaining_targets'] if not t['inside_no_fly_zone']]
        bad = [t['id'] for t in st['remaining_targets'] if t['inside_no_fly_zone']]
        plan = {'order': ids, 'skip': [{'id': i, 'reason': 'no-fly zone'} for i in bad], 'return_home_now': False,
                'note': 'mock'}
        if self.name == 'large':                       # oracle on the cost table: most targets that fit the battery
            import itertools
            c = st['battery_cost_pct']
            def cost(a, b):
                k = [f'{a}->{b}', f'{b}<->{a}', f'{a}<->{b}']
                for x in k:
                    if c.get(x) is not None:
                        return c[x]
                return 1e9
            best = ([], 0)
            for n in range(len(ids), 0, -1):
                for perm in itertools.permutations(ids, n):
                    seq = ['drone'] + list(perm) + ['home']
                    tot = sum(cost(seq[k], seq[k + 1]) for k in range(len(seq) - 1)) + n * st['inspection_cost_pct_each']
                    if st['battery_pct'] - tot >= st['reserve_pct'] + 1.5:
                        best = (list(perm), n)
                        break
                if best[1]:
                    break
            plan['order'] = best[0]
            plan['skip'] += [{'id': i, 'reason': 'battery'} for i in ids if i not in best[0]]
        elif random.random() < self.err:
            plan['order'] = ids[::-1] + bad
        return {'model': self.name, 'text': json.dumps(plan), 'plan': plan, 'tokens_in': len(user) // 4,
                'tokens_out': 40, 'latency_s': 0.1, 'energy_gross_J': 1.0, 'energy_J': 0.5, 't0': 0, 't1': 0,
                'error': None}


random.seed(0)
cmap = CityMap(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scene', 'urban_isaac.json'), 30.0)
d = MockDrone()
for seed in (0, 1):
    cmap.reset_dynamic()
    sc = tm.make_scenario(seed, cmap, d.pad[:2], 30.0)
    for agent in ('thinkfly', 'event', 'periodic', 'step'):
        cmap.reset_dynamic()
        d.pos, d.armed = d.pad.copy(), False
        r = tm.Mission(d, agent, sc, cmap, MockModel('small', 0.3), MockModel('large')).run()
        json.dumps(r)
        print(seed, agent, 'safe', r['safe_completion'], 'insp', r['inspected'], 'skip', list(r['skipped']),
              'viol', r['violations'], 'calls s/l', r['n_small'], r['n_large'], 'esc', r['n_escalations'],
              'trig', [x['trigger'] for x in r['triggers']][:8], 'bat', r['battery_end'], 'T', r['mission_time_s'])
