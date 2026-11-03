#!/usr/bin/env python3
"""ThinkFly pilot: city inspection missions flown by a PX4 SITL X500 in the Isaac Sim urban scene, with four agents.

  step      - step-by-step agent: the large model is called every STEP_S seconds with the growing step history
  periodic  - the large model re-plans every PERIOD_S seconds (compact state)
  event     - event-triggered reasoning, large model only (compact state)
  thinkfly  - event-triggered reasoning; the small onboard model proposes, the rule checker verifies, the large
              model is called only on an alarm (failed check)

Every agent flies the same scenarios (same seed => same targets, battery, events). The flight controller (A* on
the city map + PX4 position control) is the same for all; it flies the current plan's legs and holds position while
the agent is reasoning. Triggers (event, thinkfly): novelty (a new message), rule margin (predicted battery at home
below reserve + 3 %), progress stall (remaining route length not falling for 15 s while cruising).

Run inside the instance while run_sim.py runs:
  $SCENEGEN_PYTHON thinkfly_mission.py --run-dir RUN --scenarios 0-5 --agents thinkfly,event,periodic,step --out OUT
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import sys
import time

import numpy as np
from pymavlink import mavutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from px4_offboard import Drone, OFFBOARD           # noqa: E402
from citymap import CityMap, leg_cost, INSPECT_S, K_TIME, K_DIST  # noqa: E402
from ai import PowerMeter, Model, SYSTEM, schema, state_text, check_plan  # noqa: E402

STEP_S, PERIOD_S = 8.0, 30.0
RESERVE = 20.0
IDS = 'ABCDE'


class X500(Drone):
    """Drone from the base project + PX4 boot time (sim clock) from LOCAL_POSITION_NED."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.t = None

    def poll(self):
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.pos = self.pad + np.array([msg.y, msg.x, -msg.z])
                self.vel = np.array([msg.vy, msg.vx, -msg.vz])
                self.t = msg.time_boot_ms * 1e-3
            elif t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = (msg.custom_mode >> 16) & 0xFF
            elif t == 'EXTENDED_SYS_STATE':
                self.landed = msg.landed_state


# ---------------------------------------------------------------------------------------------- scenarios
def make_scenario(seed, cmap, home, alt, dmin=110.0, dmax=300.0, spacing=90.0):
    rng = random.Random(1000 + seed)
    cand = []
    for b in cmap.buildings:
        (cx, cy) = b['centre']
        d = math.hypot(cx - home[0], cy - home[1])
        if b['height'] < alt - 8.0 and dmin <= d <= dmax and cmap.reachable((cx, cy)):
            cand.append((cx, cy))
    rng.shuffle(cand)
    pick = []
    for c in cand:
        if all(math.hypot(c[0] - p[0], c[1] - p[1]) >= spacing for p in pick) and cmap.dist(home, c) is not None:
            pick.append(c)
        if len(pick) == 5:
            break
    tg = {IDS[k]: [round(p[0], 1), round(p[1], 1)] for k, p in enumerate(pick)}
    base = {k: tg[k] for k in 'ABCD'}
    # battery: enough for the best tour of A-D plus a margin, so the original mission fits
    import itertools
    best = 1e9
    for perm in itertools.permutations('ABCD'):
        pts, legs = cmap.route(home, [base[i] for i in perm], home)
        if pts is not None:
            best = min(best, sum(leg_cost(L) for L in legs) + 4 * leg_cost(0, INSPECT_S))
    battery0 = round(RESERVE + best + rng.uniform(9.0, 13.0) + 2.0, 1)   # + ~2 % for climb and landing
    ev = [{'t': round(rng.uniform(25, 45), 1), 'kind': 'nfz', 'mode': rng.choice(['route', 'target']),
           'r': round(rng.uniform(50, 70), 1)},
          {'t': round(rng.uniform(60, 85), 1), 'kind': 'battery', 'drop': round(rng.uniform(9.0, 14.0), 1)},
          {'t': round(rng.uniform(95, 120), 1), 'kind': 'task', 'id': 'E', 'xy': tg['E']}]
    return {'seed': seed, 'home': [round(home[0], 1), round(home[1], 1)], 'alt': alt, 'targets': base,
            'battery0': battery0, 'reserve': RESERVE, 'events': ev, 'best_tour_pct': round(best, 2)}


# ---------------------------------------------------------------------------------------------- one mission
class Mission:
    def __init__(self, d, agent, sc, cmap, small, large, log_every=0.5):
        self.d, self.agent, self.sc, self.cmap, self.small, self.large = d, agent, sc, cmap, small, large
        self.home = np.array(sc['home'], float)
        self.z = sc['alt']
        self.tg = {k: np.array(v, float) for k, v in sc['targets'].items()}
        self.remaining = list(sc['targets'])
        self.inspected, self.skipped = [], {}
        self.battery, self.reserve = sc['battery0'], sc['reserve']
        self.events = [dict(e, fired=False) for e in sc['events']]
        self.messages, self.new_msgs, self.history = [], [], []
        self.order = []                        # plan: targets still to inspect, in order
        self.leg, self.leg_target, self.s = None, None, 0.0
        self.inspect_until = None
        self.calls, self.triggers, self.track, self.notes = [], [], [], []
        self.viol = {'nfz_s': 0.0, 'collision': 0, 'fence': 0, 'reserve': 0, 'empty': 0}
        self.min_battery = self.battery
        self.log_every, self.last_log = log_every, -1e9
        self.pool = cf.ThreadPoolExecutor(1)
        self.pending = None
        self.last_reason_t, self.next_due = -1e9, 0.0
        self.prog = []                         # (t, remaining route length)
        self.margin_flag = False
        self.t0 = None

    # -- helpers
    def now(self):
        return (self.d.t - self.t0) if self.t0 is not None else 0.0

    def setp(self, xy, z=None):
        self.d.setpoint([xy[0], xy[1], self.z if z is None else z])

    def start_leg(self):
        """Path to the next target (or home) on the current map."""
        p = self.d.pos[:2]
        while self.order:
            nxt = self.order[0]
            path = self.cmap.path(p, self.tg[nxt])
            if path is not None:
                self.leg, self.leg_target, self.s = np.array([[x, y, self.z] for x, y in path]), nxt, 0.0
                return
            self.notes.append(f'{self.now():.0f}s controller: no safe path to {nxt}, dropped')
            self.skipped[nxt] = 'no safe path (controller)'
            self.order.pop(0)
            if nxt in self.remaining:
                self.remaining.remove(nxt)
        path = self.cmap.path(p, self.home) or [p, self.home]
        self.leg, self.leg_target, self.s = np.array([[x, y, self.z] for x, y in path]), 'home', 0.0

    def remaining_route(self):
        """Length (m) of the rest of the current leg plus straight-line legs of the plan (cheap progress function)."""
        if self.leg is None:
            return 0.0
        seg = np.linalg.norm(np.diff(self.leg[:, :2], axis=0), axis=1)
        L = max(seg.sum() - self.s, 0.0)
        pts = [self.tg[i] for i in self.order[1:]] + [self.home]
        last = self.leg[-1, :2]
        for q in pts:
            L += float(np.linalg.norm(q - last))
            last = q
        return L

    def predicted_home_battery(self):
        """Battery at home if the current plan is flown: rest of the current leg + A* lengths of the later legs."""
        if self.leg is None:
            return self.battery
        seg = np.linalg.norm(np.diff(self.leg[:, :2], axis=0), axis=1)
        L = max(seg.sum() - self.s, 0.0)
        last = self.leg[-1, :2]
        for q in [self.tg[i] for i in self.order[1:]] + [self.home]:
            dq = self.cmap.dist(last, q)
            L += dq if dq is not None else 1e4
            last = q
        return self.battery - leg_cost(L) - len(self.order) * leg_cost(0, INSPECT_S)

    def state(self, event):
        p = self.d.pos[:2]
        ids = list(self.remaining)
        targets, costs = [], {}
        for i in ids:
            inz = bool(self.cmap.in_nfz(self.tg[i]))
            targets.append({'id': i, 'pos': [round(float(self.tg[i][0])), round(float(self.tg[i][1]))],
                            'inside_no_fly_zone': inz})
        def c(a, b):
            L = self.cmap.dist(a, b)
            return None if L is None else round(leg_cost(L), 2)
        for i in ids:
            costs[f'drone->{i}'] = c(p, self.tg[i])
            costs[f'{i}->home'] = c(self.tg[i], self.home)
        for a_ in range(len(ids)):
            for b_ in range(a_ + 1, len(ids)):
                costs[f'{ids[a_]}<->{ids[b_]}'] = c(self.tg[ids[a_]], self.tg[ids[b_]])
        costs['drone->home'] = c(p, self.home)
        nfz = [{'name': n, 'centre': [round(cx), round(cy)], 'radius_m': round(r)} for cx, cy, r, n in self.cmap.nfz]
        return {'t': self.now(), 'battery': self.battery, 'reserve': self.reserve, 'targets': targets,
                'costs': costs, 'nfz': nfz, 'event': event, 'messages': self.messages[-4:], 'ids': ids}

    # -- reasoning (runs in a worker thread; the main loop keeps streaming a hold setpoint)
    def reason(self, trigger):
        st = self.state(trigger)
        pos, home = self.d.pos[:2].copy(), self.home.copy()
        sch = schema(st['ids'])
        out = {'trigger': trigger, 't': round(st['t'], 1), 'battery': round(self.battery, 2), 'calls': []}
        if self.agent == 'thinkfly':
            r = self.small.ask(SYSTEM, state_text(st), sch)
            ok, why, _, bat = check_plan(r['plan'], st, self.cmap, pos, home, self.tg)
            r.update(check=ok, check_reason=why)
            out['calls'].append(r)
            if ok:
                out.update(plan=r['plan'], by='small')
                return out
            alarm = state_text(st) + f'\nALARM: the onboard model\'s plan failed the rule check: {why}.'
            r2 = self.large.ask(SYSTEM, alarm, sch)
            ok2, why2, _, _ = check_plan(r2['plan'], st, self.cmap, pos, home, self.tg, strict=False)   # safety rules
            r2.update(check=ok2, check_reason=why2, escalated=True)
            out['calls'].append(r2)
            if ok2:
                out.update(plan=r2['plan'], by='large')
            else:                               # large plan also unsafe: safe fallback, fly home
                out.update(plan={'order': [], 'skip': [], 'return_home_now': True,
                                 'note': 'fallback: return home'}, by='fallback')
            return out
        hist = self.history if self.agent == 'step' else None
        r = self.large.ask(SYSTEM, state_text(st, hist), sch)
        ok, why, _, _ = check_plan(r['plan'], st, self.cmap, pos, home, self.tg)   # recorded only, not enforced
        r.update(check=ok, check_reason=why)
        out['calls'].append(r)
        out.update(plan=r['plan'], by='large')
        return out

    def adopt(self, res):
        self.calls.append(res)
        plan = res.get('plan')
        if plan is None:                        # unparseable answer: keep the current plan
            return
        if plan.get('return_home_now'):
            order = []
        else:
            order = []
            for i in plan.get('order', []):
                if i in self.remaining and i not in order:
                    order.append(i)
        for s in plan.get('skip', []):
            if s.get('id') in self.remaining and s.get('id') not in order:
                self.skipped[s['id']] = s.get('reason', '')
                self.remaining.remove(s['id'])
        # targets neither ordered nor skipped stay "remaining" for the next reasoning step, but are not flown now
        self.order = order
        self.start_leg()
        if self.agent == 'step':
            self.history.append(f't={res["t"]:.0f}s battery={res["battery"]:.1f}% event={res["trigger"]} '
                                f'-> order={order} skip={[s.get("id") for s in plan.get("skip", [])]} '
                                f'home_now={plan.get("return_home_now")}')

    # -- events
    def fire_events(self):
        t = self.now()
        for e in self.events:
            if e['fired'] or t < e['t']:
                continue
            e['fired'] = True
            if e['kind'] == 'nfz':
                c = self._nfz_centre(e)
                name = 'NFZ-1'
                self.cmap.add_nfz(c[0], c[1], e['r'], name)
                e['centre'] = [round(float(c[0]), 1), round(float(c[1]), 1)]
                msg = (f'ATC: temporary no-fly zone {name} active now, centre ({c[0]:.0f}, {c[1]:.0f}), radius '
                       f'{e["r"]:.0f} m (police helicopter operation).')
            elif e['kind'] == 'battery':
                self.battery -= e['drop']
                msg = f'Battery monitor: sudden drop of {e["drop"]:.1f} % (cell imbalance). Battery now {self.battery:.1f} %.'
            elif e['kind'] == 'task':
                self.tg[e['id']] = np.array(e['xy'], float)
                self.remaining.append(e['id'])
                msg = f'Operator: please also inspect target {e["id"]} at ({e["xy"][0]:.0f}, {e["xy"][1]:.0f}).'
            self.messages.append(msg)
            self.new_msgs.append(msg)
            self.notes.append(f'{t:.0f}s {msg}')
            e['t_fired'] = round(t, 1)

    def _nfz_centre(self, e):
        p = self.d.pos[:2]
        safe = lambda c: (math.hypot(c[0] - p[0], c[1] - p[1]) > e['r'] + 25 and
                          math.hypot(c[0] - self.home[0], c[1] - self.home[1]) > e['r'] + 25)
        if e['mode'] == 'target':
            for i in self.order[1:] + self.order[:1]:
                c = self.tg[i] + np.array([8.0, -6.0])
                if safe(c):
                    return c
        # on the current route, 90-140 m ahead
        pts = [p] + ([self.leg[-1, :2]] if self.leg is not None else []) + [self.tg[i] for i in self.order[1:]] + [self.home]
        for ahead in (110.0, 140.0, 90.0, 170.0):
            acc, last = 0.0, pts[0]
            for q in pts[1:]:
                seg = float(np.linalg.norm(q - last))
                if acc + seg >= ahead:
                    c = last + (q - last) * (ahead - acc) / max(seg, 1e-6)
                    if safe(c):
                        return c
                    break
                acc, last = acc + seg, q
        return self.home + np.array([200.0, 200.0])

    # -- triggers
    def due(self):
        t = self.now()
        if self.pending is not None or self.inspect_until is not None:
            return None
        if self.agent == 'step':
            return 'step' if t >= self.next_due else None
        if self.agent == 'periodic':
            if t >= self.next_due:
                if self.new_msgs:
                    self.new_msgs = []
                return 'period'
            return None
        if t - self.last_reason_t < 4.0:
            return None
        if self.new_msgs:
            self.new_msgs = []
            return 'novelty'
        if self.leg_target != 'home' or self.order:
            margin = self.predicted_home_battery() < self.reserve + 3.0
            if margin and not self.margin_flag:
                self.margin_flag = True
                return 'rule_margin'
            if not margin:
                self.margin_flag = False
        if len(self.prog) > 2:
            old = [L for tt, L in self.prog if t - tt >= 15.0]
            if old and old[-1] - self.prog[-1][1] < 15.0 and t - self.last_reason_t > 15.0 and self.leg_target:
                self.prog = []
                return 'progress_stall'
        return None

    # -- main loop
    def run(self, timeout_s=900.0):
        d = self.d
        d.path, d.s = None, 0.0
        t_wall0, last_hb, last_cmd = time.time(), 0.0, 0.0
        phase = 'arm'
        while True:
            d.poll()
            nw = time.time()
            if nw - last_hb > 0.5:
                d.gcs_heartbeat()
                last_hb = nw
            if d.pos is None or d.t is None:
                time.sleep(0.02)
                continue
            if phase == 'arm' and nw - t_wall0 > 180:     # simulator stalled or PX4 refused to arm
                self.pool.shutdown(wait=False)
                return None
            if phase == 'arm':
                d.setpoint([d.pos[0], d.pos[1], d.pad[2]])
                if nw - last_cmd > 2.0:
                    d.cmd(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, OFFBOARD)
                    d.cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                    last_cmd = nw
                if d.armed and d.mode == OFFBOARD:
                    phase, self.t0 = 'climb', d.t
                    self.t_takeoff = d.t
            elif phase == 'climb':
                self.setp(self.home)
                if d.pos[2] > self.z - 1.5:
                    phase = 'plan0'
                    self.t0 = d.t                   # mission clock starts at cruise altitude
                    self.prev = d.pos.copy()
                    self.next_due = 0.0
            elif phase in ('plan0', 'fly'):
                dt = max(d.t - getattr(self, 'tprev', d.t), 0.0)
                self.tprev = d.t
                self.battery -= K_TIME * dt + K_DIST * float(np.linalg.norm(d.pos[:2] - self.prev[:2]))
                self.prev = d.pos.copy()
                self.min_battery = min(self.min_battery, self.battery)
                self.fire_events()
                self.score()
                if self.pending is not None:        # reasoning in progress: hold position
                    self.setp(self.hold)
                    if self.pending.done():
                        res = self.pending.result()
                        self.pending = None
                        self.last_reason_t = self.now()
                        if self.agent == 'step':
                            self.next_due = self.now() + STEP_S
                        elif self.agent == 'periodic':
                            self.next_due = self.now() + PERIOD_S
                        self.adopt(res)
                        phase = 'fly'
                else:
                    trig = 'mission_start' if phase == 'plan0' else self.due()
                    if trig:
                        self.hold = d.pos[:2].copy()
                        self.triggers.append({'t': round(self.now(), 1), 'trigger': trig})
                        self.pending = self.pool.submit(self.reason, trig)
                        self.setp(self.hold)
                    elif phase == 'fly':
                        self.fly_step()
                if self.battery <= 0.0:
                    self.viol['empty'] = 1
                    self.notes.append(f'{self.now():.0f}s battery empty')
                    phase = 'land'
                if phase == 'fly' and self.leg_target == 'home' and self.leg is not None and \
                        np.linalg.norm(d.pos[:2] - self.home) < 2.0 and self.pending is None:
                    phase = 'land'
                    self.t_land = self.now()
            elif phase == 'land':
                if nw - last_cmd > 2.0:
                    d.cmd(mavutil.mavlink.MAV_CMD_NAV_LAND)
                    last_cmd = nw
                if not d.armed:
                    break
            if self.t0 is not None and d.t - self.t0 > timeout_s and phase != 'land':
                self.notes.append('timeout')
                phase = 'land'
            if time.time() - t_wall0 > 3 * timeout_s + 600:
                self.notes.append('wall timeout')
                break
            if self.t0 is not None and d.t - self.last_log >= self.log_every:
                self.last_log = d.t
                self.track.append([round(d.t - self.t0, 2), *[round(float(v), 2) for v in d.pos],
                                   round(self.battery, 2), 1 if self.pending is not None else 0])
            time.sleep(0.02)
        self.pool.shutdown(wait=False)
        return self.result()

    def fly_step(self):
        d = self.d
        if self.inspect_until is not None:
            self.setp(self.tg[self.leg_target] if self.leg_target in self.tg else d.pos[:2])
            if self.now() >= self.inspect_until:
                i = self.leg_target
                self.inspected.append(i)
                if i in self.remaining:
                    self.remaining.remove(i)
                if self.order and self.order[0] == i:
                    self.order.pop(0)
                self.inspect_until = None
                self.start_leg()
            return
        if self.leg is None:
            self.start_leg()
        d.path, d.s = self.leg, self.s
        c, done = d.carrot(10.0)
        self.s = d.s
        d.setpoint(c)
        self.prog.append((self.now(), self.remaining_route()))
        self.prog = [x for x in self.prog if self.now() - x[0] < 30.0]
        if done or np.linalg.norm(d.pos[:2] - self.leg[-1, :2]) < 2.5:
            if self.leg_target in self.tg and self.leg_target in self.remaining:
                self.inspect_until = self.now() + INSPECT_S
            elif self.leg_target != 'home':
                self.start_leg()

    def score(self):
        p = self.d.pos
        if self.cmap.in_nfz(p):
            self.viol['nfz_s'] = round(self.viol['nfz_s'] + max(self.d.t - getattr(self, 'tnfz', self.d.t), 0.0), 2)
        self.tnfz = self.d.t
        if p[2] > 3.0 and self.cmap.collides(p, margin=0.5):
            self.viol['collision'] = 1
        if not self.cmap.in_fence(p):
            self.viol['fence'] = 1

    def result(self):
        self.viol['reserve'] = int(self.battery < self.reserve)
        calls = [c for r in self.calls for c in r['calls']]
        small = [c for c in calls if c['model'] == 'small']
        large = [c for c in calls if c['model'] == 'large']
        violations = int(self.viol['nfz_s'] > 0) + self.viol['collision'] + self.viol['fence'] + self.viol['reserve'] + self.viol['empty']
        available = [i for i in self.tg if not any(math.hypot(self.tg[i][0] - cx, self.tg[i][1] - cy) <= r
                                                   for cx, cy, r, _ in self.cmap.nfz)]
        return {'agent': self.agent, 'seed': self.sc['seed'], 'landed_home': bool(np.linalg.norm(self.d.pos[:2] - self.home) < 5.0),
                'battery_end': round(self.battery, 2), 'min_battery': round(self.min_battery, 2),
                'inspected': self.inspected, 'skipped': self.skipped, 'available_targets': available,
                'violations': self.viol, 'n_violations': violations,
                'safe_completion': bool(violations == 0 and np.linalg.norm(self.d.pos[:2] - self.home) < 5.0),
                'mission_time_s': round(getattr(self, 't_land', self.now()), 1),
                'n_reasoning': len(self.calls), 'n_small': len(small), 'n_large': len(large),
                'n_escalations': sum(1 for c in large if c.get('escalated')),
                'energy_J': round(sum(c['energy_J'] for c in calls), 1),
                'energy_gross_J': round(sum(c['energy_gross_J'] for c in calls), 1),
                'energy_small_J': round(sum(c['energy_J'] for c in small), 1),
                'energy_large_J': round(sum(c['energy_J'] for c in large), 1),
                'tokens_in': sum(c['tokens_in'] for c in calls), 'tokens_out': sum(c['tokens_out'] for c in calls),
                'reasoning_time_s': round(sum(c['latency_s'] for c in calls), 1),
                'small_check_pass': [c['check'] for c in small],
                'events': self.events, 'notes': self.notes, 'triggers': self.triggers,
                'calls': [{k: v for k, v in r.items() if k != 'calls'} | {'calls': [
                    {k: v for k, v in c.items() if k not in ('t0', 't1')} | {'w0': c['t0'], 'w1': c['t1']} for c in r['calls']]}
                    for r in self.calls],
                'nfz': self.cmap.nfz, 'track': self.track}


def parse_range(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenarios', default='0-5')
    ap.add_argument('--agents', default='thinkfly,event,periodic,step')
    ap.add_argument('--alt', type=float, default=30.0)
    ap.add_argument('--small-port', type=int, default=8081)
    ap.add_argument('--large-port', type=int, default=8082)
    ap.add_argument('--gpu', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rj = os.path.join(a.run_dir, 'run.json')
    t0 = time.time()
    while not os.path.exists(rj):
        if time.time() - t0 > 900:
            sys.exit(f'{rj} not found')
        time.sleep(2)
    run = json.load(open(rj))
    pad = run['pads'][0]
    port = {int(k): v.get('offboard_udp', 14540 + int(k)) for k, v in run.get('ports', {}).items()}.get(0, 14540)
    cmap = CityMap(os.environ.get('THINKFLY_SCENE', os.path.join(HERE, '..', 'scene', 'urban_isaac.json')), a.alt)

    meter = PowerMeter(gpu=a.gpu)
    idle = meter.calibrate_idle(10.0, 3)
    print(f'reasoning GPU {a.gpu} idle power {idle:.1f} W', flush=True)
    small = Model('small', a.small_port, meter, thinking_flag=True)
    large = Model('large', a.large_port, meter, thinking_flag=True)
    json.dump({'idle_W': idle, 'idle_windows_W': meter.idle_windows, 'gpu': a.gpu, 'alt': a.alt, 'reserve': RESERVE, 'step_s': STEP_S,
               'period_s': PERIOD_S}, open(os.path.join(a.out, 'setup.json'), 'w'), indent=1)

    d = X500(0, pad, port)
    while d.pos is None or d.t is None:
        d.poll()
        d.gcs_heartbeat()
        time.sleep(0.1)
    for name, v in (('MPC_XY_VEL_MAX', 8.0), ('MPC_XY_CRUISE', 8.0), ('MPC_Z_VEL_MAX_UP', 4.0),
                    ('MPC_Z_V_AUTO_UP', 4.0), ('MPC_Z_VEL_MAX_DN', 3.0), ('MPC_Z_V_AUTO_DN', 3.0),
                    ('COM_DISARM_LAND', 1.0)):
        d.set_param(name, v)
    print('drone ready at', np.round(d.pos, 1), flush=True)

    for seed in parse_range(a.scenarios):
        cmap.reset_dynamic()
        sc = make_scenario(seed, cmap, np.array(pad[:2]), a.alt)
        json.dump(sc, open(os.path.join(a.out, f'scenario_{seed}.json'), 'w'), indent=1)
        for agent in a.agents.split(','):
            f = os.path.join(a.out, f'mission_{seed}_{agent}.json')
            if os.path.exists(f):
                continue
            cmap.reset_dynamic()
            print(f'--- scenario {seed} agent {agent} (battery {sc["battery0"]} %)', flush=True)
            # wait until landed and disarmed on the pad
            tw = time.time()
            while (d.armed or d.pos is None) and time.time() - tw < 120:
                d.poll(); d.gcs_heartbeat(); time.sleep(0.1)
            time.sleep(3.0)
            m = Mission(d, agent, sc, cmap, small, large)
            res = m.run()
            if res is None:
                print(f'    {agent}: drone did not arm in 180 s (simulator stalled?); restart needed', flush=True)
                sys.exit(3)
            json.dump(res, open(f, 'w'))
            print(f'    {agent}: safe={res["safe_completion"]} inspected={res["inspected"]} skipped={list(res["skipped"])} '
                  f'viol={res["violations"]} calls small/large={res["n_small"]}/{res["n_large"]} '
                  f'E={res["energy_J"]:.0f} J time={res["mission_time_s"]} s', flush=True)


if __name__ == '__main__':
    main()
