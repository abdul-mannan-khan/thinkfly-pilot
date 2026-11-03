"""Reasoning models, energy metering and the rule checker for the ThinkFly pilot.

- Two real open language models served by llama.cpp on a dedicated GPU (GPU 1; the simulator renders on GPU 0):
  small = Qwen3.5-4B (Q4_K_M), large = Qwen3.5-27B (Q4_K_M); same 2026 model family, thinking mode off for both.
- PowerMeter reads that GPU's NVML cumulative energy counter before and after each request.
  Net energy = measured energy minus the GPU's idle power (both models loaded) over the same window.
- Both models get the same prompt and the same JSON schema (grammar-constrained decoding, so the output is parseable;
  the content is still the model's own decision).
- check_plan(): the automatic rule checker. It recomputes the plan's path with the map and checks the mission rules.
"""
import json
import threading
import time
import urllib.request

import numpy as np

from citymap import INSPECT_S, leg_cost


class PowerMeter:
    """Energy of the reasoning GPU from NVML's cumulative energy counter (mJ since driver load; Volta and newer).
    Gross energy of a request = counter after - counter before (+ a 0.3 s tail, because the GPU stays in a high
    power state briefly after the last token). Net = gross - idle power x the same window. Idle power is measured
    with both models loaded and no request running."""
    TAIL = 0.3

    def __init__(self, gpu=1):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.idle = None

    def read(self):
        return time.time(), self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0   # J

    def calibrate_idle(self, seconds=10.0, windows=3):
        """Lowest average power over several windows (the GPU can stay in a higher power state after activity)."""
        ws = []
        for _ in range(windows):
            t0, e0 = self.read()
            time.sleep(seconds)
            t1, e1 = self.read()
            ws.append((e1 - e0) / (t1 - t0))
        self.idle = min(ws)
        self.idle_windows = ws
        return self.idle

    def finish(self, start):
        time.sleep(self.TAIL)
        t1, e1 = self.read()
        gross = e1 - start[1]
        return gross, max(gross - (self.idle or 0.0) * (t1 - start[0]), 0.0)


class JetsonMeter:
    """Energy of the whole Jetson module from its INA3221 VDD_IN rail (board input power), integrated at 50 Hz in a
    background thread. Same interface as PowerMeter: read() -> (time, cumulative J)."""
    TAIL = 0.3

    def __init__(self, rail='VDD_IN', hz=50.0):
        import glob
        self.v = self.i = None
        for lab in glob.glob('/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label'):
            if open(lab).read().strip() == rail:
                k = lab.split('/in')[-1].split('_')[0]
                d = lab.rsplit('/', 1)[0]
                self.v, self.i = f'{d}/in{k}_input', f'{d}/curr{k}_input'
        if self.v is None:
            raise RuntimeError(f'INA3221 rail {rail} not found')
        self.idle, self.E, self.dt = None, 0.0, 1.0 / hz
        self.lock = threading.Lock()
        self.t_last, self.p_last = time.time(), self.power()
        threading.Thread(target=self._loop, daemon=True).start()

    def power(self):
        return int(open(self.v).read()) * int(open(self.i).read()) * 1e-6      # mV * mA -> W

    def _loop(self):
        while True:
            time.sleep(self.dt)
            p, t = self.power(), time.time()
            with self.lock:
                self.E += 0.5 * (p + self.p_last) * (t - self.t_last)
                self.t_last, self.p_last = t, p

    def read(self):
        with self.lock:
            return self.t_last, self.E

    calibrate_idle = PowerMeter.calibrate_idle
    finish = PowerMeter.finish


class Model:
    def __init__(self, name, port, meter, thinking_flag=False, host='127.0.0.1', remote_energy=False):
        self.name, self.url, self.meter, self.tf = name, f'http://{host}:{port}/v1/chat/completions', meter, thinking_flag
        self.remote = remote_energy        # energy measured at the ground station by energy_proxy.py (returned in the reply)
        self.lock = threading.Lock()

    def ask(self, system, user, schema, max_tokens=320):
        body = {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                'temperature': 0.2, 'top_p': 0.9, 'max_tokens': max_tokens, 'seed': 7,
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'plan', 'schema': schema}}}
        if self.tf:
            body['chat_template_kwargs'] = {'enable_thinking': False}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with self.lock:                          # one request at a time on the reasoning GPU: clean energy windows
            start = self.meter.read() if not self.remote else (time.time(), 0.0)
            t0 = start[0]
            r = {}
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=180).read())
                text = r['choices'][0]['message']['content']
                tim = r.get('timings', {})
                n_in = r.get('usage', {}).get('prompt_tokens', tim.get('prompt_n', 0))      # whole prompt
                n_proc = tim.get('prompt_n')                                               # not served from cache
                n_out = r.get('usage', {}).get('completion_tokens', tim.get('predicted_n', 0))
                prefill_ms, decode_ms = tim.get('prompt_ms'), tim.get('predicted_ms')
                err = None
            except Exception as e:               # timeout or server error: the agent keeps its current plan
                text, n_in, n_out, err, n_proc = '', 0, 0, repr(e)[:200], None
                prefill_ms = decode_ms = None
            t1 = time.time()
            if self.remote:
                xe = (r.get('x_energy') if not err else None) or {}
                gross, net = xe.get('gross_J', 0.0), xe.get('net_J', 0.0)
            else:
                gross, net = self.meter.finish(start)
        try:
            plan = json.loads(text) if text else None
        except json.JSONDecodeError:
            plan, err = None, 'json'
        return {'model': self.name, 'text': text, 'plan': plan, 'tokens_in': n_in, 'tokens_prompt_processed': n_proc, 'tokens_out': n_out,
                'latency_s': round(t1 - t0, 3), 'prefill_ms': prefill_ms, 'decode_ms': decode_ms,
                'decoding': 'temperature 0.2, top_p 0.9, seed 7, JSON-schema grammar', 'energy_gross_J': round(gross, 2), 'energy_J': round(net, 2),
                't0': t0, 't1': t1, 'error': err}


SYSTEM = """You are the mission-reasoning module of a small inspection drone flying over a city.
The flight controller flies whatever plan you give it and finds collision-free paths itself.
Mission rules:
R1 Never enter a no-fly zone. A target inside a no-fly zone cannot be inspected: put it in "skip" with the reason.
R2 The battery must still be at or above the reserve when the drone lands at home. Use the battery-cost table:
   battery at home = battery now - (cost of every leg of your order, including the final leg home)
   - inspection cost for each target in your order. If the full order does not fit, drop targets (put them in
   "skip" with reason "battery") or set return_home_now = true.
R3 Inspect as many remaining reachable targets as rule R2 allows, in a short order.
Answer with JSON only: order (target ids to inspect next, in order), skip (dropped targets with a reason),
return_home_now (true to abort and fly straight home), note (one short sentence)."""


def schema(ids):
    ids = list(ids) or ['none']
    return {'type': 'object', 'properties': {
        'order': {'type': 'array', 'items': {'type': 'string', 'enum': ids}},
        'skip': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'id': {'type': 'string', 'enum': ids}, 'reason': {'type': 'string', 'maxLength': 60}},
            'required': ['id', 'reason']}},
        'return_home_now': {'type': 'boolean'},
        'note': {'type': 'string', 'maxLength': 120}},
        'required': ['order', 'skip', 'return_home_now', 'note']}


def state_text(st, history=None):
    """Compact structured state (ThinkFly) or the same plus the full step history (step-by-step agent)."""
    d = {'time_s': round(st['t'], 1), 'battery_pct': round(st['battery'], 1), 'reserve_pct': st['reserve'],
         'inspection_cost_pct_each': round(leg_cost(0, INSPECT_S), 2),
         'remaining_targets': st['targets'], 'battery_cost_pct': st['costs'],
         'no_fly_zones': st['nfz'], 'event': st['event'], 'messages': st['messages']}
    s = json.dumps(d, separators=(',', ':'))
    if history:
        s = 'History of previous steps:\n' + '\n'.join(history) + '\nCurrent state:\n' + s
    return s


def check_plan(plan, st, cmap, pos, home, targets_xy, strict=True):
    """Rule checker. Safety rules R1 (no-fly zones) and R2 (battery reserve) always; with strict=True also the
    coverage rule R3 (no reachable target dropped if it fits the battery), used to decide whether a small-model plan
    is escalated. Returns (ok, reason, route_points, predicted_battery_at_home)."""
    if plan is None:
        return False, 'no parseable plan', None, None
    rem = [t['id'] for t in st['targets']]
    order = [i for i in plan.get('order', []) if i in rem]
    if len(set(order)) != len(order):
        return False, 'duplicate target in order', None, None
    if plan.get('return_home_now'):
        order = []
    skipped = {s.get('id') for s in plan.get('skip', [])}
    for i in rem:                                   # every remaining target must be either ordered or skipped
        if i not in order and i not in skipped and not plan.get('return_home_now'):
            return False, f'target {i} neither ordered nor skipped', None, None
    for i in order:                                 # R1: no target inside a no-fly zone
        if cmap.in_nfz(targets_xy[i]):
            return False, f'target {i} is inside a no-fly zone', None, None
    pts, legs = cmap.route(pos, [targets_xy[i] for i in order], home)
    if pts is None:
        return False, f'no safe path for leg {legs + 1}', None, None
    bat = st['battery'] - sum(leg_cost(L) for L in legs) - len(order) * leg_cost(0, INSPECT_S)
    if bat < st['reserve'] + 1.0:                   # R2 with a 1 % margin for the model's estimate error
        return False, f'battery at home {bat:.1f} % < reserve {st["reserve"]} %', pts, bat
    if not strict:
        return True, 'ok', pts, bat
    for i in rem:                                   # R3: a skipped target needs a valid reason
        if i in skipped and i not in order:
            if not cmap.in_nfz(targets_xy[i]):
                # dropping a reachable target is allowed only if adding it back would break the battery rule
                for k in range(len(order) + 1):                 # try every insertion point
                    o2 = order[:k] + [i] + order[k:]
                    p2, l2 = cmap.route(pos, [targets_xy[q] for q in o2], home)
                    if p2 is None:
                        continue
                    b2 = st['battery'] - sum(leg_cost(L) for L in l2) - len(o2) * leg_cost(0, INSPECT_S)
                    if b2 >= st['reserve'] + 1.0:
                        return False, f'target {i} skipped although it fits the battery', None, None
    return True, 'ok', pts, bat
