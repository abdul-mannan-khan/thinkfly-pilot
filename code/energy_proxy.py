#!/usr/bin/env python3
"""Ground-station energy proxy for the HIL run: forwards chat requests to the large-model server and adds the
measured GPU energy of that request to the reply (field "x_energy": gross_J, net_J, window_s).

The drone (Jetson) cannot read the ground station's GPU, so the ground station measures each request itself, the same
way as in the simulation runs (NVML cumulative energy counter + 0.3 s tail, minus idle power over the same window).
  energy_proxy.py --listen 8092 --target 8082 --gpu 1
"""
import argparse
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai import PowerMeter

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen', type=int, default=8092)
    ap.add_argument('--target', type=int, default=8082)
    ap.add_argument('--gpu', type=int, default=1)
    a = ap.parse_args()
    meter = PowerMeter(a.gpu)
    print(f'idle {meter.calibrate_idle(10.0, 3):.1f} W', flush=True)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = urllib.request.urlopen(f'http://127.0.0.1:{a.target}{self.path}', timeout=10).read()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = self.rfile.read(int(self.headers['Content-Length']))
            req = urllib.request.Request(f'http://127.0.0.1:{a.target}{self.path}', data=data,
                                         headers={'Content-Type': 'application/json'})
            with LOCK:                                   # one request at a time: clean energy windows
                start = meter.read()
                r = json.loads(urllib.request.urlopen(req, timeout=180).read())
                gross, net = meter.finish(start)
            r['x_energy'] = {'gross_J': round(gross, 2), 'net_J': round(net, 2), 'idle_W': round(meter.idle, 2)}
            out = json.dumps(r).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *x):
            pass

    ThreadingHTTPServer(('127.0.0.1', a.listen), H).serve_forever()


if __name__ == '__main__':
    main()
