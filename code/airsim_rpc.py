"""Minimal AirSim client (msgpack-rpc over TCP, port 41451).

The official `airsim` Python package depends on msgpack-rpc-python / tornado 4, which do not run on Python 3.12,
so this speaks AirSim's msgpack-rpc protocol directly: request = [0, msgid, method, params], reply = [1, msgid, err, result].
call()  waits for the reply; post() sends without waiting (like the client's *Async methods; replies are drained).
"""
import itertools
import socket
import threading

import msgpack


class AirSimRPC:
    def __init__(self, host='127.0.0.1', port=41451, timeout=60.0):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.ids = itertools.count(1)
        self.unp = msgpack.Unpacker(raw=False, strict_map_key=False)
        self.lock = threading.Lock()
        self.waiting = {}            # msgid -> reply (None until it arrives)

    def _send(self, method, params):
        mid = next(self.ids)
        self.s.sendall(msgpack.packb([0, mid, method, list(params)], use_bin_type=True))
        return mid

    def _read_until(self, mid):
        while True:
            for msg in self.unp:
                if msg[0] == 1:
                    if msg[1] == mid:
                        if msg[2] is not None:
                            raise RuntimeError(f'AirSim error: {msg[2]}')
                        return msg[3]
                    # a reply to an earlier post(): ignore
            data = self.s.recv(1 << 20)
            if not data:
                raise ConnectionError('AirSim closed the connection')
            self.unp.feed(data)

    def call(self, method, *params):
        with self.lock:
            return self._read_until(self._send(method, params))

    def post(self, method, *params):
        """Send without waiting. Use a connection only for post() and call start_drain() on it once, so that the
        replies are read and discarded (otherwise the socket buffer fills and AirSim stops serving the connection)."""
        with self.lock:
            self._send(method, params)

    def start_drain(self):
        def run():
            while True:
                try:
                    if not self.s.recv(1 << 16):
                        return
                except OSError:
                    return
        self.s.settimeout(None)
        threading.Thread(target=run, daemon=True).start()
        return self


YAW_FREE = {'is_rate': True, 'yaw_or_rate': 0.0}
DRIVE_FORWARD_ONLY, DRIVE_MAX_DOF = 1, 0


def image_request(camera, image_type=0, compress=True):
    return {'camera_name': camera, 'image_type': image_type, 'pixels_as_float': False, 'compress': compress}
