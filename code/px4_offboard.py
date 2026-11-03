"""Minimal PX4 offboard helper (pymavlink): world ENU <-> PX4 local NED around the take-off pad."""
import numpy as np
from pymavlink import mavutil

OFFBOARD = 6                                   # PX4 custom main mode
MASK_POS = 0b0000111111111000                  # position setpoint only


class Drone:
    def __init__(self, i, pad, port):
        self.i, self.pad = i, np.array(pad, float)
        self.m = mavutil.mavlink_connection(f'udpin:0.0.0.0:{port}', source_system=250, source_component=190 + i % 50)
        self.pos = self.vel = None
        self.path, self.s = None, 0.0
        self.armed, self.mode, self.landed = False, None, None

    def poll(self):
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.pos = self.pad + np.array([msg.y, msg.x, -msg.z])
                self.vel = np.array([msg.vy, msg.vx, -msg.vz])
            elif t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = (msg.custom_mode >> 16) & 0xFF

    def gcs_heartbeat(self):
        self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def set_param(self, name, value):
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component, name.encode(), float(value),
                                  mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def cmd(self, c, *p):
        p = list(p) + [0] * (7 - len(p))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, c, 0, *p)

    def setpoint(self, world):
        d = np.asarray(world) - self.pad
        self.m.mav.set_position_target_local_ned_send(0, self.m.target_system, self.m.target_component,
                                                      mavutil.mavlink.MAV_FRAME_LOCAL_NED, MASK_POS,
                                                      d[1], d[0], -d[2], 0, 0, 0, 0, 0, 0, 0, 0)

    def carrot(self, lead):
        """Point on the path `lead` m ahead of the vehicle's progress; second value: reached the end."""
        segs = np.diff(self.path, axis=0)
        lens = np.linalg.norm(segs, axis=1)
        total = lens.sum()

        def at(s):
            k = 0
            while k < len(lens) - 1 and s > lens[k]:
                s -= lens[k]
                k += 1
            return self.path[k] + segs[k] * min(s / max(lens[k], 1e-9), 1.0)
        while self.s < total and np.linalg.norm(at(self.s) - self.pos) < lead:
            self.s = min(total, self.s + 0.25)
        return at(self.s), self.s >= total and np.linalg.norm(self.path[-1] - self.pos) < 1.5
