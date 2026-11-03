"""City map for ThinkFly missions in the Isaac Sim urban scene.

Built from the scene's building boxes (sim/scenes/generated/urban.json; world frame x east, y north, z up, metres).
Gives: an occupancy grid at the cruise altitude (buildings taller than the cruise altitude, inflated by a clearance),
no-fly zones (circles) announced during the mission, a square geofence, A* paths, and a simple battery model.
These are the cheap, non-learned parts of the system: the flight controller's planner and the rule checker's maths.
"""
import heapq
import json
import math

import numpy as np

# Battery model (percent of capacity). Calibrated so that a 1 km leg at 8 m/s costs ~13 % and hovering ~3 %/min.
K_TIME = 0.05      # %/s, any airborne second (hover or cruise)
K_DIST = 0.008     # %/m flown horizontally
INSPECT_S = 5.0    # hover time per inspection (s)
CRUISE = 8.0       # m/s


def leg_cost(dist_m, hover_s=0.0):
    """Battery (%) to fly dist_m at cruise speed and hover hover_s."""
    return dist_m * K_DIST + (dist_m / CRUISE + hover_s) * K_TIME


class CityMap:
    def __init__(self, scene_json, alt, clearance=6.0, res=4.0, fence_margin=30.0):
        meta = json.load(open(scene_json))
        self.buildings = meta['buildings']
        e0, e1 = meta['extent']
        self.lo, self.hi = e0 + fence_margin, e1 - fence_margin       # geofence (square)
        self.res, self.alt, self.clear = res, alt, clearance
        self.n = int(math.ceil((self.hi - self.lo) / res)) + 1
        g = self.lo + np.arange(self.n) * res
        self.X, self.Y = np.meshgrid(g, g, indexing='ij')             # grid[ix, iy]
        occ = np.zeros((self.n, self.n), bool)
        self.tall = []
        for b in self.buildings:
            if b['height'] > alt - 4.0:                               # cannot be over-flown with 4 m margin
                (cx, cy), (sx, sy) = b['centre'], b['size']
                occ |= (np.abs(self.X - cx) <= sx / 2 + clearance) & (np.abs(self.Y - cy) <= sy / 2 + clearance)
                self.tall.append(b)
        self.static = occ
        self.nfz = []                                                 # (cx, cy, r, name)
        self.version = 0
        self._cache = {}
        self._rebuild()

    # ---- map state -------------------------------------------------------------------------------------------
    def _rebuild(self):
        b = self.static.copy()
        for cx, cy, r, _ in self.nfz:
            b |= (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= (r + 10.0) ** 2   # 10 m planning margin
        self.blocked = b
        self.version += 1
        self._cache = {}

    def add_nfz(self, cx, cy, r, name):
        self.nfz.append((float(cx), float(cy), float(r), name))
        self._rebuild()

    def reset_dynamic(self):
        self.nfz = []
        self._rebuild()

    def cell(self, p):
        ix = int(round((p[0] - self.lo) / self.res))
        iy = int(round((p[1] - self.lo) / self.res))
        return min(max(ix, 0), self.n - 1), min(max(iy, 0), self.n - 1)

    def xy(self, c):
        return np.array([self.lo + c[0] * self.res, self.lo + c[1] * self.res])

    # ---- rule predicates on true positions (used to score violations) ----------------------------------------
    def in_nfz(self, p, margin=0.0):
        return [name for cx, cy, r, name in self.nfz if math.hypot(p[0] - cx, p[1] - cy) <= r + margin]

    def in_fence(self, p):
        return self.lo <= p[0] <= self.hi and self.lo <= p[1] <= self.hi

    def collides(self, p, margin=1.0):
        for b in self.buildings:
            (cx, cy), (sx, sy) = b['centre'], b['size']
            if abs(p[0] - cx) <= sx / 2 + margin and abs(p[1] - cy) <= sy / 2 + margin and p[2] <= b['height'] + margin:
                return True
        return False

    # ---- planning ---------------------------------------------------------------------------------------------
    def _nearest_free(self, c, rmax=12):
        if not self.blocked[c]:
            return c
        for r in range(1, rmax + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                    q = (c[0] + dx, c[1] + dy)
                    if 0 <= q[0] < self.n and 0 <= q[1] < self.n and not self.blocked[q]:
                        return q
        return None

    def reachable(self, p):
        """True if p is in free space (not inside a building clearance or a no-fly zone)."""
        return not self.blocked[self.cell(p)]

    def path(self, a, b):
        """A* path (list of xy points) from a to b in free space, or None. a may start in blocked space (the
        drone may already be inside a new zone's planning margin): it is snapped to the nearest free cell."""
        ca, cb = self.cell(a), self.cell(b)
        if self.blocked[cb]:
            return None                                               # goal inside a building margin or a no-fly zone
        key = (ca, cb)
        if key in self._cache:
            return self._cache[key]
        s = self._nearest_free(ca)
        if s is None:
            return None
        n, blk = self.n, self.blocked
        g = {s: 0.0}
        par = {s: None}
        h = lambda c: math.hypot(c[0] - cb[0], c[1] - cb[1])
        pq = [(h(s), 0.0, s)]
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        found = False
        while pq:
            _, gc, c = heapq.heappop(pq)
            if c == cb:
                found = True
                break
            if gc > g.get(c, 1e18) + 1e-9:
                continue
            for dx, dy, w in moves:
                q = (c[0] + dx, c[1] + dy)
                if not (0 <= q[0] < n and 0 <= q[1] < n) or blk[q]:
                    continue
                if dx and dy and (blk[c[0] + dx, c[1]] or blk[c[0], c[1] + dy]):
                    continue                                          # no corner cutting
                ng = gc + w
                if ng < g.get(q, 1e18):
                    g[q], par[q] = ng, c
                    heapq.heappush(pq, (ng + h(q), ng, q))
        if not found:
            self._cache[key] = None
            return None
        cells = []
        c = cb
        while c is not None:
            cells.append(c)
            c = par[c]
        cells.reverse()
        pts = self._smooth(cells)
        out = [np.asarray(a[:2], float)] + [self.xy(c) for c in pts[1:-1]] + [np.asarray(b[:2], float)]
        self._cache[key] = out
        return out

    def _los(self, c0, c1):
        k = int(max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))) * 2 + 1
        for t in np.linspace(0, 1, k):
            q = (int(round(c0[0] + (c1[0] - c0[0]) * t)), int(round(c0[1] + (c1[1] - c0[1]) * t)))
            if self.blocked[q]:
                return False
        return True

    def _smooth(self, cells):
        out, i = [cells[0]], 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._los(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    @staticmethod
    def length(pts):
        return float(sum(np.linalg.norm(np.asarray(pts[k + 1]) - np.asarray(pts[k])) for k in range(len(pts) - 1)))

    def dist(self, a, b):
        p = self.path(a, b)
        return None if p is None else self.length(p)

    def route(self, start, waypoints, home):
        """Polyline through waypoints then home; returns (points, per-leg lengths) or (None, failing leg index)."""
        pts, legs, cur = [np.asarray(start[:2], float)], [], start
        for k, w in enumerate(list(waypoints) + [home]):
            p = self.path(cur, w)
            if p is None:
                return None, k
            pts += p[1:]
            legs.append(self.length(p))
            cur = w
        return pts, legs
