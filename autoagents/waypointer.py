import math
import numpy as np

from agents.navigation.local_planner import RoadOption

# jxy: addition; (add display.py and fix RoutePlanner.py or Waypointer.py)
from team_code.planner import Plotter

class Waypointer:
    
    EARTH_RADIUS = 6371e3 # 6371km
    
    def __init__(self, global_plan, current_gnss, threshold_lane=10., threshold_before=4.5, threshold_after=4.5, pop_lane_change=True, debug_size=720):
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after
        self._threshold_lane = threshold_lane
        self._pop_lane_change = pop_lane_change

        self._lane_change_counter = 0
        
        # Convert lat,lon to x,y
        cos_0 = 0.
        for gnss, _ in global_plan:
            cos_0 += gnss['lat'] * (math.pi / 180)
        cos_0 = cos_0 / (len(global_plan))
        self.cos_0 = cos_0
        
        # jxy: like def set_route() as this implement rm self.route.popleft()
        self.global_plan = []
        for node in global_plan:
            gnss, cmd = node

            x, y = self.latlon_to_xy(gnss['lat'], gnss['lon'])
            self.global_plan.append((x, y, cmd))

        lat, lon, _ = current_gnss
        cx, cy = self.latlon_to_xy(lat, lon)
        self.checkpoint = (cx, cy, RoadOption.LANEFOLLOW)
        
        self.current_idx = -1

        self.debug = Plotter(debug_size)
        self.centre = np.array(self.global_plan[int(len(self.global_plan)//2)])[:2]
        self.store_wps = []
        self.store_wps_real = []

    def tick(self, gnss):
        
        lat, lon, _ = gnss
        x, y = self.latlon_to_xy(lat, lon) # jxy: like [:2] and _get_position
        gps = np.array([x,y])

        self.debug.clear()

        self.cur_veh_gps = gps
        if self.centre is None:
            self.centre = gps

        for i, (wx, wy, cmd) in enumerate(self.global_plan):

            # CMD remap... HACK...
            distance = np.linalg.norm([x-wx, y-wy])

            if self.checkpoint[2] == RoadOption.LANEFOLLOW and cmd != RoadOption.LANEFOLLOW:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            if distance < threshold and i-self.current_idx == 1:
                self.checkpoint = (wx, wy, cmd)
                self.current_idx += 1
                break

        for i, (wx, wy, cmd) in enumerate(self.global_plan):
            route_pt = (wx, wy)
            if i == self.current_idx - 1:
                self.debug.dot(self.centre, np.array(route_pt), (0, 255, 0))
            elif i < self.current_idx:
                continue
            elif i == self.current_idx:
                self.debug.dot(self.centre, np.array(route_pt), (255, 0, 0))
            else:
                r = 255 * int(True)
                g = 255 * int(cmd.value == 4)
                b = 255
                self.debug.dot(self.centre, np.array(route_pt), (r, g, b))

        self.store_wps_real.append(gps)
        for pos in self.store_wps_real:
            self.debug.dot(self.centre, pos, (0, 0, 255))

        return self.checkpoint, gps


    def latlon_to_xy(self, lat, lon):
        
        x = self.EARTH_RADIUS * lat * (math.pi / 180)
        y = self.EARTH_RADIUS * lon * (math.pi / 180) * math.cos(self.cos_0)

        return x, y

    def run_step2(self, _poses, is_gps=True, color=(255, 255, 0), store=True):
        if is_gps:
            poses = _poses.copy()
        else:
            poses = _poses + self.cur_veh_gps if len(_poses) else []

        for pos in self.store_wps:
            self.debug.dot(self.centre, pos, color)

        if store:
            self.store_wps.extend(list(poses))

        for pos in poses:
            self.debug.dot(self.centre, pos, color)

    def show_route(self):
        self.debug.show()
