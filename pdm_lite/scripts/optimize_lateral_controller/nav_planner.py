"""
Some helpful classes for planning and control for the privileged autopilot
"""

import math
from copy import deepcopy
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import carla
import warnings
from enum import IntEnum
from scipy.interpolate import splprep, splev

from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class PIDController(object):
    """
    PID controller
    """

    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self._saved_window = deque([0 for _ in range(n)], maxlen=n)
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def reset_error_integral(self):
        self._window = deque(len(self._window) * [0])

    def step(self, error):
        self._window.append(error)
        if len(self._window) >= 2:
            integral = sum(self._window) / len(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self.k_p * error + self.k_i * integral + self.k_d * derivative

    def save(self):
        self._saved_window = deepcopy(self._window)

    def load(self):
        self._window = self._saved_window


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class RoutePlanner(object):
    def __init__(self, distance_between_route_points=0.1, times_supersample=10, append_meters_to_provided_route=50):
        # min_distance, max_distance are currently not used and only implemented to make the class similar to the previous RoutePlanner
        self.distance_between_route_points = distance_between_route_points
        self.append_meters_to_provided_route = append_meters_to_provided_route
        self.times_supersample = times_supersample
        
        self.route_wp = []
        self.route_np = np.array([[]])
        self.commands = []
        self.rotation_angles = []

        self.distance_to_next_stop_sign = np.array([])
        self.next_stop_sign = []

        self.distance_to_next_traffic_light = np.array([])
        self.next_traffic_light = []

        self.speed_limits = np.array([])
        self.curvature_dependent_speed = np.array([])

        self.yaw_angles = np.array([])

        self.route_idx = 0
        self.last_route_idx = 0
        self.is_last = False

    def save(self):
        self.last_route_idx = self.route_idx

    def load(self):
        self.route_idx = self.last_route_idx
        self.is_last = self.route_idx == self.route_np.shape[0]-1

    def run_step(self, pos):
        until = min(self.route_idx+40, self.route_np.shape[0])
        self.route_idx += np.argmin(np.linalg.norm(pos[None, :2] - self.route_np[self.route_idx:until, :2], axis=1))

        self.is_last = self.route_idx == self.route_np.shape[0]-1

        return self.route_np[self.route_idx:], \
                self.route_wp[self.route_idx:], \
                self.yaw_angles[self.route_idx], \
                self.commands[self.route_idx:], \
                self.distance_to_next_traffic_light[self.route_idx], \
                self.next_traffic_light[self.route_idx], \
                self.distance_to_next_stop_sign[self.route_idx], \
                self.next_stop_sign[self.route_idx], \
                self.speed_limits[self.route_idx], \
                self.curvature_dependent_speed[self.route_idx]

    def set_route(self, global_plan, gps=False, carla_map=None, starts_with_parking_exit=False, vehicle_loc=None):
        self.route_idx = self.append_meters_to_provided_route * 10
        self.last_route_idx = self.route_idx

        ################################################################################################################################################
        # get all waypoint objects of route and add x meters to the end to make sure the ego ends the route properly and to avoid unexpected sideeffects
        ################################################################################################################################################
        route_wp = [transform.location for transform, _ in global_plan]
        route_wp = [carla_map.get_waypoint(loc) for loc in route_wp]
        if starts_with_parking_exit and vehicle_loc is not None: # workaraound for ParkingExit scenario
            self.route_idx = 0
            self.last_route_idx = 0
        else:
            for _ in range(self.append_meters_to_provided_route):
                route_wp.insert(0, route_wp[0].previous(1)[0])

        for _ in range(self.append_meters_to_provided_route):
            route_wp.append(route_wp[-1].next(1)[0])

        ################################################################################################################################################
        # generate a numpy array containing the route locations
        ################################################################################################################################################
        route_np = [wp.transform.location for wp in route_wp]
        if starts_with_parking_exit and vehicle_loc is not None: # workaraound for ParkingExit scenario
            route_np = [vehicle_loc] + route_np[5:]
            global_plan = global_plan[5:]
            
            cmds = [RoadOption.CHANGELANELEFT]*1 + [cmd for _, cmd in global_plan] + [RoadOption.LANEFOLLOW]*self.append_meters_to_provided_route
        else:
            cmds = [RoadOption.LANEFOLLOW]*self.append_meters_to_provided_route + [cmd for _, cmd in global_plan] + [RoadOption.LANEFOLLOW]*self.append_meters_to_provided_route

        route_np = np.array([[loc.x, loc.y, loc.z] for loc in route_np])

        ################################################################################################################################################
        # smooth and supersample that route
        ################################################################################################################################################
        route_np_smooth, cmds_smooth = self.smooth_and_supersample(route_np, cmds)

        ################################################################################################################################################
        # compute rotation angles
        ################################################################################################################################################
        self.rotation_angles = self.compute_rotation_angles(route_np_smooth)

        self.route_np = route_np_smooth
        self.commands = cmds_smooth
        self.yaw_angles = self.rotation_angles
        
        for route_loc in self.route_np:
            wp = carla_map.get_waypoint(carla.Location(x=route_loc[0], y=route_loc[1], z=route_loc[2]))
            self.route_wp.append(wp)
        
    def compute_rotation_angles(self, route_np):
        ################################################################################################################################################
        # compute the yaw angle the ego vehicle will have, when it's at the individual route points
        ################################################################################################################################################
        indices = np.arange(1, route_np.shape[0]-1)
        diff = route_np[indices+1] - route_np[indices-1]
        yaws = np.arctan2(diff[:,1], diff[:,0]) * 180. / np.pi
        yaws = np.concatenate([[yaws[0]], yaws, [yaws[-1]]])

        return yaws

    def smooth_and_supersample(self, route_np, cmds):
        def supersample_route(route):
            n_times = 100
            t = np.linspace(0, 1, route.shape[0])
            u = np.linspace(0, 1, route.shape[0]*n_times)

            arr = np.concatenate([t[:, None], route[:, :]], axis=1)
            tck, _ = splprep(arr.T, s=1, k=1)

            _, x, y, z = splev(u, tck)

            route_supersampled = np.vstack([x, y, z]).T
            
            return route_supersampled

        def smooth_route(route, cmds):
            indices = np.where([cmd in [5, 6] for cmd in cmds])[0]
            indices = indices[None] + np.arange(-1, 1)[:, None]
            indices = np.unique(indices.flatten())

            indices2 = indices[None] + np.arange(-1, 2)[:, None]
            locs_smooth = route[indices2].mean(axis=0)

            route_smooth = np.copy(route)
            route_smooth[indices] = locs_smooth
            
            return route_smooth

        # route_tmp = smooth_route(route_np, cmds)
        route_tmp = supersample_route(route_np)

        distances = np.cumsum(np.linalg.norm(np.diff(route_tmp, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        distances = distances % 0.1

        indices = np.insert(np.argwhere(distances[1:]<distances[:-1]), 0, 0)
        route_final = route_tmp[indices]

        l_cmds = [cmds[idx] for idx in np.minimum(np.round(indices/10), len(cmds)-1).astype('int')]

        return route_final, l_cmds

    def add_further_info_to_route(self, carla_world, carla_map):
        self.compute_distances_to_traffic_lights(carla_world, carla_map)
        self.compute_distances_to_stop_signs(carla_world, carla_map)
        self.compute_speed_limits(carla_world, carla_map)
        self.compute_curvature_dependent_speeds_limits(carla_world, carla_map)

    def compute_distances_to_traffic_lights(self, carla_world, carla_map):
        ################################################################################################################################################
        # compute the distance to the next traffic light from each individual route location
        ################################################################################################################################################
        self.distance_to_next_traffic_light = np.full(self.route_np.shape[0], np.inf)
        self.next_traffic_light = [None] * self.route_np.shape[0]

        next_traffic_light = None

        traffic_light_already_recorded = False
        distance_idx = np.inf
        for i in range(len(self.route_np)-1, -1, -1):
            waypoint = self.route_wp[i]
            traffic_lights = carla_world.get_traffic_lights_from_waypoint(waypoint, 5)

            if traffic_lights:
                if not traffic_light_already_recorded:
                    distance_idx = 0
                    next_traffic_light = traffic_lights[0]
                else:
                    distance_idx += 1

                traffic_light_already_recorded = True
            else:
                distance_idx += 1
                traffic_light_already_recorded = False

            self.next_traffic_light[i] = next_traffic_light
            self.distance_to_next_traffic_light[i] = float(distance_idx)/10.

        self.distance_to_next_traffic_light = np.concatenate([self.distance_to_next_traffic_light[:-40], 40 * [np.inf]])
        self.next_traffic_light = self.next_traffic_light[:-40] + (40 * [None])

    def compute_distances_to_stop_signs(self, carla_world, carla_map):
        ################################################################################################################################################
        # compute the distance to the next stop sign from each individual route location
        ################################################################################################################################################
        def point_inside_boundingbox(point, bb_center, bb_extent, multiplier=1.2):
            """Checks whether or not a point is inside a bounding box."""

            # pylint: disable=invalid-name
            A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
            B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
            D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
            M = carla.Vector2D(point.x, point.y)

            AB = B - A
            AD = D - A
            AM = M - A
            am_ab = AM.x * AB.x + AM.y * AB.y
            ab_ab = AB.x * AB.x + AB.y * AB.y
            am_ad = AM.x * AD.x + AM.y * AD.y
            ad_ad = AD.x * AD.x + AD.y * AD.y

            return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad  # pylint: disable=chained-comparison

        def is_actor_affected_by_stop(wp_list, stop):
            """
            Check if the given actor is affected by the stop.
            Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
            """

            # Quick distance test
            stop_location = stop.get_transform().transform(stop.trigger_volume.location)
            actor_location = wp_list[0].transform.location
            if stop_location.distance(actor_location) > 4.0:
                return False

            # Check if the any of the actor wps is inside the stop's bounding box.
            # Using more than one waypoint removes issues with small trigger volumes and backwards movement
            stop_extent = stop.trigger_volume.extent
            stop_extent.x = max(stop_extent.x, 1)
            stop_extent.y = max(stop_extent.y, 1)
            for actor_wp in wp_list:
                if point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
                    return True

            return False

        def _scan_for_stop_sign(_list_stop_signs, wp_list):
            for stop in _list_stop_signs:
                if is_actor_affected_by_stop(wp_list, stop):
                    return stop

        def _get_waypoints(start_loc, carla_map):
            """Returns a list of waypoints starting from the ego location and a set amount forward"""
            wp_list = []
            steps = int(4.0 / 0.01)

            # Add the actor location
            wp = carla_map.get_waypoint(start_loc)
            wp_list.append(wp)

            # And its forward waypoints
            next_wp = wp
            for _ in range(steps):
                next_wps = next_wp.next(0.01)
                if not next_wps:
                    break
                next_wp = next_wps[0]
                wp_list.append(next_wp)

            return wp_list
        self.distance_to_next_stop_sign = np.full(self.route_np.shape[0], np.inf, dtype=np.float)
        self.next_stop_sign = [None] * self.route_np.shape[0]

        _list_stop_signs = []
        for _actor in carla_world.get_actors():
            if 'traffic.stop' in _actor.type_id:
                _list_stop_signs.append(_actor)

        next_stop_sign = None
        distance_idx = 0
        if _list_stop_signs:
            for i in range(self.route_np.shape[0]):
                loc = self.route_np[i]
                start_loc = carla.Location(x=loc[0], y=loc[1], z=loc[2])
                check_wps = _get_waypoints(start_loc, carla_map)
                stop_sign = _scan_for_stop_sign(_list_stop_signs, check_wps)
                self.next_stop_sign[i] = stop_sign

            for i in range(self.distance_to_next_stop_sign.shape[0]-1, -1, -1):
                if self.next_stop_sign[i] is not None:
                    next_stop_sign = self.next_stop_sign[i]
                    distance_idx = 0
                else:
                    distance_idx += 1

                self.next_stop_sign[i] = next_stop_sign
                self.distance_to_next_stop_sign[i] = float(distance_idx)/10.

    def compute_speed_limits(self, carla_world, carla_map):
        ################################################################################################################################################
        # compute the distance to the speed limits for each individual route location
        ################################################################################################################################################
        map_name = carla_map.name.split('/')[-1]
        file_name_speed_limits = f'/home/jens/Desktop/leaderboard2_human_data/team_code/speed_limits/{map_name}_speed_limits.npy'
        file_content = np.load(file_name_speed_limits, allow_pickle=True)
        map_locations = file_content.item().get('locations')
        map_speed_limits = file_content.item().get('speed_limits')

        self.speed_limits = np.empty(self.route_np.shape[0], dtype=np.float)
        previous_speed_limit = -1
        for i, loc in enumerate(self.route_np):
            if i % 100 == 0: # calculate for waypoints every 10m
                dist = np.linalg.norm(loc[None] - map_locations, axis=1)
                min_idx = np.argmin(dist)
                speed_limit = map_speed_limits[min_idx]/3.6
                self.speed_limits[i] = speed_limit
                previous_speed_limit = speed_limit
            else:
                self.speed_limits[i] = previous_speed_limit

    def compute_curvature_dependent_speeds_limits(self, carla_world, carla_map):
        ################################################################################################################################################
        # compute the distance to curvature dependent speed limits for each individual route location
        ################################################################################################################################################
       
        # route_locations = self.route_np[:, :2]
        # n = 40
        self.curvature_dependent_speed = 120/3.6 * np.ones(self.route_np.shape[0], dtype=np.float)

        # for i in range(n, self.route_np.shape[0] - n):
        #     diff = np.diff(route_locations[i - n : i + n+1][::10], axis=0) # [40, 2]
        #     a, b = diff[1:], diff[:-1] # [3, 2]
        #     l1, l2 = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)

        #     t = ((a * b).sum(axis = 1) / (l1 * l2 + 1e-8)).sum()
        #     fac = 1.53
        #     curvature = np.exp(-fac * np.abs(9 - 1 - t))
        #     corrected_speed_limit = (120/3.6 - 50/3.6) * curvature + 50/3.6

        #     # discretize the continuous values of corrected_speed_limit
        #     if corrected_speed_limit < 65/3.6:
        #         corrected_speed_limit = 50/3.6
        #     elif corrected_speed_limit < 90/3.6:
        #         corrected_speed_limit = 80/3.6
        #     elif corrected_speed_limit < 110/3.6:
        #         corrected_speed_limit = 100/3.6
        #     else:
        #         corrected_speed_limit = 120/3.6

        #     actual_speed_limit = self.speed_limits[i]

        #     # if curvature < 0.5 and actual_speed_limit > corrected_speed_limit + 10:
        #     braking_distance = int(1.1 * (((actual_speed_limit * 3.6) / 10.0) ** 2 / 2.0) - (((corrected_speed_limit * 3.6) / 10.0) ** 2 / 2.0))
        #     for j in range(i, max(-1, i-braking_distance-1), -1):
        #         self.curvature_dependent_speed[j] = min(corrected_speed_limit, self.speed_limits[j])

    def compute_leading_vehicles(self, list_vehicles, ego_vehicle_id):
        vehicle_ids = np.array([vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])

        if len(vehicle_ids) and self.route_idx != self.route_np.shape[0]:
            vehicle_yaws = np.array([vehicle.get_transform().rotation.yaw for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = [vehicle.get_location() for vehicle in list_vehicles if vehicle.id != ego_vehicle_id]
            vehicle_locations = np.array([[loc.x, loc.y, loc.z] for loc in vehicle_locations])

            # compute vehicles as leading vehicles up to 80m ahead
            distances = vehicle_locations[:, None, :2] - self.route_np[None, self.route_idx:self.route_idx+800, :2][:, ::10, :]
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[self.route_idx:self.route_idx+800][::10]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)
            leading_vehicle_ids = vehicle_ids[(distances<3) & (yaw_differences<35)]

            return leading_vehicle_ids.tolist()
        else:
            return []
    
    def compute_vehicles_behind(self, list_vehicles, ego_vehicle_id):
        vehicle_ids = np.array([vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])

        if len(vehicle_ids) and self.route_idx != 0:
            vehicle_yaws = np.array([vehicle.get_transform().rotation.yaw for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = [vehicle.get_location() for vehicle in list_vehicles if vehicle.id != ego_vehicle_id]
            vehicle_locations = np.array([[loc.x, loc.y, loc.z] for loc in vehicle_locations])

            # compute vehicles as leading vehicles up to 80m ahead
            from_idx = max(0, self.route_idx-800)
            distances = vehicle_locations[:, None, :2] - self.route_np[None, from_idx:self.route_idx, :2][:, ::10, :]
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[from_idx:self.route_idx][::10]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)
            vehicles_behind_ids = vehicle_ids[(distances<3) & (yaw_differences<30)]

            return vehicles_behind_ids.tolist()
        else:
            return []