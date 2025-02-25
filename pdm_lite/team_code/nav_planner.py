"""
Some helpful classes for planning and control for the privileged autopilot
"""

import math
from copy import deepcopy
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner


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
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self.k_p * error + self.k_i * integral + self.k_d * derivative

    def save(self):
        self._saved_window = deepcopy(self._window)

    def load(self):
        self._window = self._saved_window


class LateralPIDController(object):
    """
    PID controller
    """

    def __init__(self, config):
        self.k_p = config.lateral_k_p
        self.k_d = config.lateral_k_d
        self.k_i = config.lateral_k_i
        self.speed_scale = config.lateral_speed_scale
        self.speed_offset = config.lateral_speed_offset
        self.default_lookahead = config.lateral_default_lookahead
        self.speed_threshold = config.lateral_speed_threshold
        self.n = config.lateral_n

        self._saved_window = []
        self._window = []

    def step(self, route_np, current_speed, ego_vehicle_location, ego_vehicle_rotation):
        current_speed = current_speed * 3.6
        # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
        n_lookahead = (
            np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
        )  # range [2.4, 10.5]
        n_lookahead = n_lookahead - 2  # range [0.4, 8.5]
        n_lookahead = int(
            min(n_lookahead, route_np.shape[0] - 1)
        )  # range [0, 8] - but 0 and 1 are never used because n_lookahead is overwritten below

        n_lookahead = min(n_lookahead, len(route_np) - 1)
        desired_heading_vec = route_np[n_lookahead] - ego_vehicle_location

        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = (yaw_path - ego_vehicle_rotation) % (2 * np.pi)
        heading_error = (
            heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        )

        # the scaling doesn't deserve any specific purpose but is a leftover from a previous less efficient implementation,
        # on which we optimized the parameters
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(heading_error)
        self._window = self._window[-self.n :]

        derivative = (
            0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        )
        integral = np.mean(self._window)

        steering = np.clip(
            self.k_p * heading_error + self.k_d * derivative + self.k_i * integral,
            -1.0,
            1.0,
        ).item()

        return steering

    def save(self):
        self._saved_window = self._window.copy()

    def load(self):
        self._window = self._saved_window.copy()


def get_throttle(config, brake, target_speed, speed):
    if target_speed < 1e-5 or brake:
        return 0.0, True
    elif target_speed < 1.0 / 3.6:  # to avoid very small target speeds
        target_speed = 1.0 / 3.6

    speed = speed * 3.6
    target_speed = target_speed * 3.6
    params = config.longitudinal_params
    speed_error = target_speed - speed

    # maximum acceleration 1.9 m/tick
    if speed_error > config.longitudinal_max_acceleration:
        return 1.0, False

    if speed / target_speed > params[-1] or brake:
        throttle, control_brake = 0.0, True
        return throttle, control_brake

    speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
    speed /= 100.0
    features = np.array(
        [
            speed,
            speed**2,
            100 * speed_error_cl,
            speed_error_cl**2,
            speed * speed_error_cl,
            speed**2 * speed_error_cl,
        ]
    )

    throttle, control_brake = np.clip(features @ params[:-1], 0.0, 1.0), False

    return throttle, control_brake


class RoutePlanner(object):
    """
    Gets the next waypoint along a path
    """

    def __init__(self, min_distance, max_distance, lat_ref=0.0, lon_ref=0.0):
        self.saved_route = deque()
        self.route = deque()
        self.saved_route_distances = deque()
        self.route_distances = deque()

        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

    def convert_gps_to_carla(self, gps):
        """
        Converts GPS signal into the CARLA coordinate frame
        :param gps: gps from gnss sensor
        :return: gps as numpy array in CARLA coordinates
        """
        EARTH_RADIUS_EQUA = 6378137.0  # Constant from CARLA leaderboard GPS simulation

        lat, lon, _ = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (
            EARTH_RADIUS_EQUA * scale
        )
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = (
            scale
            * EARTH_RADIUS_EQUA
            * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0))
            - my
        )
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        gps = np.array([x, y, gps[2]])

        return gps

    def set_route(self, global_plan, gps=False, carla_map=None):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos["lat"], pos["lon"], pos["z"]])
                pos = self.convert_gps_to_carla(pos)
            else:
                # important to use the z variable, otherwise there are some rare bugs at carla.map.get_waypoint(carla.Location)
                pos = np.array([pos.location.x, pos.location.y, pos.location.z])

            self.route.append((pos, cmd))

        if carla_map is not None:
            for _ in range(50):
                loc = carla.Location(
                    x=self.route[-1][0][0],
                    y=self.route[-1][0][1],
                    z=self.route[-1][0][2],
                )
                next_loc = carla_map.get_waypoint(loc).next(1)[0].transform.location
                next_loc = np.array([next_loc.x, next_loc.y, next_loc.z])
                self.route.append((next_loc, self.route[-1][1]))

        # We do the calculations in the beginning once so that we don't have
        # to do them every time in run_step
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            self.route_distances.append(distance)

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += self.route_distances[i]

            diff = self.route[i][0] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5

            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()

        return self.route

    def save(self):
        # because self.route saves objects of traffic lights and traffic signs a deep copy is not possible
        self.saved_route = []
        for (
            loc,
            cmd,
            d_traffic,
            traffic,
            d_stop,
            stop,
            speed_limit,
            corrected_speed_limit,
        ) in self.route:
            self.saved_route.append(
                (
                    np.copy(loc),
                    cmd,
                    d_traffic,
                    traffic,
                    d_stop,
                    stop,
                    speed_limit,
                    corrected_speed_limit,
                )
            )

        self.saved_route = deque(self.saved_route)
        self.saved_route_distances = deepcopy(self.route_distances)

    def load(self):
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False


def interpolate_trajectory(
    world_map, waypoints_trajectory, hop_resolution=1.0, max_len=400
):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used
    by the user.
    returns the full interpolated route both in GPS coordinates and also in
    its original form.

    Args:
    - world: a reference to the CARLA world so we can use the planner
    - waypoints_trajectory: the current coarse trajectory
    - hop_resolution: is the resolution, how dense is the provided
    trajectory going to be made
    """

    grp = GlobalRoutePlanner(world_map, hop_resolution)
    # Obtain route plan
    lat_ref, lon_ref = _get_latlon_ref(world_map)

    route = []
    gps_route = []

    for i in range(len(waypoints_trajectory) - 1):
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            if len(interpolated_trace) > max_len:
                waypoints_trajectory[i + 1] = waypoints_trajectory[i]
            else:
                for wp, connection in interpolated_trace:
                    route.append((wp.transform, connection))
                    gps_coord = _location_to_gps(
                        lat_ref, lon_ref, wp.transform.location
                    )
                    gps_route.append((gps_coord, connection))

    return gps_route, route


def extrapolate_waypoint_route(waypoint_route, route_points):
    # guard against inplace mutation
    route = deepcopy(waypoint_route)

    # determine length of route before extrapolation
    remaining_waypoints = len(route)

    # we start at the end of the unextrapolated route and move linearly
    heading_vector = route[-1][0] - route[-2][0]
    heading_vector = heading_vector / (np.linalg.norm(heading_vector) + 1.0e-7)

    # we extrapolate 2 meters ahead for each point and skip the first two
    extrapolation = []
    for i in range(2, route_points + 2):
        next_wp = route[-1][0] + i * 2 * heading_vector
        extrapolation.append((next_wp, route[-1][1]))
    route.extend(extrapolation)

    # the waypoint_planner does not pop the last (few) waypoints in its
    # route. We manually pop those when they are the only remaining points
    # in the original route and only pass the extrapolation to the cost.
    if remaining_waypoints == 2:
        route.popleft()
        route.popleft()
    elif remaining_waypoints == 1:
        route.popleft()
    return route


def _get_latlon_ref(world_map):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world_map.to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(" ")
                    for item in str_list:
                        if "+lat_0" in item:
                            lat_ref = float(item.split("=")[1])
                        if "+lon_0" in item:
                            lon_ref = float(item.split("=")[1])
    return lat_ref, lon_ref


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0  # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    )
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {"lat": lat, "lon": lon, "z": z}
