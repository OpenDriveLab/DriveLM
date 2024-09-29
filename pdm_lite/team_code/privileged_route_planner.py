"""
This Python script defines a PrivilegedRoutePlanner class for planning and modifying routes in the CARLA 
simulation environment. The class provides functionalities to create a smooth and interpolated route,
compute distances to traffic lights and stop signs, handle lane changes, and identify leading and trailing vehicles.
"""

import numpy as np
import carla
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from agents.navigation.local_planner import RoadOption


class PrivilegedRoutePlanner(object):
    """
    This class is used the experts navigation and provides not only the route but preprocesses it and provides useful
    information like the next stop sign and the next traffic light.
    """

    def __init__(self, config):
        """
        Initialize the RoutePlanner object.

        Args:
            config (GlobalConfig): Object of the config for hyperparameters.
        """

        self.config = config

        self.points_per_meter = self.config.points_per_meter
        self.ego_vehicles_route_point_search_distance = self.config.ego_vehicles_route_point_search_distance
        self.lane_shift_extension_length_for_yield_to_emergency_vehicle = \
            self.config.lane_shift_extension_length_for_yield_to_emergency_vehicle
        self.transition_smoothness_distance = self.config.transition_smoothness_distance
        self.route_shift_start_distance_invading_turn = self.config.route_shift_start_distance_invading_turn
        self.route_shift_end_distance_invading_turn = self.config.route_shift_end_distance_invading_turn
        self.fence_avoidance_margin_invading_turn = self.config.fence_avoidance_margin_invading_turn
        self.minimum_lane_width_threshold = self.config.minimum_lane_width_threshold
        self.speed_limit_waypoints_spacing_check = self.config.speed_limit_waypoints_spacing_check
        self.leading_vehicles_max_route_distance = self.config.leading_vehicles_max_route_distance
        self.leading_vehicles_max_route_angle_distance = self.config.leading_vehicles_max_route_angle_distance
        self.leading_vehicles_maximum_detection_radius = self.config.leading_vehicles_maximum_detection_radius
        self.trailing_vehicles_max_route_distance = self.config.trailing_vehicles_max_route_distance
        self.trailing_vehicles_max_route_distance_lane_change = \
            self.config.trailing_vehicles_max_route_distance_lane_change
        self.tailing_vehicles_maximum_detection_radius = self.config.tailing_vehicles_maximum_detection_radius
        self.max_distance_lane_change_trailing_vehicles = self.config.max_distance_lane_change_trailing_vehicles
        self.extra_route_length = self.config.extra_route_length

        self.route_waypoints = []
        self.route_points = np.array([[]])
        self.original_route_points = np.array([[]])
        self.commands = []
        self.rotation_angles = []

        self.distances_to_next_stop_signs = np.array([])
        self.next_stop_signs = []

        self.distances_to_next_traffic_lights = np.array([])
        self.next_traffic_lights = []

        self.speed_limits = np.array([])

        self.route_index = 0
        self.last_route_index = 0

    def save(self):
        """
        Save the current route index location, which could be saved before forecasting the ego vehicle.
        """
        self.last_route_index = self.route_index

    def load(self):
        """
        Load the previously saved route index location, which could be used for forecasting the ego vehicle.
        """
        self.route_index = self.last_route_index

    def run_step(self, agent_position):
        """
        Update the route index based on the agent's current position and retrieve relevant information at that index.

        Args:
            agent_position (numpy.ndarray): Current location of the agent.
        """
        till = self.ego_vehicles_route_point_search_distance
        search_range = min(self.route_index + till, self.route_points.shape[0])

        # Find the index of the nearest route point to the agent's position
        self.route_index += np.argmin(np.linalg.norm(agent_position[None, :2] - \
                                                    self.route_points[self.route_index:search_range, :2], axis=1))

        return (self.route_points[self.route_index:], self.route_waypoints[self.route_index:],
                self.commands[self.route_index:], self.distances_to_next_traffic_lights[self.route_index],
                self.next_traffic_lights[self.route_index], self.distances_to_next_stop_signs[self.route_index],
                self.next_stop_signs[self.route_index], self.speed_limits[self.route_index])

    def extend_lane_shift_transition_for_yield_to_emergency_vehicle(self, shift_to_left_lane, previous_shift_end_index):
        """
        Extend the lane shift transition to yield to an emergency vehicle.

        Args:
            shift_to_left_lane (bool): Whether to route was initially shifted to the left lane.
            previous_shift_end_index (int): The index of the route waypoint where the initial lane shift started.

        Returns:
            int: The index of the route waypoint after which the extended lane shift transition is complete.
        """
        # Calculate the end index for the extended lane shift transition
        end_shift_index = self.route_index + self.lane_shift_extension_length_for_yield_to_emergency_vehicle
        transition_start_index = previous_shift_end_index
        transition_end_index = max(end_shift_index + 2 * self.transition_smoothness_distance,
                                   previous_shift_end_index + self.transition_smoothness_distance)

        # Extend the lane shift transition smoothly
        for idx in range(transition_start_index, transition_end_index):
            # Calculate the transition factor for smooth shifting and the commands
            transition_factor = 1.
            if transition_end_index - idx < self.transition_smoothness_distance:
                transition_factor = self._smooth_transition(
                    float(transition_end_index - idx) / self.transition_smoothness_distance)
                self.commands[idx] = RoadOption.CHANGELANERIGHT if shift_to_left_lane else RoadOption.CHANGELANELEFT
            else:
                self.commands[idx] = self.commands_orig[idx]

            # Update the route points with the shifted lane location
            target_lane = self.route_waypoints[idx].get_left_lane(
            ) if shift_to_left_lane else self.route_waypoints[idx].get_right_lane()
            if target_lane is None:
                target_lane = self.route_waypoints[idx]

            target_lane_location = target_lane.transform.location
            target_lane_location = np.array([target_lane_location.x, target_lane_location.y, target_lane_location.z])

            self.route_points[idx] = transition_factor * target_lane_location + (
                1. - transition_factor) * self.original_route_points[idx]

        return transition_end_index - self.transition_smoothness_distance

    def extend_lane_shift_transition_for_hazard_at_side_lane(self, last_bicycle, previous_shift_end_index):
        """
        Extend the lane shift transition to ensure the vehicle can safely pass the bicycles in HazardAtSideLane.

        Args:
            last_bicycle (carla.Actor): The actor representing the side obstacle.
            previous_shift_end_index (int): The index of the route waypoint where the previous lane shift ended.

        Returns:
            int: The index of the route waypoint after which the extended lane shift transition is complete.
        """
        # Find the closest route index to the bicycle
        obstacle_route_index = self.get_closest_route_index(self.route_index, last_bicycle.get_location())

        # Calculate the extent of the bicycle
        bicycle_extent = last_bicycle.bounding_box.extent.x

        # Calculate the start and end indices for the extended lane shift transition
        transition_start_index = previous_shift_end_index
        transition_end_index = max(
            obstacle_route_index + int(self.points_per_meter * bicycle_extent) +
            2 * self.transition_smoothness_distance, previous_shift_end_index + self.transition_smoothness_distance)

        # Extend the lane shift transition smoothly
        for idx in range(transition_start_index, transition_end_index):
            # Calculate the transition factor for smooth shifting
            transition_factor = 1.
            if transition_end_index - idx < self.transition_smoothness_distance:
                transition_factor = self._smooth_transition(
                    float(transition_end_index - idx) / self.transition_smoothness_distance)
                self.commands[idx] = RoadOption.CHANGELANERIGHT
            else:
                self.commands[idx] = self.commands_orig[idx]

            # Update the route points with the shifted lane location
            target_lane = self.route_waypoints[idx].get_left_lane()
            if target_lane is None:
                target_lane = self.route_waypoints[idx]

            target_lane_location = target_lane.transform.location
            target_lane_location = np.array([target_lane_location.x, target_lane_location.y, target_lane_location.z])

            self.route_points[idx] = transition_factor * target_lane_location + (
                1.0 - transition_factor) * self.original_route_points[idx]

        return transition_end_index - self.transition_smoothness_distance

    def _smooth_transition(self, value):
        """
        Transforms the linear transition between 0 to 1 into a cosine one.

        Args:
            value (float): The input value between 0 and 1.

        Returns:
            float: The smoothed transition value between 0 and 1.
        """

        smoothed_value = -np.cos(value * np.pi) / 2.0 + 0.5
        return smoothed_value

    def shift_route_smoothly(self,
                             start_index,
                             end_index,
                             shift_to_left_lane,
                             transition_length=120.,
                             lane_transition_factor=1.):
        """
        Shift the route smoothly to the left or right lane between the specified start and end indices.

        Args:
            start_index (int): The index of the route waypoint where the shift should start.
            end_index (int): The index of the route waypoint where the shift should end.
            shift_to_left_lane (bool): Whether to shift the route to the left lane.
            transition_length (int): The length of the transition in waypoints.
            lane_transition_factor (float): A factor between 0 and 1 that controls the amount of shift towards 
                                                the neighboring lane.
                                            A value of 1.0 means the route will be shifted to the center of 
                                                the neighboring lane,
                                            while a value of 0.0 means the route will stay in the center of 
                                                the current lane.
        """
        for idx in range(start_index, end_index):
            # Get the location of the left / right center lane
            if shift_to_left_lane:
                loc = self.route_waypoints[idx].get_left_lane()
            else:
                loc = self.route_waypoints[idx].get_right_lane()

            loc = self.route_waypoints[idx].transform.location if loc is None else loc.transform.location
            loc = np.array([loc.x, loc.y, loc.z])

            # Calculate the new commands and the transition factor, which controls the smooth transition
            # to the center of the neighbor lane
            transition_factor = 1.
            if idx <= start_index + transition_length and idx - start_index < end_index - idx:
                transition_factor = self._smooth_transition(float(idx - start_index) / transition_length)
                if shift_to_left_lane:
                    self.commands[idx] = RoadOption.CHANGELANELEFT
                else:
                    self.commands[idx] = RoadOption.CHANGELANERIGHT
            elif idx >= end_index - transition_length:
                transition_factor = self._smooth_transition(float(end_index - idx) / transition_length)
                if shift_to_left_lane:
                    self.commands[idx] = RoadOption.CHANGELANERIGHT
                else:
                    self.commands[idx] = RoadOption.CHANGELANELEFT

            # The actual route shift
            self.route_points[idx] = lane_transition_factor * transition_factor * loc + (
                1. - lane_transition_factor * transition_factor) * self.route_points[idx]

    def get_closest_route_index(self, begin_idx, location):
        """
        Finds the index of the closest route point to a given location using gradient descent with constant gradient.

        Args:
            begin_idx (int): Starting index for the search.
            location (carla.Location): Location for which the closest route point is to be found.

        Returns:
            int: Index of the closest route point.
        """
        index = begin_idx
        location_np = np.array([location.x, location.y])

        # calculate the search direction
        direction = 1
        if np.linalg.norm(location_np -
                          self.original_route_points[index, :2]) < np.linalg.norm(location_np -
                                                                                  self.original_route_points[index +
                                                                                                             1, :2]):
            direction = -1

        # The following is like a gradient descent with a constant gradient.
        while True:
            # check if we have reached the first or last route point
            if index + direction == 0 or index + direction == self.original_route_points.shape[0]:
                return index

            dist1 = np.linalg.norm(location_np - self.original_route_points[index, :2])
            dist2 = np.linalg.norm(location_np - self.original_route_points[index + direction, :2])
            # check if we have found the closest route point
            if dist1 < dist2:
                return index

            index += direction

    def shift_route_for_invading_turn(self, first_cone, last_cone, lateral_offset):
        """
        Shift the route laterally to overcome the InvadingTurn scenario.

        Args:
            first_cone (carla.Actor): The first cone marking the start of the invading turn scenario.
            last_cone (carla.Actor): The last cone marking the end of the invading turn scenario.
            lateral_offset (float): The lateral offset distance (in meters) to shift the route.
        """
        # Find the route indices corresponding to the first and last cones
        first_cone_index = self.get_closest_route_index(self.route_index, first_cone.get_location())
        # Begin 10 meters for the search after the first cone.
        last_cone_index = self.get_closest_route_index(first_cone_index + 10 * self.points_per_meter,
                                                       last_cone.get_location())

        # Adjust the route by shifting it laterally between 15m before the first cone and 10m after the last cone
        for idx in range(first_cone_index - self.route_shift_start_distance_invading_turn,
                         last_cone_index + self.route_shift_end_distance_invading_turn):
            shift_vector = self.route_points[idx + 1, :2] - self.route_points[idx, :2]
            shift_vector = np.array([[0, -1], [1, 0]]) @ shift_vector

            # Adjust the lateral offset if the route cannot be shifted due to a fence
            adjusted_offset = lateral_offset
            right_lane = self.route_waypoints[idx].get_right_lane()
            if right_lane is not None and right_lane.lane_type == carla.LaneType.Shoulder:
                adjusted_offset = min(
                    lateral_offset - np.sign(lateral_offset) * self.fence_avoidance_margin_invading_turn,
                    right_lane.lane_width)

            shift_vector = shift_vector / np.linalg.norm(shift_vector) * np.abs(adjusted_offset)
            self.route_points[idx, :2] += shift_vector

    def shift_route_around_actors(self,
                                  first_actor,
                                  last_actor=None,
                                  obstacle_direction="right",
                                  transition_length=120.,
                                  lane_transition_factor=1.,
                                  extra_length_before=0.,
                                  extra_length_after=0.):
        """
        Shift the route smoothly to the left or right lane around the specified actors.

        Args:
            first_actor (carla.Actor): The first actor around which the route should be shifted.
            last_actor (carla.Actor, optional): The last actor around which the route should be shifted. If None,
                        the shift will end after a certain distance from the first actor.
            obstacle_direction (str): The direction in which the obstacle is located. If it is to the left, we shift
                        the route to the right and vice versa.
            transition_length (int): The length of the transition in waypoints.
            lane_transition_factor (float): A factor between 0 and 1 that controls the amount of shift 
                                                towards the neighboring lane.
                                            A value of 1.0 means the route will be shifted to the center of 
                                                the neighboring lane,
                                            while a value of 0.0 means the route will stay in the center of 
                                                the current lane.
            extra_length_before (float): Additional length (in meters) to be added before the first actor for the shift.
            extra_length_after (float): Additional length (in meters) to be added after the last actor for the shift.

        Returns:
            tuple: A tuple containing the start and end indices of the route shift.
        """
        # Find the closest route index to the first actor
        tree = cKDTree(self.original_route_points[self.route_index:, :2])
        first_actor_location = np.array([first_actor.get_location().x, first_actor.get_location().y])
        _, closest_idx = tree.query(first_actor_location, k=1)
        first_idx = closest_idx + self.route_index

        # Calculate where we the route shift starts
        first_actor_extent = first_actor.bounding_box.extent.x
        shift_start_index = first_idx - int(first_actor_extent * self.points_per_meter + transition_length +
                                            extra_length_before)

        # Calculate where we the route shift ends
        if last_actor is None:
            shift_end_index = first_idx + int(first_actor_extent * self.points_per_meter + transition_length +
                                              extra_length_after)
        else:
            last_actor_location = last_actor.get_location()
            last_idx = self.get_closest_route_index(first_idx, last_actor_location)
            last_actor_extent = last_actor.bounding_box.extent.x
            shift_end_index = last_idx + int(last_actor_extent * self.points_per_meter + transition_length +
                                             extra_length_after)

        # Determine the shift direction
        shift_to_left_lane = True if obstacle_direction == "right" else False

        # Shift the route smoothly
        self.shift_route_smoothly(shift_start_index,
                                  shift_end_index,
                                  shift_to_left_lane,
                                  transition_length=transition_length,
                                  lane_transition_factor=lane_transition_factor)

        return shift_start_index, shift_end_index

    def setup_route(self, global_plan, carla_world, carla_map, starts_with_parking_exit, vehicle_loc):
        """
        Set up the route for the autonomous vehicle based on the given global plan.

        Args:
            global_plan (list): A list of (carla.Transform, carla.RoadOption) tuples representing the global plan.
            carla_world (carla.World): The CARLA world object.
            carla_map (carla.Map): The CARLA map object.
            starts_with_parking_exit (bool): A flag indicating if the route starts with a parking exit scenario.
            vehicle_location (carla.Location): The initial location of the vehicle.
        """
        self.route_index = self.extra_route_length * self.points_per_meter
        self.last_route_index = self.route_index

        # Get all waypoint objects of the route and add extra waypoints at the end
        # to ensure the vehicle completes the route properly and avoids unexpected side effects
        route_waypoints = [transform.location for transform, _ in global_plan]
        route_waypoints = [carla_map.get_waypoint(loc) for loc in route_waypoints]
        cmds = [cmd for _, cmd in global_plan]

        # Handle the case where the route starts with a parking exit scenario
        # In this case the firstt wp is on the center of the road, not the parking lot,
        # where the agent starts
        if starts_with_parking_exit:  # workaraound for ParkingExit scenario
            self.route_index = 0
            self.last_route_index = 0

            cmds.insert(0, RoadOption.CHANGELANELEFT)
            route_waypoints.insert(0, carla_map.get_waypoint(vehicle_loc))
        else:
            # Add extra waypoints at the beginning of the route
            for _ in range(self.extra_route_length):
                prev_wps = route_waypoints[0].previous(1)
                if len(prev_wps) == 0:
                    break
                route_waypoints.insert(0, prev_wps[0])
                cmds.insert(0, RoadOption.LANEFOLLOW)
                self.route_index += 1
                self.last_route_index += 1

        # Add extra waypoints at the end of the route
        for _ in range(self.extra_route_length):
            next_wps = route_waypoints[-1].next(1)
            if len(next_wps) == 0:
                break

            route_waypoints.append(next_wps[0])
            cmds.append(RoadOption.LANEFOLLOW)

        # Generate a numpy array containing the route locations
        route_points = [wp.transform.location for wp in route_waypoints]
        route_points = np.array([[loc.x, loc.y, loc.z] for loc in route_points])

        # Smooth and interpolate the route
        self.route_points, self.commands = self.smooth_and_supersample(route_points, cmds)
        self.original_route_points = np.copy(self.route_points)
        self.commands_orig = self.commands.copy()

        # Get the waypoint objects for the route points
        self.route_waypoints = []
        for route_loc in self.route_points:
            wp = carla_map.get_waypoint(carla.Location(x=route_loc[0], y=route_loc[1], z=route_loc[2]))
            self.route_waypoints.append(wp)

        self.compute_route_info(carla_world, carla_map)

    def compute_rotation_angles(self, route_points):
        """
        Computes the yaw angles corresponding to the ego vehicle's orientation at individual route points in degrees.

        Args:
            route_points (numpy.ndarray): Array containing the route points.

        Returns:
            numpy.ndarray: Array containing the yaw angles at each route point.
        """

        # Compute differences between consecutive route points
        indices = np.arange(1, route_points.shape[0] - 1)
        differences = route_points[indices + 1] - route_points[indices - 1]

        # Compute yaw angles in degrees
        yaws = np.arctan2(differences[:, 1], differences[:, 0]) * 180. / np.pi

        # Add first and last yaw angles to maintain array length
        yaws = np.concatenate([[yaws[0]], yaws, [yaws[-1]]])

        return yaws

    def smooth_and_supersample(self, original_route_points, commands):
        """
        Smooths and supersamples the given route to increase density and matches commands accordingly.

        Args:
            original_route_points (numpy.ndarray): Array containing the original route points.
            commands (list): List of commands corresponding to the route points.

        Returns:
            tuple: A tuple containing the smoothed and supersampled route points, and the updated commands.
        """

        num_supersample_per_point = 10  # sample x points per number of route points for later
        # number of points to interpolate betweeneach pair of original points
        num_samples = self.points_per_meter * num_supersample_per_point
        segment_length = 1. / self.points_per_meter  # Length of segments along the smoothed route
        num_original_points = original_route_points.shape[0]

        # Create interpolation functions for each dimension
        interp_x = interp1d(np.arange(num_original_points), original_route_points[:, 0])
        interp_y = interp1d(np.arange(num_original_points), original_route_points[:, 1])
        interp_z = interp1d(np.arange(num_original_points), original_route_points[:, 2])

        # Interpolate points along the original route
        x_supersampled = interp_x(np.arange(0, num_original_points - 1, 1. / num_samples))
        y_supersampled = interp_y(np.arange(0, num_original_points - 1, 1. / num_samples))
        z_supersampled = interp_z(np.arange(0, num_original_points - 1, 1. / num_samples))

        route_supersampled = np.column_stack([x_supersampled, y_supersampled, z_supersampled])

        # Calculate cumulative distances along the supersampled route
        cumulative_distances = np.cumsum(np.linalg.norm(np.diff(route_supersampled, axis=0), axis=1))
        cumulative_distances = np.insert(cumulative_distances, 0, 0)
        cumulative_distances = cumulative_distances % segment_length

        # Find indices of points at segment boundaries
        segment_indices = np.insert(np.argwhere(cumulative_distances[1:] < cumulative_distances[:-1]), 0, 0)
        smoothed_points = route_supersampled[segment_indices]

        # Interpolate commands for the smoothed points
        num_original_commands = len(commands)
        command_indices = np.minimum(
            np.round(segment_indices.astype("float") / self.points_per_meter / num_supersample_per_point),
            num_original_commands - 1).astype("int")
        smoothed_commands = np.array([commands[idx] for idx in command_indices])

        return smoothed_points, smoothed_commands

    def compute_route_info(self, carla_world, carla_map):
        """
        Computes additional information for the route such as distances to traffic lights and stop signs,
        speed limits, and prevents too early lane changes and computes yaw angles corresponding to the ego 
        vehicle's orientation at individual route points in degrees.

        Args:
            carla_world: Carla world instance.
            carla_map: Carla map instance.
        """
        self.rotation_angles = self.compute_rotation_angles(self.route_points)
        self.compute_distances_to_traffic_lights(carla_world)
        self.compute_distances_to_stop_signs(carla_world, carla_map)
        self.compute_speed_limits(carla_map)
        self.prevent_too_early_lane_changes()

    def prevent_too_early_lane_changes(self):
        """
        Prevents too early lane changes by ensuring that the agent continues on the previous lane for a bit longer
        in case the lane is too narrow.
        """
        lane_threshold = self.minimum_lane_width_threshold

        # Iterate over route waypoints
        for i in range(len(self.route_waypoints) - 2):
            # Check that we have not reached the last waypoint and the lane width increases
            if self.route_waypoints[i + 1].lane_width < lane_threshold and self.route_waypoints[
                    i + 2].lane_width < lane_threshold and self.route_waypoints[
                        i + 1].lane_width < self.route_waypoints[i + 2].lane_width:
                j = i + 1
                to_left = self.commands[i] == RoadOption.CHANGELANELEFT

                # Continue on the previous lane until it's wide enough
                while True:
                    if j == len(self.route_waypoints) or self.route_waypoints[j].lane_width >= lane_threshold:
                        break

                    # Get the waypoint of the previous lane
                    wp = self.route_waypoints[j].get_right_lane() if to_left \
                                                                        else self.route_waypoints[j].get_left_lane()
                    wp = self.route_waypoints[j] if wp is None else wp

                    # Update route waypoints and points
                    self.route_waypoints[j] = wp
                    self.route_points[j] = np.array(
                        [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z])
                    self.original_route_points[j] = np.array(
                        [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z])
                    j += 1

    def compute_distances_to_traffic_lights(self, carla_world):
        """
        Compute the distance to the next traffic light from each individual route location.

        Args:
            carla_world: Carla world instance.
        """
        # Initialize arrays to store distances and next traffic lights
        self.distances_to_next_traffic_lights = np.full(self.route_points.shape[0], np.inf)
        self.next_traffic_lights = [None] * self.route_points.shape[0]

        # Initialize variables
        next_traffic_light = None
        traffic_light_already_recorded = False
        distance_idx = np.inf

        # Iterate over route points in reverse order
        for i in range(len(self.route_points) - 1, -1, -1):
            waypoint = self.route_waypoints[i]
            traffic_lights = carla_world.get_traffic_lights_from_waypoint(waypoint, 5)

            # Check if the found traffic light was already recorded in the past
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

            # Update arrays with distance and next traffic light
            self.next_traffic_lights[i] = next_traffic_light
            self.distances_to_next_traffic_lights[i] = float(distance_idx) / self.points_per_meter

        # Since we search for traffic lights up to 5m away, we have to shift the arrays
        self.distances_to_next_traffic_lights = np.concatenate(
            [self.distances_to_next_traffic_lights[:-40], 40 * [np.inf]])
        self.next_traffic_lights = self.next_traffic_lights[:-40] + (40 * [None])

    def compute_distances_to_stop_signs(self, carla_world, carla_map):
        """
        Compute the distance to the next stop sign from each individual route location.
        We use the official implementation that is used to test whether we ran a stop sign
        The logic is copied from the class RunningStopTest in 
        scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria

        Args:
            carla_world: Carla world instance.
            carla_map: Carla map instance.
        """

        def point_inside_boundingbox(point, bb_center, bb_extent, multiplier=1.2):
            """Checks whether or not a point is inside a bounding box."""

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

        def is_actor_affected_by_stop(wp_list, stop_extent, stop_location):
            """
            Check if the given actor is affected by the stop.
            Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
            """

            # Quick distance test
            actor_location = wp_list[0].transform.location
            if stop_location.distance(actor_location) > 4.0:
                return False

            # Check if the any of the actor wps is inside the stop's bounding box.
            # Using more than one waypoint removes issues with small trigger volumes and backwards movement
            for actor_wp in wp_list:
                if point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
                    return True

            return False

        def _scan_for_stop_sign(list_stop_signs, list_stop_signs_extent, wp_list, stop_locations):
            """Check which stop sign affects the actor."""
            for (stop, stop_extent, stop_location) in zip(list_stop_signs, list_stop_signs_extent, stop_locations):
                if is_actor_affected_by_stop(wp_list, stop_extent, stop_location):
                    return stop

            return None

        def _get_waypoints(start_loc, carla_map):
            """Returns a list of waypoints starting from the ego location and a set amount forward"""
            wp_list = []
            steps = int(4.0 / 0.5)

            # Add the actor location
            wp = carla_map.get_waypoint(start_loc)
            wp_list.append(wp)

            # And its forward waypoints
            next_wp = wp
            for _ in range(steps):
                next_wps = next_wp.next(0.5)
                if not next_wps:
                    break
                next_wp = next_wps[0]
                wp_list.append(next_wp)

            return wp_list

        # Initialize arrays to store distances and next stop signs
        self.distances_to_next_stop_signs = np.full(self.route_points.shape[0], np.inf, dtype=np.float)
        self.next_stop_signs = [None] * self.route_points.shape[0]

        # Get list of all stop signs
        list_stop_signs = carla_world.get_actors().filter("*traffic.stop*")

        next_stop_signs = None
        distance_idx = np.inf

        if list_stop_signs:
            list_stop_signs_extent = [x.trigger_volume.extent for x in list_stop_signs]

            # Adjust minimum extent for stop signs. That is necessary, since some stop signs are only 2cm thick
            # and because we use waypoints 50 cm apart it's likely we would miss it
            for extent in list_stop_signs_extent:
                extent.x = max(extent.x, 1)
                extent.y = max(extent.y, 1)

            stop_locations = [stop.get_transform().transform(stop.trigger_volume.location) for stop in list_stop_signs]
            stop_locations_np = np.array([[x.x, x.y, x.z] for x in stop_locations])

            for i in range(self.route_points.shape[0]):
                loc = self.route_points[i]
                stop_sign = None

                # Quick distance check to safe computation later
                if np.linalg.norm(loc[None] - stop_locations_np, axis=1).min() < 4:
                    start_loc = carla.Location(x=loc[0], y=loc[1], z=loc[2])
                    check_wps = _get_waypoints(start_loc, carla_map)
                    stop_sign = _scan_for_stop_sign(list_stop_signs, list_stop_signs_extent, check_wps, stop_locations)
                self.next_stop_signs[i] = stop_sign

            # Compute distances to next stop signs
            for i in range(self.distances_to_next_stop_signs.shape[0] - 1, -1, -1):
                if self.next_stop_signs[i] is not None:
                    next_stop_signs = self.next_stop_signs[i]
                    distance_idx = 0
                else:
                    distance_idx += 1

                self.next_stop_signs[i] = next_stop_signs
                self.distances_to_next_stop_signs[i] = float(distance_idx) / self.points_per_meter

    def compute_speed_limits(self, carla_map):
        """
        Compute speed limits for each location along the route.

        Args:
            carla_map: Carla map instance.
        """
        # Get the name of the map
        map_name = carla_map.name.split("/")[-1]

        # Load speed limit data from file
        file_name_speed_limits = f"team_code/speed_limits/{map_name}_speed_limits.npy"
        file_content = np.load(file_name_speed_limits, allow_pickle=True)
        map_data = file_content.item()

        # Use cKDTree for fast nearest neighbor search
        map_locations = map_data.get("locations")
        map_speed_limits = map_data.get("speed_limits")
        tree = cKDTree(map_locations)

        # Initialize array to store speed limits
        self.speed_limits = np.empty(self.route_points.shape[0], dtype=np.float)
        previous_speed_limit = -1

        # Iterate through route locations
        for i, loc in enumerate(self.route_points):
            if i % self.speed_limit_waypoints_spacing_check == 0:  # Calculate for waypoints every 5 m for efficiency
                _, min_idx = tree.query(loc, k=1)
                speed_limit = map_speed_limits[min_idx] / 3.6  # Convert speed from km/h to m/s
                self.speed_limits[i] = speed_limit
                previous_speed_limit = speed_limit
            else:
                self.speed_limits[i] = previous_speed_limit

    def compute_leading_vehicles(self, list_vehicles, ego_vehicle_id):
        """
        Computes the IDs of vehicles leading ahead of the ego vehicle.

        Args:
            list_vehicles (list): List of all vehicles.
            ego_vehicle_id (int): ID of the ego vehicle.

        Returns:
            list: IDs of vehicles leading ahead of the ego vehicle.
        """
        # Get IDs of all vehicles except the ego vehicle
        vehicle_ids = np.array([vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])

        # Check if there are vehicles and the route index is not at the end
        if len(vehicle_ids) and self.route_index != self.route_points.shape[0]:
            max_distance = self.leading_vehicles_maximum_detection_radius

            vehicle_yaws = np.array(
                [vehicle.get_transform().rotation.yaw for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = [vehicle.get_location() for vehicle in list_vehicles if vehicle.id != ego_vehicle_id]
            vehicle_locations = np.array([[loc.x, loc.y, loc.z] for loc in vehicle_locations])

            # Compute leading vehicles up to 80m ahead
            # Computes if vehicle is leading ahead of the ego vehicle and its orientation is closer than
            # 35 degrees to the road
            # Both is necessary to ensure it is leading ahead of the ego vehicle and is not only crossing
            # its future path
            distances = vehicle_locations[:,
                                          None, :2] - self.route_points[None, self.route_index:self.route_index +
                                                                        max_distance, :2][:, ::self.points_per_meter, :]
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[self.route_index:self.route_index +
                                                   max_distance][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)

            # Define the maximum distance and yaw difference thresholds
            max_distance = self.leading_vehicles_max_route_distance
            max_yaw_difference = self.leading_vehicles_max_route_angle_distance

            # Usually the road is 3.5 m wide, but in case of ParkingCrossingPedestrian it's less
            leading_vehicle_ids = vehicle_ids[(distances < max_distance) & (yaw_differences < max_yaw_difference)]

            return leading_vehicle_ids.tolist()
        else:
            return []

    def compute_trailing_vehicles(self, list_vehicles, ego_vehicle_id):
        """
        Computes the IDs of vehicles trailing behind the ego vehicle.

        Args:
            list_vehicles (list): List of all vehicles.
            ego_vehicle_id (int): ID of the ego vehicle.

        Returns:
            list: IDs of vehicles trailing behind the ego vehicle
        """
        # Get IDs of all vehicles except the ego vehicle
        vehicle_ids = np.array([vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])

        # Maximum distance of vehicles to ego's route
        max_distance = self.trailing_vehicles_max_route_distance

        # Check if there was a lane change in the past
        max_distance_lane_change = self.max_distance_lane_change_trailing_vehicles
        for i in range(max(0, self.route_index - max_distance_lane_change), self.route_index):
            if self.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                max_distance = self.trailing_vehicles_max_route_distance_lane_change
                break

        # Check if there are vehicles and the route index is not at the beginning
        if len(vehicle_ids) and self.route_index != 0:
            # Get yaw angles and locations of non-ego vehicles
            vehicle_yaws = np.array(
                [vehicle.get_transform().rotation.yaw for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = [vehicle.get_location() for vehicle in list_vehicles if vehicle.id != ego_vehicle_id]
            vehicle_locations = np.array([[loc.x, loc.y, loc.z] for loc in vehicle_locations])

            max_distance_trailing_vehicles = self.tailing_vehicles_maximum_detection_radius
            # Computes if vehicle is behind ego vehicle and its orientation is closer than 30 degrees to the road
            # Both is necessary to ensure it is trailing the ego vehicle and is not only crossing its previous path
            from_idx = max(0, self.route_index - max_distance_trailing_vehicles)
            distances = vehicle_locations[:, None, :2] - self.route_points[
                None, from_idx:self.route_index, :2][:, ::self.points_per_meter, :]
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[from_idx:self.route_index][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)
            vehicles_behind_ids = vehicle_ids[(distances < max_distance) & (yaw_differences < 30)]

            return vehicles_behind_ids.tolist()
        else:
            return []
