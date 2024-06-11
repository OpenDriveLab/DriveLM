"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.
"""

import os
import ujson
import datetime
import pathlib
import gzip
from collections import deque
from agents.navigation.local_planner import RoadOption
import math
import numpy as np
import carla
from scipy.integrate import RK45

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from nav_planner import RoutePlanner
from lateral_controller import LateralPIDController
from privileged_route_planner import PrivilegedRoutePlanner
from config import GlobalConfig
import transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from longitudinal_controller import LongitudinalLinearRegressionController
from kinematic_bicycle_model import KinematicBicycleModel

def get_entry_point():
    return "AutoPilot"


class AutoPilot(autonomous_agent_local.AutonomousAgent):
    """
      Privileged driving agent used for data collection.
      Drives by accessing the simulator directly.
      """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            config_file_path (str): Path to the configuration file.
            route_index (int, optional): Index of the route to follow.
            traffic_manager (object, optional): The traffic manager object.

        """
        self.recording = False
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.save_path = None
        self.route_index = route_index

        self.datagen = int(os.environ.get("DATAGEN", 0)) == 1

        self.config = GlobalConfig()

        self.speed_histogram = []
        self.make_histogram = int(os.environ.get("HISTOGRAM", 0))

        self.tp_stats = False
        self.tp_sign_agrees_with_angle = []
        if int(os.environ.get("TP_STATS", 0)):
            self.tp_stats = True

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config)
        self.vehicle_model = KinematicBicycleModel(self.config)

        # Configuration
        self.visualize = int(os.environ.get("DEBUG_CHALLENGE", 0))

        self.walker_close = False
        self.distance_to_walker = np.inf
        self.stop_sign_close = False

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.ego_blocked_for_ticks = 0

        # Controllers
        self._turn_controller = LateralPIDController(self.config)

        self.list_traffic_lights = []

        # Navigation command buffer, needed because the correct command comes from the last cleared waypoint
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.next_commands = deque(maxlen=2)
        self.next_commands.append(4)
        self.next_commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = self.config.target_speed_fast

        self.augmentation_translation = 0
        self.augmentation_rotation = 0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.angle = 0.0
        self.stop_sign_hazard = False
        self.traffic_light_hazard = False
        self.walker_hazard = False
        self.vehicle_hazard = False
        self.junction = False
        self.aim_wp = None  # Waypoint the expert is steering towards
        self.remaining_route = None  # Remaining route
        self.remaining_route_original = None  # Remaining original route
        self.close_traffic_lights = []
        self.close_stop_signs = []
        self.was_at_stop_sign = False
        self.cleared_stop_sign = False
        self.visible_walker_ids = []
        self.walker_past_pos = {}  # Position of walker in the last frame

        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # Get the world map and the ego vehicle
        self.world_map = CarlaDataProvider.get_map()

        # Set up the save path if specified
        if os.environ.get("SAVE_PATH", None) is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += f"route{self.route_index}_"
            string += "_".join(map(lambda x: f"{x:02}", (now.month, now.day, now.hour, now.minute, now.second)))

            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            if self.datagen:
                (self.save_path / "measurements").mkdir()

            self.lon_logger = ScenarioLogger(
                save_path=self.save_path,
                route_index=route_index,
                logging_freq=self.config.logging_freq,
                log_only=True,
                route_only=False,  # with vehicles
                roi=self.config.logger_region_of_interest,
            )

    def toggle_recording(self, force_stop=False):
        """
        Toggle the recording of the simulation data.

        Args:
            force_stop (bool, optional): If True, stop the recording regardless of the current state.
        """
        # Toggle the recording state and determine the text
        self.recording = not self.recording

        if self.recording and not force_stop:
            self.client = CarlaDataProvider.get_client()

            # Determine the scenario name and number
            scenario_name = pathlib.Path(self.config_path).parent.stem
            scenario_number = pathlib.Path(self.config_path).stem

            # Construct the log file path
            log_path = f"{pathlib.Path(os.environ['SAVE_PATH'])}/{scenario_name}/{scenario_number}.log"

            print(f"Saving to {log_path}")
            pathlib.Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

            # Start the recorder with the specified log path
            self.client.start_recorder(log_path, True)
        else:
            # Stop the recorder
            self.client.stop_recorder()

    def _init(self, hd_map):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map (carla.Map): The map object of the CARLA world.
        """
        print("Sparse Waypoints:", len(self._global_plan))
        print("Dense Waypoints:", len(self.org_dense_route_world_coord))

        # Get the hero vehicle and the CARLA world
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2

        # Set up the route planner and extrapolation
        self._waypoint_planner = PrivilegedRoutePlanner(self.config)
        self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                           starts_with_parking_exit, self._vehicle.get_location())
        self._waypoint_planner.save()

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = LongitudinalLinearRegressionController(self.config)
        self._command_planner = RoutePlanner(self.config.route_planner_min_distance,
                                             self.config.route_planner_max_distance)
        self._command_planner.set_route(self._global_plan_world_coord)

        # Set up logging
        if self.save_path is not None:
            self.lon_logger.ego_vehicle = self._vehicle
            self.lon_logger.world = self._world

        # Preprocess traffic lights
        all_actors = self._world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = t_u.get_traffic_light_waypoints(actor, self.world_map)
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

        self.initialized = True

    def sensors(self):
        """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
        sensor_specs = [{
            "type": "sensor.opendrive_map",
            "reading_frequency": 1e-6,
            "id": "hd_map"
        }, {
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.05,
            "id": "imu"
        }, {
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed"
        }]

        return sensor_specs

    def tick_autopilot(self, input_data):
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data (dict): Input data containing sensor information.

        Returns:
            dict: A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        # Get the vehicle's speed from its velocity vector
        speed = self._vehicle.get_velocity().length()

        # Get the IMU data from the input data
        imu_data = input_data["imu"][1][-1]

        # Preprocess the compass data from the IMU
        compass = t_u.preprocess_compass(imu_data)

        # Get the vehicle's position from its location
        position = self._vehicle.get_location()
        gps = np.array([position.x, position.y, position.z])

        # Create a dictionary containing the vehicle's state
        vehicle_state = {
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        return vehicle_state

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        """
        Run a single step of the agent's control loop.

        Args:
            input_data (dict): Input data for the current step.
            timestamp (float): Timestamp of the current step.
            sensors (list, optional): List of sensor objects. Default is None.
            plant (bool, optional): Flag indicating whether to run the plant simulation or not. Default is False.

        Returns:
            If plant is False, it returns the control commands (steer, throttle, brake).
            If plant is True, it returns the driving data for the current step.
        """
        self.step += 1

        # Initialize the agent if not done yet
        if not self.initialized:
            client = CarlaDataProvider.get_client()
            world_map = client.get_world().get_map()
            self._init(world_map)

        # Get the control commands and driving data for the current step
        control, driving_data = self._get_control(input_data, plant)

        if plant:
            return driving_data
        else:
            return control

    def _get_control(self, input_data, plant):
        """
        Compute the control commands and save the driving data for the current frame.

        Args:
            input_data (dict): Input data for the current frame.
            plant (object): The plant object representing the vehicle dynamics.

        Returns:
            tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
        """
        tick_data = self.tick_autopilot(input_data)
        ego_position = tick_data["gps"]

        # Waypoint planning and route generation
        route_np, route_wp, _, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,\
                                        next_stop_sign, speed_limit = self._waypoint_planner.run_step(ego_position)

        # Extract relevant route information
        self.remaining_route = route_np[self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]
        self.remaining_route_original = self._waypoint_planner.original_route_points[
            self._waypoint_planner.route_index:][self.config.tf_first_checkpoint_distance:][::self.config.
                                                                                            points_per_meter]

        # Get the current speed and target speed
        ego_speed = tick_data["speed"]
        target_speed = speed_limit * self.config.ratio_target_speed_limit

        # Reduce target speed if there is a junction ahead
        for i in range(min(self.config.max_lookahead_to_check_for_junction, len(route_wp))):
            if route_wp[i].is_junction:
                target_speed = min(target_speed, self.config.max_speed_in_junction)
                break

        # Get the list of vehicles in the scene
        actors = self._world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))

        # Manage route obstacle scenarios and adjust target speed
        target_speed_route_obstacle, keep_driving, speed_reduced_by_obj = self._manage_route_obstacle_scenarios(
            target_speed, ego_speed, route_wp, vehicles, route_np)

        # In case the agent overtakes an obstacle, keep driving in case the opposite lane is free instead of using idm
        # and the kinematic bicycle model forecasts
        if keep_driving:
            brake, target_speed = False, target_speed_route_obstacle
        else:
            brake, target_speed, speed_reduced_by_obj = self.get_brake_and_target_speed(
                plant, route_np, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,
                next_stop_sign, vehicles, actors, target_speed, speed_reduced_by_obj)

        target_speed = min(target_speed, target_speed_route_obstacle)

        # Determine if the ego vehicle is at a junction
        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_vehicle_waypoint.is_junction

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(brake, target_speed, ego_speed)

        # Compute steering control
        steer = self._get_steer(route_np, ego_position, tick_data["compass"], ego_speed)

        # Create the control command
        control = carla.VehicleControl()
        control.steer = steer + self.config.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if control.throttle == 0 and ego_speed < self.config.minimum_speed_to_prevent_rolling_back:
            control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = CarlaDataProvider.get_velocity(self._vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.config.max_blocked_ticks:
            control.throttle = 1
            control.brake = 0

        # Save control commands and target speed
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        # Update speed histogram if enabled
        if self.make_histogram:
            self.speed_histogram.append((self.target_speed * 3.6) if not brake else 0.0)

        # Get the target and next target points from the command planner
        command_route = self._command_planner.run_step(ego_position)
        if len(command_route) > 2:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[2]
        elif len(command_route) > 1:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[1]
        else:
            target_point, far_command = command_route[0]
            next_target_point, next_far_command = command_route[0]

        # Update command history and save driving datas
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
            self.next_commands.append(next_far_command.value)

        driving_data = self.save(target_point, next_target_point, steer, throttle, brake, control_brake, target_speed,
                                 speed_limit, tick_data, speed_reduced_by_obj)

        return control, driving_data

    def _manage_route_obstacle_scenarios(self, target_speed, ego_speed, route_waypoints, list_vehicles, route_points):
        """
        This method handles various obstacle and scenario situations that may arise during navigation.
        It adjusts the target speed, modifies the route, and determines if the ego vehicle should keep driving or wait.
        The method supports different scenario types such as InvadingTurn, Accident, ConstructionObstacle, 
        ParkedObstacle, AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, VehicleOpensDoorTwoWays, 
        HazardAtSideLaneTwoWays, HazardAtSideLane, and YieldToEmergencyVehicle.

        Args:
            target_speed (float): The current target speed of the ego vehicle.
            ego_speed (float): The current speed of the ego vehicle.
            route_waypoints (list): A list of waypoints representing the current route.
            list_vehicles (list): A list of all vehicles in the simulation.
            route_points (numpy.ndarray): A numpy array containing the current route points.

        Returns:
            tuple: A tuple containing the updated target speed, a boolean indicating whether to keep driving,
                and a list containing information about a potential decreased target speed due to an object.
        """

        def compute_min_time_for_distance(distance, target_speed, ego_speed):
            """
            Computes the minimum time the ego vehicle needs to travel a given distance.

            Args:
                distance (float): The distance to be traveled.
                target_speed (float): The target speed of the ego vehicle.
                ego_speed (float): The current speed of the ego vehicle.

            Returns:
                float: The minimum time needed to travel the given distance.
            """
            min_time_needed = 0.
            remaining_distance = distance
            current_speed = ego_speed

            # Iterate over time steps until the distance is covered
            while True:
                # Takes less than a tick to cover remaining_distance with current_speed
                if remaining_distance - current_speed * self.config.fps_inv < 0:
                    break

                remaining_distance -= current_speed * self.config.fps_inv
                min_time_needed += self.config.fps_inv

                # Values from kinematic bicycle model
                normalized_speed = current_speed / 120.
                speed_change_params = self.config.compute_min_time_to_cover_distance_params
                speed_change = np.clip(
                    speed_change_params[0] + normalized_speed * speed_change_params[1] +
                    speed_change_params[2] * normalized_speed**2 + speed_change_params[3] * normalized_speed**3, 0.,
                    np.inf)
                current_speed = np.clip(120 * (normalized_speed + speed_change), 0, target_speed)

            # Add remaining time at the current speed
            min_time_needed += remaining_distance / current_speed

            return min_time_needed

        def get_previous_road_lane_ids(starting_waypoint):
            """
            Retrieves the previous road and lane IDs for a given starting waypoint.

            Args:
                starting_waypoint (carla.Waypoint): The starting waypoint.

            Returns:
                list: A list of tuples containing road IDs and lane IDs.
            """
            current_waypoint = starting_waypoint
            previous_lane_ids = [(current_waypoint.road_id, current_waypoint.lane_id)]

            # Traverse backwards up to 100 waypoints to find previous lane IDs
            for _ in range(self.config.previous_road_lane_retrieve_distance):
                previous_waypoints = current_waypoint.previous(1)

                # Check if the road ends and no previous route waypoints exist
                if len(previous_waypoints) == 0:
                    break
                current_waypoint = previous_waypoints[0]

                if (current_waypoint.road_id, current_waypoint.lane_id) not in previous_lane_ids:
                    previous_lane_ids.append((current_waypoint.road_id, current_waypoint.lane_id))

            return previous_lane_ids

        def is_overtaking_path_clear(from_index,
                                     to_index,
                                     list_vehicles,
                                     ego_location,
                                     target_speed,
                                     ego_speed,
                                     previous_lane_ids,
                                     min_speed=50. / 3.6):
            """
            Checks if the path between two route indices is clear for the ego vehicle to overtake.

            Args:
                from_index (int): The starting route index.
                to_index (int): The ending route index.
                list_vehicles (list): A list of all vehicles in the simulation.
                ego_location (carla.Location): The location of the ego vehicle.
                target_speed (float): The target speed of the ego vehicle.
                ego_speed (float): The current speed of the ego vehicle.
                previous_lane_ids (list): A list of tuples containing previous road IDs and lane IDs.
                min_speed (float, optional): The minimum speed to consider for overtaking. Defaults to 50/3.6 km/h.

            Returns:
                bool: True if the path is clear for overtaking, False otherwise.
            """
            # 10 m safety distance, overtake with max. 50 km/h
            to_location = self._waypoint_planner.route_points[to_index]
            to_location = carla.Location(to_location[0], to_location[1], to_location[2])

            from_location = self._waypoint_planner.route_points[from_index]
            from_location = carla.Location(from_location[0], from_location[1], from_location[2])

            # Compute the distance and time needed for the ego vehicle to overtake
            ego_distance = to_location.distance(
                ego_location) + self._vehicle.bounding_box.extent.x * 2 + self.config.check_path_free_safety_distance
            ego_time = compute_min_time_for_distance(ego_distance, min(min_speed, target_speed), ego_speed)

            path_clear = True
            for vehicle in list_vehicles:
                # Sort out ego vehicle
                if vehicle.id == self._vehicle.id:
                    continue

                vehicle_location = vehicle.get_location()
                vehicle_waypoint = self.world_map.get_waypoint(vehicle_location)

                # Check if the vehicle is on the previous lane IDs
                if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) in previous_lane_ids:
                    diff_vector = vehicle_location - ego_location
                    dot_product = self._vehicle.get_transform().get_forward_vector().dot(diff_vector)
                    # Skip if the vehicle is not relevant, because its not on the overtaking path and behind
                    # the ego vehicle
                    if dot_product < 0:
                        continue

                    diff_vector_2 = to_location - vehicle_location
                    dot_product_2 = vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                    # The overtaking path is blocked by vehicle
                    if dot_product_2 < 0:
                        path_clear = False
                        break

                    other_vehicle_distance = to_location.distance(vehicle_location) - vehicle.bounding_box.extent.x
                    other_vehicle_time = other_vehicle_distance / max(1., vehicle.get_velocity().length())

                    # Add 200 ms safety margin
                    # Vehicle needs less time to arrive at to_location than the ego vehicle
                    if other_vehicle_time < ego_time + self.config.check_path_free_safety_time:
                        path_clear = False
                        break

            return path_clear

        def get_horizontal_distance(actor1, actor2):
            """
            Calculates the horizontal distance between two actors (ignoring the z-coordinate).

            Args:
                actor1 (carla.Actor): The first actor.
                actor2 (carla.Actor): The second actor.

            Returns:
                float: The horizontal distance between the two actors.
            """
            location1, location2 = actor1.get_location(), actor2.get_location()

            # Compute the distance vector (ignoring the z-coordinate)
            diff_vector = carla.Vector3D(location1.x - location2.x, location1.y - location2.y, 0)

            return diff_vector.length()

        def sort_scenarios_by_distance(ego_location):
            """
            Sorts the active scenarios based on the distance from the ego vehicle.

            Args:
                ego_location (carla.Location): The location of the ego vehicle.
            """
            distances = []

            # Calculate the distance of each scenario's first actor from the ego vehicle
            for (_, scenario_data) in CarlaDataProvider.active_scenarios:
                first_actor = scenario_data[0]
                distances.append(ego_location.distance(first_actor.get_location()))

            # Sort the scenarios based on the calculated distances
            indices = np.argsort(distances)
            CarlaDataProvider.active_scenarios = [CarlaDataProvider.active_scenarios[i] for i in indices]

        keep_driving = False
        speed_reduced_by_obj = [target_speed, None, None, None]  # [target_speed, type, id, distance]

        # Remove scenarios that ended with a scenario timeout
        active_scenarios = CarlaDataProvider.active_scenarios.copy()
        for i, (scenario_type, scenario_data) in enumerate(active_scenarios):
            first_actor, last_actor = scenario_data[:2]
            if not first_actor.is_alive or (last_actor is not None and not last_actor.is_alive):
                CarlaDataProvider.active_scenarios.remove(active_scenarios[i])

        # Only continue if there are some active scenarios available
        if len(CarlaDataProvider.active_scenarios) != 0:
            ego_location = self._vehicle.get_location()

            # Sort the scenarios by distance if there is more than one active scenario
            if len(CarlaDataProvider.active_scenarios) != 1:
                sort_scenarios_by_distance(ego_location)

            scenario_type, scenario_data = CarlaDataProvider.active_scenarios[0]

            if scenario_type == "InvadingTurn":
                first_cone, last_cone, offset = scenario_data

                closest_distance = first_cone.get_location().distance(ego_location)

                if closest_distance < self.config.default_max_distance_to_process_scenario:
                    self._waypoint_planner.shift_route_for_invading_turn(first_cone, last_cone, offset)
                    CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

            elif scenario_type in ["Accident", "ConstructionObstacle", "ParkedObstacle"]:
                first_actor, last_actor, direction = scenario_data[:3]

                horizontal_distance = get_horizontal_distance(self._vehicle, first_actor)

                # Shift the route around the obstacles
                if horizontal_distance < self.config.default_max_distance_to_process_scenario:
                    transition_length = {
                        "Accident": self.config.transition_smoothness_distance,
                        "ConstructionObstacle": self.config.transition_smoothness_factor_construction_obstacle,
                        "ParkedObstacle": self.config.transition_smoothness_distance
                    }[scenario_type]
                    _, _ = self._waypoint_planner.shift_route_around_actors(first_actor, last_actor, direction,
                                                                            transition_length)
                    CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

            elif scenario_type in [
                    "AccidentTwoWays", "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "VehicleOpensDoorTwoWays"
            ]:
                first_actor, last_actor, direction, changed_route, from_index, to_index, path_clear = scenario_data

                # change the route if the ego is close enough to the obstacle
                horizontal_distance = get_horizontal_distance(self._vehicle, first_actor)

                # Shift the route around the obstacles
                if horizontal_distance < self.config.default_max_distance_to_process_scenario and not changed_route:
                    transition_length = {
                        "AccidentTwoWays": self.config.transition_length_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.transition_length_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.transition_length_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.transition_length_vehicle_opens_door_two_ways
                    }[scenario_type]
                    add_before_length = {
                        "AccidentTwoWays": self.config.add_before_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.add_before_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.add_before_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.add_before_vehicle_opens_door_two_ways
                    }[scenario_type]
                    add_after_length = {
                        "AccidentTwoWays": self.config.add_after_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.add_after_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.add_after_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.add_after_vehicle_opens_door_two_ways
                    }[scenario_type]
                    factor = {
                        "AccidentTwoWays": self.config.factor_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.factor_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.factor_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.factor_vehicle_opens_door_two_ways
                    }[scenario_type]

                    from_index, to_index = self._waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, direction, transition_length, factor, add_before_length,
                        add_after_length)

                    changed_route = True
                    scenario_data[3] = changed_route
                    scenario_data[4] = from_index
                    scenario_data[5] = to_index

                # Check if the ego can overtake the obstacle
                if changed_route and from_index - self._waypoint_planner.route_index < \
                                self.config.max_distance_to_overtake_two_way_scnearios and not path_clear:
                    # Get previous roads and lanes of the target lane
                    target_lane = route_waypoints[0].get_left_lane(
                    ) if direction == "right" else route_waypoints[0].get_right_lane()
                    if target_lane is None:
                        return target_speed, keep_driving, speed_reduced_by_obj
                    prev_road_lane_ids = get_previous_road_lane_ids(target_lane)

                    overtake_speed = self.config.overtake_speed_vehicle_opens_door_two_ways \
                                if scenario_type == "VehicleOpensDoorTwoWays" else self.config.default_overtake_speed
                    path_clear = is_overtaking_path_clear(from_index,
                                                          to_index,
                                                          list_vehicles,
                                                          ego_location,
                                                          target_speed,
                                                          ego_speed,
                                                          prev_road_lane_ids,
                                                          min_speed=overtake_speed)

                    scenario_data[6] = path_clear

                # If the overtaking path is clear, keep driving; otherwise, wait behind the obstacle
                if path_clear:
                    if self._waypoint_planner.route_index >= to_index - \
                                                            self.config.distance_to_delete_scenario_in_two_ways:
                        CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]
                    target_speed = {
                        "AccidentTwoWays": self.config.default_overtake_speed,
                        "ConstructionObstacleTwoWays": self.config.default_overtake_speed,
                        "ParkedObstacleTwoWays": self.config.default_overtake_speed,
                        "VehicleOpensDoorTwoWays": self.config.overtake_speed_vehicle_opens_door_two_ways
                    }[scenario_type]
                    keep_driving = True
                else:
                    distance_to_leading_actor = float(from_index + 15 -
                                                      self._waypoint_planner.route_index) / self.config.points_per_meter
                    target_speed = self._compute_target_speed_idm(
                        desired_speed=target_speed,
                        leading_actor_length=self._vehicle.bounding_box.extent.x,
                        ego_speed=ego_speed,
                        leading_actor_speed=0,
                        distance_to_leading_actor=distance_to_leading_actor,
                        s0=self.config.idm_two_way_scenarios_minimum_distance,
                        T=self.config.idm_two_way_scenarios_time_headway
                    )

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed:
                        speed_reduced_by_obj = [
                            target_speed, first_actor.type_id, first_actor.id, distance_to_leading_actor
                        ]

            elif scenario_type == "HazardAtSideLaneTwoWays":
                first_actor, last_actor, changed_route, from_index, to_index, path_clear = scenario_data

                horizontal_distance = get_horizontal_distance(self._vehicle, first_actor)

                if horizontal_distance < self.config.max_distance_to_process_hazard_at_side_lane_two_ways \
                                                                                        and not changed_route:
                    to_index = self._waypoint_planner.get_closest_route_index(self._waypoint_planner.route_index,
                                                                              last_actor.get_location())

                    # Assume the bicycles don't drive more than 7.5 m during the overtaking process
                    to_index += 135
                    from_index = self._waypoint_planner.route_index

                    starting_wp = route_waypoints[0].get_left_lane()
                    prev_road_lane_ids = get_previous_road_lane_ids(starting_wp)
                    path_clear = is_overtaking_path_clear(from_index,
                                                          to_index,
                                                          list_vehicles,
                                                          ego_location,
                                                          target_speed,
                                                          ego_speed,
                                                          prev_road_lane_ids,
                                                          min_speed=self.config.default_overtake_speed)

                    if path_clear:
                        transition_length = self.config.transition_smoothness_distance
                        self._waypoint_planner.shift_route_smoothly(from_index, to_index, True, transition_length)
                        changed_route = True
                        scenario_data[2] = changed_route
                        scenario_data[3] = from_index
                        scenario_data[4] = to_index
                        scenario_data[5] = path_clear

                # the overtaking path is clear
                if path_clear:
                    # Check if the overtaking is done
                    if self._waypoint_planner.route_index >= to_index:
                        CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]
                    # Overtake with max. 50 km/h
                    target_speed, keep_driving = self.config.default_overtake_speed, True

            elif scenario_type == "HazardAtSideLane":
                first_actor, last_actor, changed_first_part_of_route, from_index, to_index, path_clear = scenario_data

                horizontal_distance = get_horizontal_distance(self._vehicle, last_actor)

                if horizontal_distance < self.config.max_distance_to_process_hazard_at_side_lane \
                                                                and not changed_first_part_of_route:
                    transition_length = self.config.transition_smoothness_distance
                    from_index, to_index = self._waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, "right", transition_length)

                    to_index -= transition_length
                    changed_first_part_of_route = True
                    scenario_data[2] = changed_first_part_of_route
                    scenario_data[3] = from_index
                    scenario_data[4] = to_index

                if changed_first_part_of_route:
                    to_idx_ = self._waypoint_planner.extend_lane_shift_transition_for_hazard_at_side_lane(
                        last_actor, to_index)
                    to_index = to_idx_
                    scenario_data[4] = to_index

                if self._waypoint_planner.route_index > to_index:
                    CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

            elif scenario_type == "YieldToEmergencyVehicle":
                emergency_veh, _, changed_route, from_index, to_index, to_left = scenario_data

                horizontal_distance = get_horizontal_distance(self._vehicle, emergency_veh)

                if horizontal_distance < self.config.default_max_distance_to_process_scenario and not changed_route:
                    # Assume the emergency vehicle doesn't drive more than 20 m during the overtaking process
                    from_index = self._waypoint_planner.route_index + 30 * self.config.points_per_meter
                    to_index = from_index + int(2 * self.config.points_per_meter) * self.config.points_per_meter

                    transition_length = self.config.transition_smoothness_distance
                    to_left = self._waypoint_planner.route_waypoints[from_index].lane_change != carla.LaneChange.Right
                    self._waypoint_planner.shift_route_smoothly(from_index, to_index, to_left, transition_length)

                    changed_route = True
                    to_index -= transition_length
                    scenario_data[2] = changed_route
                    scenario_data[3] = from_index
                    scenario_data[4] = to_index
                    scenario_data[5] = to_left

                if changed_route:
                    to_idx_ = self._waypoint_planner.extend_lane_shift_transition_for_yield_to_emergency_vehicle(
                        to_left, to_index)
                    to_index = to_idx_
                    scenario_data[4] = to_index

                    # Check if the emergency vehicle is in front of the ego vehicle
                    diff = emergency_veh.get_location() - ego_location
                    dot_res = self._vehicle.get_transform().get_forward_vector().dot(diff)
                    if dot_res > 0:
                        CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

        # Visualization for debugging
        if self.visualize == 1:
            for i in range(min(route_points.shape[0] - 1, self.config.draw_future_route_till_distance)):
                loc = route_points[i]
                loc = carla.Location(loc[0], loc[1], loc[2] + 0.1)
                self._world.debug.draw_point(location=loc,
                                             size=0.05,
                                             color=self.config.future_route_color,
                                             life_time=self.config.draw_life_time)

        return target_speed, keep_driving, speed_reduced_by_obj

    def save(self, target_point, next_target_point, steering, throttle, brake, control_brake, target_speed, speed_limit,
             tick_data, speed_reduced_by_obj):
        """
        Save the driving data for the current frame.

        Args:
            target_point (numpy.ndarray): Coordinates of the target point.
            next_target_point (numpy.ndarray): Coordinates of the next target point.
            steering (float): The steering angle for the current frame.
            throttle (float): The throttle value for the current frame.
            brake (float): The brake value for the current frame.
            control_brake (bool): Whether the brake is controlled by the agent or not.
            target_speed (float): The target speed for the current frame.
            speed_limit (float): The speed limit for the current frame.
            tick_data (dict): Dictionary containing the current state of the vehicle.
            speed_reduced_by_obj (tuple): Tuple containing information about the object that caused speed reduction.

        Returns:
            dict: A dictionary containing the driving data for the current frame.
        """
        frame = self.step // self.config.data_save_freq

        # Extract relevant data from inputs
        target_point_2d = target_point[:2]
        next_target_point_2d = next_target_point[:2]
        ego_position = tick_data["gps"][:2]
        ego_orientation = tick_data["compass"]
        ego_speed = tick_data["speed"]

        # Convert target points to ego vehicle's local coordinate frame
        ego_target_point = t_u.inverse_conversion_2d(target_point_2d, ego_position, ego_orientation).tolist()
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point_2d, ego_position, ego_orientation).tolist()
        ego_aim_point = t_u.inverse_conversion_2d(self.aim_wp[:2], ego_position, ego_orientation).tolist()

        # Get the remaining route points in the local coordinate frame
        dense_route = []
        dense_route_original = []
        remaining_route = self.remaining_route[:self.config.num_route_points_saved]
        remaining_route_original = self.remaining_route_original[:self.config.num_route_points_saved]

        changed_route = bool(
            (self._waypoint_planner.route_points[self._waypoint_planner.route_index]
             != self._waypoint_planner.original_route_points[self._waypoint_planner.route_index]).any())
        for (checkpoint, checkpoint_original) in zip(remaining_route, remaining_route_original):
            dense_route.append(t_u.inverse_conversion_2d(checkpoint[:2], ego_position[:2], ego_orientation).tolist())
            dense_route_original.append(
                t_u.inverse_conversion_2d(checkpoint_original[:2], ego_position[:2], ego_orientation).tolist())

        # Extract speed reduction object information
        speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = None, None, None
        if speed_reduced_by_obj is not None:
            speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = speed_reduced_by_obj[1:]

        data = {
            "pos_global": ego_position.tolist(),
            "theta": ego_orientation,
            "speed": ego_speed,
            "target_speed": target_speed,
            "speed_limit": speed_limit,
            "target_point": ego_target_point,
            "target_point_next": ego_next_target_point,
            "command": self.commands[-2],
            "next_command": self.next_commands[-2],
            "aim_wp": ego_aim_point,
            "route": dense_route,
            "route_original": dense_route_original,
            "changed_route": changed_route,
            "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
            "speed_reduced_by_obj_id": speed_reduced_by_obj_id,
            "speed_reduced_by_obj_distance": speed_reduced_by_obj_distance,
            "steer": steering,
            "throttle": throttle,
            "brake": bool(brake),
            "control_brake": bool(control_brake),
            "junction": bool(self.junction),
            "vehicle_hazard": bool(self.vehicle_hazard),
            "vehicle_affecting_id": self.vehicle_affecting_id,
            "light_hazard": bool(self.traffic_light_hazard),
            "walker_hazard": bool(self.walker_hazard),
            "walker_affecting_id": self.walker_affecting_id,
            "stop_sign_hazard": bool(self.stop_sign_hazard),
            "stop_sign_close": bool(self.stop_sign_close),
            "walker_close": bool(self.walker_close),
            "walker_close_id": self.walker_close_id,
            "angle": self.angle,
            "augmentation_translation": self.augmentation_translation,
            "augmentation_rotation": self.augmentation_rotation,
            "ego_matrix": self._vehicle.get_transform().get_matrix()
        }

        if self.tp_stats:
            deg_pred_angle = -math.degrees(math.atan2(-ego_aim_point[1], ego_aim_point[0]))

            tp_angle = -math.degrees(math.atan2(-ego_target_point[1], ego_target_point[0]))
            if abs(tp_angle) > 1.0 and abs(deg_pred_angle) > 1.0:
                same_direction = float(tp_angle * deg_pred_angle >= 0.0)
                self.tp_sign_agrees_with_angle.append(same_direction)

        if ((self.step % self.config.data_save_freq == 0) and (self.save_path is not None) and self.datagen):
            measurements_file = self.save_path / "measurements" / f"{frame:04}.json.gz"
            with gzip.open(measurements_file, "wt", encoding="utf-8") as f:
                ujson.dump(data, f, indent=4)

        return data

    def destroy(self, results=None):
        """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results (optional): Any additional results to be processed or saved.
        """
        if self.save_path is not None:
            self.lon_logger.dump_to_json()

            # Save the target speed histogram to a compressed JSON file
            if len(self.speed_histogram) > 0:
                with gzip.open(self.save_path / "target_speeds.json.gz", "wt", encoding="utf-8") as f:
                    ujson.dump(self.speed_histogram, f, indent=4)

            del self.speed_histogram

            if self.tp_stats:
                if len(self.tp_sign_agrees_with_angle) > 0:
                    print("Agreement between TP and steering: ",
                          sum(self.tp_sign_agrees_with_angle) / len(self.tp_sign_agrees_with_angle))
                    with gzip.open(self.save_path / "tp_agreements.json.gz", "wt", encoding="utf-8") as f:
                        ujson.dump(self.tp_sign_agrees_with_angle, f, indent=4)

        del self.tp_sign_agrees_with_angle
        del self.visible_walker_ids
        del self.walker_past_pos

    def _get_steer(self, route_points, current_position, current_heading, current_speed):
        """
        Calculate the steering angle based on the current position, heading, speed, and the route points.

        Args:
            route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
            current_position (tuple): The current position (x, y) of the vehicle.
            current_heading (float): The current heading angle (in radians) of the vehicle.
            current_speed (float): The current speed of the vehicle (in m/s).

        Returns:
            float: The calculated steering angle.
        """
        speed_scale = self.config.lateral_pid_speed_scale
        speed_offset = self.config.lateral_pid_speed_offset

        # Calculate the lookahead index based on the current speed
        speed_in_kmph = current_speed * 3.6
        lookahead_distance = speed_scale * speed_in_kmph + speed_offset
        lookahead_distance = np.clip(lookahead_distance, self.config.lateral_pid_default_lookahead,
                                     self.config.lateral_pid_maximum_lookahead_distance)
        lookahead_index = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Get the target point from the route points
        target_point = route_points[lookahead_index]

        # Calculate the angle between the current heading and the target point
        angle_unnorm = self._get_angle_to(current_position, current_heading, target_point)
        normalized_angle = angle_unnorm / 90

        self.aim_wp = target_point
        self.angle = normalized_angle

        # Calculate the steering angle using the turn controller
        steering_angle = self._turn_controller.step(route_points, current_speed, current_position, current_heading)
        steering_angle = round(steering_angle, 3)

        return steering_angle

    def _compute_target_speed_idm(self,
                                  desired_speed,
                                  leading_actor_length,
                                  ego_speed,
                                  leading_actor_speed,
                                  distance_to_leading_actor,
                                  s0=4.,
                                  T=0.5):
        """
        Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

        Args:
            desired_speed (float): The desired speed of the ego vehicle.
            leading_actor_length (float): The length of the leading actor (vehicle or obstacle).
            ego_speed (float): The current speed of the ego vehicle.
            leading_actor_speed (float): The speed of the leading actor.
            distance_to_leading_actor (float): The distance to the leading actor.
            s0 (float, optional): The minimum desired net distance.
            T (float, optional): The desired time headway.

        Returns:
            float: The computed target speed for the ego vehicle.
        """
        a = self.config.idm_maximum_acceleration # Maximum acceleration [m/s²]
        b = self.config.idm_comfortable_braking_deceleration_high_speed if ego_speed > \
                        self.config.idm_comfortable_braking_deceleration_threshold else \
                        self.config.idm_comfortable_braking_deceleration_low_speed # Comfortable deceleration [m/s²]
        delta = self.config.idm_acceleration_exponent # Acceleration exponent
        
        t_bound = self.config.idm_t_bound

        def idm_equations(t, x):
            """
            Differential equations for the Intelligent Driver Model.

            Args:
                t (float): Time.
                x (list): State variables [position, speed].

            Returns:
                list: Derivatives of the state variables.
            """
            ego_position, ego_speed = x

            speed_diff = ego_speed - leading_actor_speed
            s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2. / np.sqrt(a * b)
            # The maximum is needed to avoid numerical unstabilities
            s = max(0.1, distance_to_leading_actor + t * leading_actor_speed - ego_position - leading_actor_length)
            dvdt = a * (1. - (ego_speed / desired_speed)**delta - (s_star / s)**2)

            return [ego_speed, dvdt]

        # Set the initial conditions
        y0 = [0., ego_speed]

        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equations, t0=0., y0=y0, t_bound=t_bound)
        while rk45.status == "running":
            rk45.step()

        # The target speed is the final speed obtained from the integration
        target_speed = rk45.y[1]

        # Clip the target speed to non-negative values
        return np.clip(target_speed, 0, np.inf)

    def is_near_lane_change(self, ego_velocity, route_points):
        """
        Computes if the ego agent is/was close to a lane change maneuver.

        Args:
            ego_velocity (float): The current velocity of the ego agent in m/s.
            route_points (numpy.ndarray): An array of locations representing the planned route.

        Returns:
            bool: True if the ego agent is close to a lane change, False otherwise.
        """
        # Calculate the braking distance based on the ego velocity
        braking_distance = ((
            (ego_velocity * 3.6) / 10.0)**2 / 2.0) + self.config.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(self.config.minimum_lookahead_distance_to_compute_near_lane_change,
                                min(route_points.shape[0], self.config.points_per_meter * int(braking_distance)))
        current_route_index = self._waypoint_planner.route_index
        max_route_length = len(self._waypoint_planner.commands)

        from_index = max(0, current_route_index - self.config.check_previous_distance_for_lane_change)
        to_index = min(max_route_length - 1, current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self._waypoint_planner.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return True

        return False

    def predict_other_actors_bounding_boxes(self, plant, actor_list, ego_vehicle_location, num_future_frames,
                                            near_lane_change):
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            plant (bool): Whether to use PlanT.
            actor_list (list): A list of actors (e.g., vehicles) in the simulation.
            ego_vehicle_location (carla.Location): The current location of the ego vehicle.
            num_future_frames (int): The number of future frames to predict.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.

        Returns:
            dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """
        predicted_bounding_boxes = {}

        if not plant:
            # Filter out nearby actors within the detection radius, excluding the ego vehicle
            nearby_actors = [
                actor for actor in actor_list if actor.id != self._vehicle.id and
                actor.get_location().distance(ego_vehicle_location) < self.config.detection_radius
            ]

            # If there are nearby actors, calculate their future bounding boxes
            if nearby_actors:
                # Get the previous control inputs (steering, throttle, brake) for the nearby actors
                previous_controls = [actor.get_control() for actor in nearby_actors]
                previous_actions = np.array(
                    [[control.steer, control.throttle, control.brake] for control in previous_controls])

                # Get the current velocities, locations, and headings of the nearby actors
                velocities = np.array([actor.get_velocity().length() for actor in nearby_actors])
                locations = np.array([[actor.get_location().x,
                                       actor.get_location().y,
                                       actor.get_location().z] for actor in nearby_actors])
                headings = np.deg2rad(np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

                # Initialize arrays to store future locations, headings, and velocities
                future_locations = np.empty((num_future_frames, len(nearby_actors), 3), dtype="float")
                future_headings = np.empty((num_future_frames, len(nearby_actors)), dtype="float")
                future_velocities = np.empty((num_future_frames, len(nearby_actors)), dtype="float")

                # Forecast the future locations, headings, and velocities for the nearby actors
                for i in range(num_future_frames):
                    locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                        locations, headings, velocities, previous_actions)
                    future_locations[i] = locations.copy()
                    future_velocities[i] = velocities.copy()
                    future_headings[i] = headings.copy()

                # Convert future headings to degrees
                future_headings = np.rad2deg(future_headings)

                # Calculate the predicted bounding boxes for each nearby actor and future frame
                for actor_idx, actor in enumerate(nearby_actors):
                    predicted_actor_boxes = []

                    for i in range(num_future_frames):
                        # Calculate the future location of the actor
                        location = carla.Location(x=future_locations[i, actor_idx, 0].item(),
                                                  y=future_locations[i, actor_idx, 1].item(),
                                                  z=future_locations[i, actor_idx, 2].item())

                        # Calculate the future rotation of the actor
                        rotation = carla.Rotation(pitch=0, yaw=future_headings[i, actor_idx], roll=0)

                        # Get the extent (dimensions) of the actor's bounding box
                        extent = actor.bounding_box.extent
                        # Otherwise we would increase the extent of the bounding box of the vehicle
                        extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                        # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                        # uncertainty during forecasting
                        s = self.config.high_speed_min_extent_x_other_vehicle_lane_change if near_lane_change \
                            else self.config.high_speed_min_extent_x_other_vehicle
                        extent.x *= self.config.slow_speed_extent_factor_ego if future_velocities[
                            i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                                s,
                                self.config.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames))
                        extent.y *= self.config.slow_speed_extent_factor_ego if future_velocities[
                            i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                                self.config.high_speed_min_extent_y_other_vehicle,
                                self.config.high_speed_extent_y_factor_other_vehicle * float(i) /
                                float(num_future_frames))

                        # Create the bounding box for the future frame
                        bounding_box = carla.BoundingBox(location, extent)
                        bounding_box.rotation = rotation

                        # Append the bounding box to the list of predicted bounding boxes for this actor
                        predicted_actor_boxes.append(bounding_box)

                    # Store the predicted bounding boxes for this actor in the dictionary
                    predicted_bounding_boxes[actor.id] = predicted_actor_boxes

                if self.visualize == 1:
                    for actor_idx, actors_forecasted_bounding_boxes in predicted_bounding_boxes.items():
                        for bb in actors_forecasted_bounding_boxes:
                            self._world.debug.draw_box(box=bb,
                                                       rotation=bb.rotation,
                                                       thickness=0.1,
                                                       color=self.config.other_vehicles_forecasted_bbs_color,
                                                       life_time=self.config.draw_life_time)

        return predicted_bounding_boxes

    def compute_target_speed_wrt_leading_vehicle(self, initial_target_speed, predicted_bounding_boxes, near_lane_change,
                                                 ego_location, rear_vehicle_ids, leading_vehicle_ids,
                                                 speed_reduced_by_obj, plant):
        """
        Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            ego_location (carla.Location): The current location of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] 
                for the object that caused the most speed reduction, or None if no speed reduction.
            plant (bool): Whether to use plant.

        Returns:
            float: The target speed considering the leading vehicle.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed

        if not plant:
            for vehicle_id, _ in predicted_bounding_boxes.items():
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    # Vehicle is in front of the ego vehicle
                    ego_speed = self._vehicle.get_velocity().length()
                    vehicle = self._world.get_actor(vehicle_id)
                    other_speed = vehicle.get_velocity().length()
                    distance_to_vehicle = ego_location.distance(vehicle.get_location())

                    # Compute the target speed using the IDM
                    target_speed_wrt_leading_vehicle = min(
                        target_speed_wrt_leading_vehicle,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=vehicle.bounding_box.extent.x * 2,
                            ego_speed=ego_speed,
                            leading_actor_speed=other_speed,
                            distance_to_leading_actor=distance_to_vehicle,
                            s0=self.config.idm_leading_vehicle_minimum_distance,
                            T=self.config.idm_leading_vehicle_time_headway
                        ))

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle:
                        speed_reduced_by_obj = [
                            target_speed_wrt_leading_vehicle, vehicle.type_id, vehicle.id, distance_to_vehicle
                        ]

            if self.visualize == 1:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if vehicle_id in leading_vehicle_ids and not near_lane_change:
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self._world.debug.draw_box(box=bb,
                                                   rotation=bb.rotation,
                                                   thickness=0.5,
                                                   color=self.config.leading_vehicle_color,
                                                   life_time=self.config.draw_life_time)
                    elif vehicle_id in rear_vehicle_ids:
                        vehicle = self._world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self._world.debug.draw_box(box=bb,
                                                   rotation=bb.rotation,
                                                   thickness=0.5,
                                                   color=self.config.trailing_vehicle_color,
                                                   life_time=self.config.draw_life_time)

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    def compute_target_speeds_wrt_all_actors(self, initial_target_speed, ego_bounding_boxes, predicted_bounding_boxes,
                                             near_lane_change, leading_vehicle_ids, rear_vehicle_ids,
                                             speed_reduced_by_obj, nearby_walkers, nearby_walkers_ids):
        """
        Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles, 
        and pedestrians) by checking for intersecting bounding boxes.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            ego_bounding_boxes (list): A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, 
                object ID, distance] for the object that caused the most speed reduction, or None if 
                no speed reduction.
            nearby_walkers (dict): A list of predicted bounding boxes of nearby pedestrians.
            nearby_walkers_ids (list): A list of IDs for nearby pedestrians.

        Returns:
            tuple: A tuple containing the target speeds for bicycles, pedestrians, vehicles, and the updated 
                speed_reduced_by_obj list.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = self._vehicle.get_location()
        hazard_color = self.config.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.config.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = self.check_obb_intersection(ego_bounding_box, bounding_boxes[i])
                    ego_speed = self._vehicle.get_velocity().length()

                    if intersects_with_ego:
                        blocking_actor = self._world.get_actor(vehicle_id)

                        # Handle the case when the blocking actor is a bicycle
                        if "base_type" in blocking_actor.attributes and blocking_actor.attributes[
                                "base_type"] == "bicycle":
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(blocking_actor.get_location())

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                self._compute_target_speed_idm(
                                    desired_speed=initial_target_speed,
                                    leading_actor_length=blocking_actor.bounding_box.extent.x * 2,
                                    ego_speed=ego_speed,
                                    leading_actor_speed=other_speed,
                                    distance_to_leading_actor=distance_to_actor,
                                    s0=self.config.idm_bicycle_minimum_distance,
                                    T=self.config.idm_bicycle_desired_time_headway
                                ))

                            # Update the object causing the most speed reduction
                            if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_bicycle:
                                speed_reduced_by_obj = [
                                    target_speed_bicycle, blocking_actor.type_id, blocking_actor.id, distance_to_actor
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = 0  # Set the target speed for vehicles to zero
                            distance_to_actor = blocking_actor.get_location().distance(ego_vehicle_location)

                            # Update the object causing the most speed reduction
                            if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_vehicle:
                                speed_reduced_by_obj = [
                                    target_speed_vehicle, blocking_actor.type_id, blocking_actor.id, distance_to_actor
                                ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(nearby_walkers, nearby_walkers_ids):
                if self.check_obb_intersection(ego_bounding_box, pedestrian_bb[i]):
                    color = hazard_color
                    ego_speed = self._vehicle.get_velocity().length()
                    blocking_actor = self._world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(blocking_actor.get_location())

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=0.5 + self._vehicle.bounding_box.extent.x,
                            ego_speed=ego_speed,
                            leading_actor_speed=0.,
                            distance_to_leading_actor=distance_to_actor,
                            s0=self.config.idm_pedestrian_minimum_distance,
                            T=self.config.idm_pedestrian_desired_time_headway
                        ))

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_pedestrian:
                        speed_reduced_by_obj = [
                            target_speed_pedestrian, blocking_actor.type_id, blocking_actor.id, distance_to_actor
                        ]

            if self.visualize == 1:
                self._world.debug.draw_box(box=ego_bounding_box,
                                           rotation=ego_bounding_box.rotation,
                                           thickness=0.1,
                                           color=color,
                                           life_time=self.config.draw_life_time)

        return target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, speed_reduced_by_obj

    def get_brake_and_target_speed(self, plant, route_points, distance_to_next_traffic_light, next_traffic_light,
                                   distance_to_next_stop_sign, next_stop_sign, vehicle_list, actor_list,
                                   initial_target_speed, speed_reduced_by_obj):
        """
        Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            plant (bool): Whether to use PlanT.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.
            distance_to_next_traffic_light (float): The distance to the next traffic light.
            next_traffic_light (carla.TrafficLight): The next traffic light actor.
            distance_to_next_stop_sign (float): The distance to the next stop sign.
            next_stop_sign (carla.StopSign): The next stop sign actor.
            vehicle_list (list): A list of vehicle actors in the simulation.
            actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.
            initial_target_speed (float): The initial target speed for the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] 
                    for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            tuple: A tuple containing the brake command (bool), target speed (float), and the updated 
                    speed_reduced_by_obj list.
        """
        ego_speed = self._vehicle.get_velocity().length()
        target_speed = initial_target_speed

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(self._vehicle.bounding_box.location)
        ego_bb_global = carla.BoundingBox(center_ego_bb_global, self._vehicle.bounding_box.extent)
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.visualize == 1:
            self._world.debug.draw_box(box=ego_bb_global,
                                       rotation=ego_bb_global.rotation,
                                       thickness=0.1,
                                       color=self.config.ego_vehicle_bb_color,
                                       life_time=self.config.draw_life_time)

        # Reset hazard flags
        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.walker_hazard = False
        self.stop_sign_close = False

        # Compute if there will be a lane change close
        near_lane_change = self.is_near_lane_change(ego_speed, route_points)

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config.bicycle_frame_rate *
            (self.config.forecast_length_lane_change if near_lane_change else self.config.default_forecast_length))

        # Get future bounding boxes of pedestrians
        if not plant:
            nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers(actor_list, ego_vehicle_location,
                                                                              num_future_frames)

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent(ego_vehicle_transform, ego_speed, num_future_frames,
                                                     initial_target_speed, route_points)

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes(plant, vehicle_list, ego_vehicle_location,
                                                                            num_future_frames, near_lane_change)

        # Compute the leading and trailing vehicle IDs
        leading_vehicle_ids = self._waypoint_planner.compute_leading_vehicles(vehicle_list, self._vehicle.id)
        trailing_vehicle_ids = self._waypoint_planner.compute_trailing_vehicles(vehicle_list, self._vehicle.id)

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading, speed_reduced_by_obj = self.compute_target_speed_wrt_leading_vehicle(
            initial_target_speed, predicted_bounding_boxes, near_lane_change, ego_vehicle_location,
            trailing_vehicle_ids, leading_vehicle_ids, speed_reduced_by_obj, plant)

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, speed_reduced_by_obj = \
            self.compute_target_speeds_wrt_all_actors(initial_target_speed, ego_bounding_boxes,
            predicted_bounding_boxes, near_lane_change, leading_vehicle_ids, trailing_vehicle_ids, speed_reduced_by_obj,
            nearby_pedestrians, nearby_pedestrian_ids)

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light(ego_vehicle_location, ego_speed, 
                                        distance_to_next_traffic_light, next_traffic_light, route_points, 
                                        initial_target_speed)

        # Update the object causing the most speed reduction
        if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_red_light:
            speed_reduced_by_obj = [
                target_speed_red_light, None if next_traffic_light is None else next_traffic_light.type_id,
                None if next_traffic_light is None else next_traffic_light.id, distance_to_next_traffic_light
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign(ego_vehicle_location, ego_speed, next_stop_sign,
                                                                      initial_target_speed, actor_list)
        # Update the object causing the most speed reduction
        if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed_stop_sign:
            speed_reduced_by_obj = [
                target_speed_stop_sign, None if next_stop_sign is None else next_stop_sign.type_id,
                None if next_stop_sign is None else next_stop_sign.id, distance_to_next_stop_sign
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(target_speed_leading, target_speed_bicycle, target_speed_vehicle, target_speed_pedestrian,
                           target_speed_red_light, target_speed_stop_sign)

        # Set the hazard flags based on the target speed and its cause
        if target_speed == target_speed_pedestrian and target_speed_pedestrian != initial_target_speed:
            self.walker_hazard = True
            self.walker_close = True
        elif target_speed == target_speed_red_light and target_speed_red_light != initial_target_speed:
            self.traffic_light_hazard = True
        elif target_speed == target_speed_stop_sign and target_speed_stop_sign != initial_target_speed:
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed, speed_reduced_by_obj

    def forecast_ego_agent(self, current_ego_transform, current_ego_speed, num_future_frames, initial_target_speed,
                           route_points):
        """
        Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
        check subsequently whether the ego vehicle would collide.

        Args:
            current_ego_transform (carla.Transform): The current transform of the ego vehicle.
            current_ego_speed (float): The current speed of the ego vehicle in m/s.
            num_future_frames (int): The number of future frames to forecast.
            initial_target_speed (float): The initial target speed for the ego vehicle.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.

        Returns:
            list: A list of bounding boxes representing the future states of the ego vehicle.
        """
        self._turn_controller.save_state()
        self._waypoint_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [current_ego_transform.location.x, current_ego_transform.location.y, current_ego_transform.location.z])
        heading_angle = np.array([np.deg2rad(current_ego_transform.rotation.yaw)])
        speed = np.array([current_ego_speed])

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(initial_target_speed, current_ego_speed)
        steering = self._turn_controller.step(route_points, speed, location, heading_angle.item())
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(location, heading_angle, speed, action)

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _, _ = self._waypoint_planner.run_step(location)
            steering = self._turn_controller.step(extrapolated_route, speed, location, heading_angle.item())
            throttle = self._longitudinal_controller.get_throttle_extrapolation(initial_target_speed, speed)
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self._vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= self.config.slow_speed_extent_factor_ego if current_ego_speed < \
                            self.config.extent_ego_bbs_speed_threshold else self.config.high_speed_extent_factor_ego_x
            extent.y *= self.config.slow_speed_extent_factor_ego if current_ego_speed < \
                            self.config.extent_ego_bbs_speed_threshold else self.config.high_speed_extent_factor_ego_y

            transform = carla.Transform(carla.Location(x=location[0].item(), y=location[1].item(),
                                                       z=location[2].item()))

            ego_bounding_box = carla.BoundingBox(transform.location, extent)
            ego_bounding_box.rotation = carla.Rotation(pitch=0, yaw=heading_angle_degrees, roll=0)

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()
        self._waypoint_planner.load()

        return future_bounding_boxes

    def forecast_walkers(self, actors, ego_vehicle_location, number_of_future_frames):
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they 
        keep their velocity and direction

        Args:
            actors (carla.ActorList): A list of actors in the simulation.
            ego_vehicle_location (carla.Location): The current location of the ego vehicle.
            number_of_future_frames (int): The number of future frames to forecast.

        Returns:
            tuple: A tuple containing two lists:
                - list: A list of lists, where each inner list contains the future bounding boxes for a pedestrian.
                - list: A list of IDs for the pedestrians whose locations were forecasted.
        """
        nearby_pedestrians_bbs, nearby_pedestrian_ids = [], []

        # Filter pedestrians within the detection radius
        pedestrians = [
            ped for ped in actors.filter("*walker*")
            if ped.get_location().distance(ego_vehicle_location) < self.config.detection_radius
        ]

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array([[ped.get_location().x,
                                          ped.get_location().y,
                                          ped.get_location().z] for ped in pedestrians])
        pedestrian_speeds = np.array([ped.get_velocity().length() for ped in pedestrians])
        pedestrian_speeds = np.maximum(pedestrian_speeds, self.config.min_walker_speed)
        pedestrian_directions = np.array(
            [[ped.get_control().direction.x,
              ped.get_control().direction.y,
              ped.get_control().direction.z] for ped in pedestrians])

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = pedestrian_locations[:, None, :] + np.arange(1, number_of_future_frames + 1)[
            None, :, None] * pedestrian_directions[:,
                                                   None, :] * pedestrian_speeds[:, None,
                                                                                None] / self.config.bicycle_frame_rate

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(pitch=bb.rotation.pitch + transform.rotation.pitch,
                                      yaw=bb.rotation.yaw + transform.rotation.yaw,
                                      roll=bb.rotation.roll + transform.rotation.roll)
            extent = bb.extent
            extent.x = max(self.config.pedestrian_minimum_extent, extent.x)  # Ensure a minimum width
            extent.y = max(self.config.pedestrian_minimum_extent, extent.y)  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(number_of_future_frames):
                location = carla.Location(future_pedestrian_locations[i, j, 0], future_pedestrian_locations[i, j, 1],
                                          future_pedestrian_locations[i, j, 2])

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(pedestrian_future_bboxes)

        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.visualize == 1:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self._world.debug.draw_box(box=bbox,
                                               rotation=bbox.rotation,
                                               thickness=0.1,
                                               color=self.config.pedestrian_forecasted_bbs_color,
                                               life_time=self.config.draw_life_time)

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    def ego_agent_affected_by_red_light(self, ego_vehicle_location, ego_vehicle_speed, distance_to_traffic_light, 
                                        next_traffic_light, route_points, target_speed):
        """
        Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            ego_vehicle_location (carla.Location): The ego vehicle location.
            ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
            distance_to_traffic_light (float): The distance from the ego vehicle to the next traffic light.
            next_traffic_light (carla.TrafficLight or None): The next traffic light in the route.
            route_points (numpy.ndarray): An array of (x, y, z) coordinates representing the route.
            target_speed (float): The target speed for the ego vehicle.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """

        self.close_traffic_lights.clear()

        for light, center, waypoints in self.list_traffic_lights:

            center_loc = carla.Location(center)
            if center_loc.distance(ego_vehicle_location) > self.config.light_radius:
                continue

            for wp in waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bounding_box = carla.Vector3D((wp.lane_width / 2.0) * 0.9, light.trigger_volume.extent.y,
                                                                                         light.trigger_volume.extent.z)
                length_bounding_box = carla.Vector3D(1.5, 1.5, 0.5)

                bounding_box = carla.BoundingBox(wp.transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch=gloabl_rot.pitch,
                                                       yaw=gloabl_rot.yaw,
                                                       roll=gloabl_rot.roll)

                affects_ego = next_traffic_light is not None and light.id==next_traffic_light.id

                self.close_traffic_lights.append([bounding_box, light.state, light.id, affects_ego])

                if self.visualize == 1:
                    if light.state == carla.libcarla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Yellow:
                        color = carla.Color(255, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Off:
                        color = carla.Color(0, 0, 0, 255)
                    else:  # unknown
                        color = carla.Color(0, 0, 255, 255)

                    self._world.debug.draw_box(box=bounding_box,
                                                    rotation=bounding_box.rotation,
                                                    thickness=0.1,
                                                    color=color,
                                                    life_time=0.051)

                    self._world.debug.draw_point(wp.transform.location + carla.Location(z=light.trigger_volume.location.z),
                                                                             size=0.1,
                                                                             color=color,
                                                                             life_time=(1.0 / self.config.carla_fps)+1e-6)

        if next_traffic_light is None or next_traffic_light.state == carla.TrafficLightState.Green:
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(
            desired_speed=target_speed,
            leading_actor_length=0.,
            ego_speed=ego_vehicle_speed,
            leading_actor_speed=0.,
            distance_to_leading_actor=distance_to_traffic_light,
            s0=self.config.idm_red_light_minimum_distance,
            T=self.config.idm_red_light_desired_time_headway
        )

        return target_speed

    def ego_agent_affected_by_stop_sign(self, ego_vehicle_location, ego_vehicle_speed, next_stop_sign, target_speed, 
                                        actor_list):
        """
        Handles the behavior of the ego vehicle when approaching a stop sign.

        Args:
            ego_vehicle_location (carla.Location): The location of the ego vehicle.
            ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
            next_stop_sign (carla.TrafficSign or None): The next stop sign in the route.
            target_speed (float): The target speed for the ego vehicle.
            actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """
        self.close_stop_signs.clear()
        stop_signs = self.get_nearby_object(ego_vehicle_location, actor_list.filter('*traffic.stop*'), self.config.light_radius)
        
        for stop_sign in stop_signs:
            center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
            wp = self.world_map.get_waypoint(center_bb_stop_sign)
            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bounding_box_stop_sign = carla.BoundingBox(center_bb_stop_sign, stop_sign_extent)
            rotation_stop_sign = stop_sign.get_transform().rotation
            bounding_box_stop_sign.rotation = carla.Rotation(pitch=rotation_stop_sign.pitch,
                                                             yaw=rotation_stop_sign.yaw,
                                                             roll=rotation_stop_sign.roll)

            affects_ego = (next_stop_sign is not None and next_stop_sign.id==stop_sign.id and not self.cleared_stop_sign)
            self.close_stop_signs.append([bounding_box_stop_sign, stop_sign.id, affects_ego])

            if self.visualize:
                color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
                self._world.debug.draw_box(box=bounding_box_stop_sign,
                                                                     rotation=bounding_box_stop_sign.rotation,
                                                                     thickness=0.1,
                                                                     color=color,
                                                                     life_time=(1.0 / self.config.carla_fps)+1e-6)

        if next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_to_stop_sign = next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location) \
            .distance(ego_vehicle_location)

        # Reset the stop sign flag if we are farther than 10m away
        if distance_to_stop_sign > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if ego_vehicle_speed < 0.1 and distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
                self.cleared_stop_sign = True

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_to_stop_sign = np.inf if self.cleared_stop_sign else distance_to_stop_sign

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(
            desired_speed=target_speed,
            leading_actor_length=0,  #self._vehicle.bounding_box.extent.x,
            ego_speed=ego_vehicle_speed,
            leading_actor_speed=0.,
            distance_to_leading_actor=distance_to_stop_sign,
            s0=self.config.idm_stop_sign_minimum_distance,
            T=self.config.idm_stop_sign_desired_time_headway)

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    def _dot_product(self, vector1, vector2):
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def cross_product(self, vector1, vector2):
        """
        Calculate the cross product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            carla.Vector3D: The cross product of the two vectors.
        """
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)

    def get_separating_plane(self, relative_position, plane_normal, obb1, obb2):
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(self._dot_product(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (abs(self._dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
                           abs(self._dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)) +
                           abs(self._dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal)))

        obb2_projection = (abs(self._dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
                           abs(self._dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)) +
                           abs(self._dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal)))

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection

    def check_obb_intersection(self, obb1, obb2):
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (self.get_separating_plane(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb1.rotation.get_right_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb1.rotation.get_up_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb2.rotation.get_right_vector(), obb1, obb2) or
                self.get_separating_plane(relative_position, obb2.rotation.get_up_vector(), obb1, obb2)):

            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2)):

            return False

        # If no separating plane is found, the OBBs intersect
        return True

    def _get_angle_to(self, current_position, current_heading, target_position):
        """
        Calculate the angle (in degrees) from the current position and heading to a target position.

        Args:
            current_position (list): A list of (x, y) coordinates representing the current position.
            current_heading (float): The current heading angle in radians.
            target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

        Returns:
            float: The angle (in degrees) from the current position and heading to the target position.
        """
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        # Calculate the vector from the current position to the target position
        position_delta = target_position - current_position

        # Calculate the dot product of the position delta vector and the current heading vector
        aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
        aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

        # Calculate the angle (in radians) from the current heading to the target position
        angle_radians = -math.atan2(-aim_y, aim_x)

        # Convert the angle from radians to degrees
        angle_degrees = np.float_(math.degrees(angle_radians))

        return angle_degrees

    def get_nearby_object(self, ego_vehicle_position, all_actors, search_radius):
        """
        Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

        Args:
            ego_vehicle_position (carla.Location): The position of the ego vehicle.
            all_actors (list): A list of all actors.
            search_radius (float): The radius (in meters) around the ego vehicle to search for nearby actors.

        Returns:
            list: A list of actors within the specified search radius.
        """
        nearby_objects = []
        for actor in all_actors:
            try:
                trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            except:
                print("Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)")
                print("Skipping this object.")
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x,
                                                    y=trigger_box_global_pos.y,
                                                    z=trigger_box_global_pos.z)

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(ego_vehicle_position) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects
