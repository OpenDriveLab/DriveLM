"""
Config class containing hyperparameters for building models. After the report, IDM parameters were updated to reduce
creeping, improving imitation learning performance. This script provides the updated parameters, while the report
parameters are in config_report.py. By default we use these parameters.
"""

import carla
import numpy as np


class GlobalConfig:
    """
    Config class that contains all the hyperparameters needed to build any model.
    """
    # Weather azimuths used for data collection
    # Defined outside of init because carla objects can't be pickled
    weathers = {
        'Clear': carla.WeatherParameters.ClearNoon,
        'Cloudy': carla.WeatherParameters.CloudySunset,
        'Wet': carla.WeatherParameters.WetSunset,
        'MidRain': carla.WeatherParameters.MidRainSunset,
        'WetCloudy': carla.WeatherParameters.WetCloudySunset,
        'HardRain': carla.WeatherParameters.HardRainNoon,
        'SoftRain': carla.WeatherParameters.SoftRainSunset,
    }

    def __init__(self):
        """ base architecture configurations """
        # -----------------------------------------------------------------------------
        # Autopilot
        # -----------------------------------------------------------------------------
        # Frame rate used for the bicycle models in the autopilot
        self.bicycle_frame_rate = 20
        self.target_speed_slow = 5.0  # Speed at junctions, m/s
        self.target_speed_fast = 8.0  # Speed outside junctions, m/s
        self.target_speed_walker = 2.0  # Speed when there is a pedestrian close by
        self.steer_noise = 1e-3  # Noise added to expert steering angle
        # Amount of seconds we look into the future to predict collisions (>= 1 frame)
        self.extrapolation_seconds_lane_change = 2.2
        # Amount of seconds we look into the future to predict collisions at junctions
        self.extrapolation_seconds = 2.
        # Distance of obstacles (in meters) in which we will check for collisions
        self.detection_radius = 50.0
        # Distance of walker (in meters) in which we will check for collisions
        self.distance_for_walker_speed = 1.0  # set that dynamically later
        # Variables for detecting stuck vehicles
        self.stuck_buffer_size = 30
        self.stuck_vel_threshold = 0.1
        self.stuck_throttle_threshold = 0.1
        self.stuck_brake_threshold = 0.1
        self.num_route_points_saved = 20  # Number of future route points we save per step.
        # Since the traffic manager can scratch cars, we reduce the safety box width
        # in the forecasting to model that.
        self.traffic_safety_box_width_multiplier = 0.5
        # Bounding box of cars with opened doors doesn't increase, hence multiply its extend
        self.vehicle_with_opened_door_y_extend_multiplier = 2.0
        # Safety distance that we model other traffic participants to keep.
        self.traffic_safety_box_length = 1.9
        # Distance of traffic lights considered relevant (in meters)
        self.light_radius = 15.0
        # Bounding boxes in this radius around the car will be saved in the dataset.
        self.bb_save_radius = 40.0
        # Number of meters we will keep a distance from vehicles in front of us.
        self.safety_box_safety_margin = 2.5
        # Whether the forecast will consider that vehicles yield to other cars in front of them.
        self.model_interactions = False
        # Distance to keep to the traffic light if it's red
        self.safety_distance_red_light = 7
        # Ratio between the the speed limit / curvature dependent speed limit and the target speed.
        # By default the other vehicles drive with 70 % of the speed limit. To avoid collisions we have to be a bit
        # faster.
        self.ratio_target_speed_limit = 0.72
        # Maximum number of ticks the agent doesn't take any action. The maximum is 179 and it's speed must be >0.1.
        # After taking 180 ticks no action the route ends with an AgentBlockTest infraction.
        self.max_blocked_ticks = 170
        # To detect a lane change we check the braking distance plus these additional default meters
        self.default_distance_to_check_for_lane_change = 2
        # Minimum distance in meters to check for a close lane change
        self.minimum_distance_to_check_for_lane_change = 10
        # Multiplier for the safety box safety margin when standing still to the leading vehicle
        self.safety_distance_multiplier_to_leading_vehicle_when_standing_still = 3.3
        # Multiplier for the safety box safety margin when driving to the leading vehicle
        self.safety_distance_multiplier_to_leading_vehicle_when_driving = 2.0
        # Multiplier for the safety box safety margin when standing still to other vehicles
        self.safety_distance_multiplier_to_other_vehicles_when_standing_still = 1.2
        # Multiplier for the safety box safety margin when driving to other vehicles
        self.safety_distance_multiplier_to_other_vehicles_when_driving = 1.7
        # Distance the leading vehicle has to drive before the agent starts accelerating when standing still
        self.needed_driving_distance_of_leading_vehicle_before_accelerating = 0.2
        # Number of meters we will keep a distance from walkers in front of us.
        self.safety_box_safety_margin_walker = 2 * self.safety_box_safety_margin
        # Multiplier for the length of the bounding boxes of the other vehicles when they are driving
        self.other_vehicles_bounding_box_length_multiplier_when_driving = 1.5
        # Multiplier for the width of the bounding boxes of the other vehicles
        self.other_vehicles_bounding_box_width_multiplier = 1.1
        # Multiplier for the width of the bounding boxes of the ego vehicle
        self.ego_vehicles_bounding_box_multiplier_width = 1.25
        # Multiplier for the length of the bounding boxes of the ego vehicle
        self.ego_vehicles_bounding_box_multiplier_length = 1.25
        # Minimum walker speed
        self.min_walker_speed = 0.5
        # Maximum angle difference between the bounding box to distinguish between leading vehicles and other vehicles
        self.max_angle_difference_of_leading_vehicle_and_ego = 27
        # Minimum bounding box length extend of walkers
        self.minimum_walkers_extent_length = 1.
        # Minimum bounding box width extend of walkers
        self.minimum_walkers_extent_width = 1.
        # Time in seconds to draw the things during debugging.
        self.draw_life_time = 0.051
        # Colors used for drawing during debugging
        self.future_route_color = carla.Color(0, 1, 0)
        self.other_vehicles_forecasted_bbs_color = carla.Color(0, 0, 1, 1)
        self.leading_vehicle_color = carla.Color(1, 0, 0, 0)
        self.trailing_vehicle_color = carla.Color(1, 1, 1, 0)
        self.ego_vehicle_bb_color = carla.Color(0, 0, 0, 1)
        self.pedestrian_forecasted_bbs_color = carla.Color(0, 0, 1, 1)
        self.red_traffic_light_color = carla.Color(0, 1, 0, 1)
        self.green_traffic_light_color = carla.Color(1, 0, 0, 1)
        self.cleared_stop_sign_color = carla.Color(0, 1, 0, 1)
        self.uncleared_stop_sign_color = carla.Color(1, 0, 0, 1)
        self.ego_vehicle_forecasted_bbs_hazard_color = carla.Color(1, 0, 0, 0)
        self.ego_vehicle_forecasted_bbs_normal_color = carla.Color(0, 1, 0, 0)
        # Points sampled per meter when interpolating route.
        self.points_per_meter = 10
        # FPS of the simulation
        self.fps = 20.
        # Inverse of the FPS
        self.fps_inv = 1. / self.fps
        # Distance to the stop sign, when the previous stop sign is uncleared
        self.unclearing_distance_to_stop_sign = 10
        # Distance to the stop sign, when the previous stop sign is cleared
        self.clearing_distance_to_stop_sign = 3.
        # IDM minimum distance for stop signs
        self.idm_stop_sign_minimum_distance = 2.
        # IDM desrired time headway for stop signs
        self.idm_stop_sign_desired_time_headway = 0.1
        # IDM minimum distance for red lights
        self.idm_red_light_minimum_distance = 6.
        # IDM desrired time headway for red lights
        self.idm_red_light_desired_time_headway = 0.1
        # IDM minimum distance for pedestrians
        self.idm_pedestrian_minimum_distance = 4.
        # IDM desrired time headway for pedestrians
        self.idm_pedestrian_desired_time_headway = 0.25
        # IDM minimum distance for bicycles
        self.idm_bicycle_minimum_distance = 4.
        # IDM desrired time headway for bicycles
        self.idm_bicycle_desired_time_headway = 0.25
        # IDM minimum distance for leading vehicles
        self.idm_leading_vehicle_minimum_distance = 4.
        # IDM desrired time headway for leading vehicles
        self.idm_leading_vehicle_time_headway = 0.25
        # IDM minimum distance for two way scenarios
        self.idm_two_way_scenarios_minimum_distance = 2.
        # IDM desrired time headway for two way scenarios
        self.idm_two_way_scenarios_time_headway = 0.1
        # Boundary time - the integration won’t continue beyond it.
        self.idm_t_bound = 0.05
        # IDM maximum accelaration parameter per frame
        self.idm_maximum_acceleration = 24.
        # The following parameters were determined by measuring the vehicle's braking performance.
        # IDM maximum deceleration parameter per frame while driving slow
        self.idm_comfortable_braking_deceleration_low_speed = 8.7
        # IDM maximum deceleration parameter per frame while driving fast
        self.idm_comfortable_braking_deceleration_high_speed = 3.72
        # Threshold to determine, when to use idm_comfortable_braking_deceleration_low_speed and 
        # idm_comfortable_braking_deceleration_high_speed
        self.idm_comfortable_braking_deceleration_threshold = 6.02
        # IDM acceleration exponent (default = 4.)
        self.idm_acceleration_exponent = 4.
        # Minimum extent for pedestrian during bbs forecasting
        self.pedestrian_minimum_extent = 1.5
        # Factor to increase the ego vehicles bbs in driving direction during forecasting
        # when speed > extent_ego_bbs_speed_threshold
        self.high_speed_extent_factor_ego_x = 1.3
        # Factor to increase the ego vehicles bbs in y direction during forecasting
        # when speed > extent_ego_bbs_speed_threshold
        self.high_speed_extent_factor_ego_y = 1.2
        # Threshold to decide, when which bbs increase factor is used
        self.extent_ego_bbs_speed_threshold = 5
        # Forecast length in seconds when near a lane change
        self.forecast_length_lane_change = 1.1
        # Forecast length in seconds when not near a lane change
        self.default_forecast_length = 2.0
        # Factor to increase the ego vehicles bbs during forecasting when speed < extent_ego_bbs_speed_threshold
        self.slow_speed_extent_factor_ego = 1.
        # Speed threshold to select which factor is used during other vehicle bbs forecasting
        self.extent_other_vehicles_bbs_speed_threshold = 1.
        # Minimum extent of bbs, while forecasting other vehicles
        self.high_speed_min_extent_y_other_vehicle = 1.0
        # Extent factor to scale bbs during forecasting other vehicles in y direction
        self.high_speed_extent_y_factor_other_vehicle = 1.3
        # Extent factor to scale bbs during forecasting other vehicles in x direction
        self.high_speed_extent_x_factor_other_vehicle = 1.5
        # Minimum extent factor to scale bbs during forecasting other vehicles in x direction
        self.high_speed_min_extent_x_other_vehicle = 1.2
        # Minimum extent factor to scale bbs during forecasting other vehicles in x direction during lane changes to
        # account fore forecasting inaccuracies
        self.high_speed_min_extent_x_other_vehicle_lane_change = 2.0
        # Safety distance to be added to emergency braking distance
        self.braking_distance_calculation_safety_distance = 10
        # Minimum speed in m/s to prevent rolling back, when braking no throttle is applied
        self.minimum_speed_to_prevent_rolling_back = 0.5
        # Maximum seed in junctions in m/s
        self.max_speed_in_junction = 64 / 3.6
        # Lookahead distance to check, whether the ego is close to a junction
        self.max_lookahead_to_check_for_junction = 30 * self.points_per_meter
        # Distance of the first checkpoint for TF++
        self.tf_first_checkpoint_distance = int(2.5 * self.points_per_meter)
        # Parameters to calculate how much the ego agent needs to cover a given distance. Values are taken from
        # the kinematic bicycle model
        self.compute_min_time_to_cover_distance_params = np.array([0.00904221, 0.00733342, -0.03744807, 0.0235038])
        # Distance to check for road_id/lane_id for RouteObstacle scenarios
        self.previous_road_lane_retrieve_distance = 100
        # Safety distance during checking if the path is free for RouteObstacle scenarios
        self.check_path_free_safety_distance = 10
        # Safety time headway during checking if the path is free for RouteObstacle scenarios
        self.check_path_free_safety_time = 0.2
        # Transition length for change lane in scenario ConstructionObstacle
        self.transition_smoothness_factor_construction_obstacle = 10.5 * self.points_per_meter
        # Check in x meters if there is lane change ahead
        self.minimum_lookahead_distance_to_compute_near_lane_change = 20 * self.points_per_meter
        # Check if did a lane change in the previous x meters
        self.check_previous_distance_for_lane_change = 15 * self.points_per_meter
        # Draw x meters of the route during debugging
        self.draw_future_route_till_distance = 50 * self.points_per_meter
        # Default minimum distance to process the route obstacle scenarios
        self.default_max_distance_to_process_scenario = 50
        # Minimum distance to process HazardAtSideLane
        self.max_distance_to_process_hazard_at_side_lane = 25
        # Minimum distance to process HazardAtSideLaneTwoWays
        self.max_distance_to_process_hazard_at_side_lane_two_ways = 10
        # Transition length for sceneario AccidentTwoWays to change lanes
        self.transition_length_accident_two_ways = int(4 * self.points_per_meter)
        # Transition length for sceneario ConstructionObstacleTwoWays to change lanes
        self.transition_length_construction_obstacle_two_ways = int(4 * self.points_per_meter)
        # Transition length for sceneario ParkedObstacleTwoWays to change lanes
        self.transition_length_parked_obstacle_two_ways = int(4 * self.points_per_meter)
        # Transition length for sceneario VehicleOpensDoorTwoWays to change lanes
        self.transition_length_vehicle_opens_door_two_ways = int(4 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario AccidentTwoWays before the obstacle
        self.add_before_accident_two_ways = int(-1.5 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario ConstructionObstacleTwoWays
        # before the obstacle
        self.add_before_construction_obstacle_two_ways = int(1.5 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario ParkedObstacleTwoWays before the obstacle
        self.add_before_parked_obstacle_two_ways = int(-0.5 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario VehicleOpensDoorTwoWays before the obstacle
        self.add_before_vehicle_opens_door_two_ways = int(-2.0 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario AccidentTwoWays after the obstacle
        self.add_after_accident_two_ways = int(-1.5 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario ConstructionObstacleTwoWays
        # after the obstacle
        self.add_after_construction_obstacle_two_ways = int(1.5 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario ParkedObstacleTwoWays after the obstacle
        self.add_after_parked_obstacle_two_ways = int(-0.5 * self.points_per_meter)
        # Increase overtaking maneuver by distance in meters in the scenario VehicleOpensDoorTwoWays after the obstacle
        self.add_after_vehicle_opens_door_two_ways = int(-2.0 * self.points_per_meter)
        # How much to drive to the center of the opposite lane while handling the scenario AccidentTwoWays
        self.factor_accident_two_ways = 1.
        # How much to drive to the center of the opposite lane while handling the scenario ConstructionObstacleTwoWays
        self.factor_construction_obstacle_two_ways = 1.
        # How much to drive to the center of the opposite lane while handling the scenario ParkedObstacleTwoWays
        self.factor_parked_obstacle_two_ways = 0.6
        # How much to drive to the center of the opposite lane while handling the scenario VehicleOpensDoorTwoWays
        self.factor_vehicle_opens_door_two_ways = 0.475
        # Maximum distance to start the overtaking maneuver
        self.max_distance_to_overtake_two_way_scnearios = int(8 * self.points_per_meter)
        # Overtaking speed in m/s for vehicle opens door two ways scenarios
        self.overtake_speed_vehicle_opens_door_two_ways = 40. / 3.6
        # Default overtaking speed in m/s for all route obstacle scenarios
        self.default_overtake_speed = 50. / 3.6
        # Distance in meters at which two ways scenarios are considered finished
        self.distance_to_delete_scenario_in_two_ways = int(2 * self.points_per_meter)
        # -----------------------------------------------------------------------------
        # Longitudinal Linear Regression controller
        # -----------------------------------------------------------------------------
        # These parameters are tuned with Bayesian Optimization on a test track
        # Minimum threshold for target speed (< 1 km/h) for longitudinal linear regression controller.
        self.longitudinal_linear_regression_minimum_target_speed = 0.278
        # Coefficients of the linear regression model used for throttle calculation.
        self.longitudinal_linear_regression_params = np.array([
            1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335, 1.556497522998393,
            -0.7013479734904027, 1.031266635497984
        ])
        # Maximum acceleration rate (approximately 1.9 m/tick) for the longitudinal linear regression controller.
        self.longitudinal_linear_regression_maximum_acceleration = 1.89
        # Maximum deceleration rate (approximately -4.82 m/tick) for the longitudinal linear regression controller.
        self.longitudinal_linear_regression_maximum_deceleration = -4.82
        # -----------------------------------------------------------------------------
        # Longitudinal PID controller
        # -----------------------------------------------------------------------------
        # These parameters are tuned with Bayesian Optimization on a test track
        # Gain factor for proportional control for longitudinal pid controller.
        self.longitudinal_pid_proportional_gain = 1.0016429066823955
        # Gain factor for derivative control for longitudinal pid controller.
        self.longitudinal_pid_derivative_gain = 1.5761818624794222
        # Gain factor for integral control for longitudinal pid controller.
        self.longitudinal_pid_integral_gain = 0.2941563856687906
        # Maximum length of the window for cumulative error for longitudinal pid controller.
        self.longitudinal_pid_max_window_length = 0
        # Scaling factor for speed error based on current speed for longitudinal pid controller.
        self.longitudinal_pid_speed_error_scaling = 0.0
        # Ratio to determine when to apply braking for longitudinal pid controller.
        self.longitudinal_pid_braking_ratio = 1.0324622059220139
        # Minimum threshold for target speed (< 1 km/h) for longitudinal pid controller.
        self.longitudinal_pid_minimum_target_speed = 0.278
        # -----------------------------------------------------------------------------
        # Lateral PID controller
        # -----------------------------------------------------------------------------
        # These parameters are tuned with Bayesian Optimization on a test track
        # The proportional gain for the lateral PID controller.
        self.lateral_pid_kp = 3.118357247806046
        # The derivative gain for the lateral PID controller.
        self.lateral_pid_kd = 1.3782508892109167
        # The integral gain for the lateral PID controller.
        self.lateral_pid_ki = 0.6406067986034124
        # The scaling factor used in the calculation of the lookahead distance based on the current speed.
        self.lateral_pid_speed_scale = 0.9755321901954155
        # The offset used in the calculation of the lookahead distance based on the current speed.
        self.lateral_pid_speed_offset = 1.9152884533402488
        # The default lookahead distance for the lateral PID controller.
        self.lateral_pid_default_lookahead = 2.4 * self.points_per_meter
        # The speed threshold (in km/h) for switching between the default and variable lookahead distance.
        self.lateral_pid_speed_threshold = 2.3150102938235136 * self.points_per_meter
        # The size of the sliding window used to store the error history for the lateral PID controller.
        self.lateral_pid_window_size = 6
        # The minimum allowed lookahead distance for the lateral PID controller.
        self.lateral_pid_minimum_lookahead_distance = 2.4 * self.points_per_meter
        # The maximum allowed lookahead distance for the lateral PID controller.
        self.lateral_pid_maximum_lookahead_distance = 10.5 * self.points_per_meter
        # -----------------------------------------------------------------------------
        # Kinematic Bicycle Model
        # -----------------------------------------------------------------------------
        #  Time step for the model (20 frames per second).
        self.time_step = 1. / 20.
        # Kinematic bicycle model parameters tuned from World on Rails.
        # Distance from the rear axle to the front axle of the vehicle.
        self.front_wheel_base = -0.090769015
        # Distance from the rear axle to the center of the rear wheels.
        self.rear_wheel_base = 1.4178275
        # Gain factor for steering angle to wheel angle conversion.
        self.steering_gain = 0.36848336
        # Deceleration rate when braking (m/s^2) of other vehicles.
        self.brake_acceleration = -4.952399
        # Acceleration rate when throttling (m/s^2) of other vehicles.
        self.throttle_acceleration = 0.5633837
        # Tuned parameters for the polynomial equations modeling speed changes
        # Numbers are tuned parameters for the polynomial equations below using
        # a dataset where the car drives on a straight highway, accelerates to
        # and brakes again
        # Coefficients for polynomial equation estimating speed change with throttle input for ego model.
        self.throttle_values = np.array([
            9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
            -1.05996152e-03, 6.71079346e-04
        ])
        # Coefficients for polynomial equation estimating speed change with brake input for the ego model.
        self.brake_values = np.array([
            9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
            -4.91381935e-12
        ])
        # Minimum throttle value that has an affect during forecasting the ego vehicle.
        self.throttle_threshold_during_forecasting = 0.3
        # -----------------------------------------------------------------------------
        # Privileged Route Planner
        # -----------------------------------------------------------------------------
        # Max distance to search ahead for updating ego route index  in meters.
        self.ego_vehicles_route_point_search_distance = 4 * self.points_per_meter
        # Length to extend lane shift transition for YieldToEmergencyVehicle  in meters.
        self.lane_shift_extension_length_for_yield_to_emergency_vehicle = 20 * self.points_per_meter
        # Distance over which lane shift transition is smoothed  in meters.
        self.transition_smoothness_distance = 8 * self.points_per_meter
        # Distance over which lane shift transition is smoothed for InvadingTurn  in meters.
        self.route_shift_start_distance_invading_turn = 15 * self.points_per_meter
        self.route_shift_end_distance_invading_turn = 10 * self.points_per_meter
        # Margin from fence when shifting route in InvadingTurn.
        self.fence_avoidance_margin_invading_turn = 0.3
        # Minimum lane width to avoid early lane changes.
        self.minimum_lane_width_threshold = 2.5
        # Spacing for checking and updating speed limits  in meters.
        self.speed_limit_waypoints_spacing_check = 5 * self.points_per_meter
        # Max distance on route for detecting leading vehicles.
        self.leading_vehicles_max_route_distance = 2.5
        # Max angle difference for detecting leading vehicles  in meters.
        self.leading_vehicles_max_route_angle_distance = 35.
        # Max radius for detecting any leading vehicles in meters.
        self.leading_vehicles_maximum_detection_radius = 80 * self.points_per_meter
        # Max distance on route for detecting trailing vehicles.
        self.trailing_vehicles_max_route_distance = 3.0
        # Max route distance for trailing vehicles after lane change.
        self.trailing_vehicles_max_route_distance_lane_change = 6.0
        # Max radius for detecting any trailing vehicles in meters.
        self.tailing_vehicles_maximum_detection_radius = 80 * self.points_per_meter
        # Max distance to check for lane changes when detecting trailing vehicles in meters.
        self.max_distance_lane_change_trailing_vehicles = 15 * self.points_per_meter
        # Distance to extend the end of the route in meters. This makes sure we always have checkpoints,
        # also at the end of the route.
        self.extra_route_length = 50
        # -----------------------------------------------------------------------------
        # DataAgent
        # -----------------------------------------------------------------------------
        # Data agent hyperparameters
        self.azimuths = [45.0 * i for i in range(8)]
        # Daytimes used for data collection
        self.daytimes = {
            'Night': -80.0,
            'Twilight': 0.0,
            'Dawn': 5.0,
            'Sunset': 15.0,
            'Morning': 35.0,
            'Noon': 75.0,
        }
        # Max and min values by which the augmented camera is shifted left and right
        self.camera_translation_augmentation_min = -1.0
        self.camera_translation_augmentation_max = 1.0
        # Max and min values by which the augmented camera is rotated around the yaw
        # Numbers are in degree
        self.camera_rotation_augmentation_min = -5.0
        self.camera_rotation_augmentation_max = 5.0
        # Every data_save_freq frame the data is stored during training
        # Set to one for backwards compatibility. Released dataset was collected with 5
        self.data_save_freq = 5
        # LiDAR compression parameters
        self.point_format = 0  # LARS point format used for storing
        self.point_precision = 0.001  # Precision up to which LiDAR points are stored

        # -----------------------------------------------------------------------------
        # Sensor config
        # -----------------------------------------------------------------------------
        self.lidar_pos = [0.0, 0.0, 2.5]  # x, y, z mounting position of the LiDAR
        self.lidar_rot = [0.0, 0.0, -90.0]  # Roll Pitch Yaw of LiDAR in degree
        self.lidar_rotation_frequency = 10  # Number of Hz at which the Lidar operates
        # Number of points the LiDAR generates per second.
        # Change in proportion to the rotation frequency.
        self.lidar_points_per_second = 600000
        self.camera_pos = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
        self.camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

        # Therefore their size is smaller
        self.camera_width = 1024  # Camera width in pixel during data collection
        self.camera_height = 512  # Camera height in pixel during data collection
        self.camera_fov = 110

        # -----------------------------------------------------------------------------
        # Dataloader
        # -----------------------------------------------------------------------------
        self.carla_fps = 20  # Simulator Frames per second
        self.seq_len = 1  # input timesteps
        # use different seq len for image and lidar
        self.img_seq_len = 1
        self.lidar_seq_len = 1
        # Number of initial frames to skip during data loading
        self.skip_first = int(2.5 * self.carla_fps) // self.data_save_freq
        self.pred_len = int(2.0 * self.carla_fps) // self.data_save_freq  # number of future waypoints predicted
        # Width and height of the LiDAR grid that the point cloud is voxelized into.
        self.lidar_resolution_width = 256
        self.lidar_resolution_height = 256
        # Number of LiDAR hits a bounding box needs for it to be a valid label
        self.num_lidar_hits_for_detection = 7
        # How many pixels make up 1 meter.
        # 1 / pixels_per_meter = size of pixel in meters
        self.pixels_per_meter = 2.0
        # Max number of LiDAR points per pixel in voxelized LiDAR
        self.hist_max_per_pixel = 5
        # Height at which the LiDAR points are split into the 2 channels.
        # Is relative to lidar_pos[2]
        self.lidar_split_height = 0.2
        self.realign_lidar = True
        self.use_ground_plane = False
        # Max and minimum LiDAR ranges used for voxelization
        self.min_x = -32
        self.max_x = 32
        self.min_y = -32
        self.max_y = 32
        self.min_z = -4
        self.max_z = 4
        self.min_z_projection = -10
        self.max_z_projection = 14
        # Bin in for the target speed one hot vector.
        self.target_speed_bins = [
            self.target_speed_walker + 0.1, self.target_speed_slow + 0.1, self.target_speed_fast + 0.1
        ]
        # Index 0 is the brake action
        self.target_speeds = [0.0, self.target_speed_walker, self.target_speed_slow, self.target_speed_fast]
        # Angle bin thresholds
        self.angle_bins = [-0.375, -0.125, 0.125, 0.375]
        # Discrete steering angles
        self.angles = [-0.5, -0.25, 0.0, 0.25, 0.5]
        # Whether to estimate the class weights or use the default from the config.
        self.estimate_class_distributions = False
        self.estimate_semantic_distribution = False
        # Class weights applied to the cross entropy losses
        # Computed from the v07 all dataset
        self.target_speed_weights = [0.866605263873406, 7.4527377240841775, 1.2281629310898465, 0.5269622904065803]
        self.angle_weights = [
            204.25901201602136, 7.554315623148331, 0.21388916461734406, 5.476446162657503, 207.86684782608697
        ]
        # We don't use weighting here
        self.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # # -----------------------------------------------------------------------------
        # # DataAgent
        # # -----------------------------------------------------------------------------
        self.augment = 1  # Whether to use rotation and translation augmentation

        # -----------------------------------------------------------------------------
        # Logger
        # -----------------------------------------------------------------------------
        self.logging_freq = 10  # Log every 10 th frame
        self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
        self.route_points = 10  # Number of route points to render in logger
        # Minimum distance to the next waypoint in the logger
        self.log_route_planner_min_distance = 4.0

        # -----------------------------------------------------------------------------
        # Agent file
        # -----------------------------------------------------------------------------
        self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
        # Iou threshold used for non-maximum suppression on the Bounding Box
        # predictions for the ensembles
        self.iou_treshold_nms = 0.2
        self.route_planner_min_distance = 7.5
        self.route_planner_max_distance = 50.0
        # Min distance to the waypoint in the dense rout that the expert is trying to follow
        self.dense_route_planner_min_distance = 3.5
        self.dense_route_planner_max_distance = 50.0
        self.action_repeat = 1  # Number of times we repeat the networks action.
        # Number of frames after which the creep controller starts triggering. 1100 is larger than wait
        # time at red light.
        self.stuck_threshold = 1100 / self.action_repeat
        self.creep_duration = 20 / self.action_repeat  # Number of frames we will creep forward
        self.creep_throttle = 0.4
        # CARLA needs some time to initialize in which the cars actions are blocked.
        # Number tuned empirically
        self.inital_frames_delay = 2.0 / self.carla_frame_rate

        # Extent of the ego vehicles bounding box
        self.ego_extent_x = 2.4508416652679443
        self.ego_extent_y = 1.0641621351242065
        self.ego_extent_z = 0.7553732395172119

        # Size of the safety box
        self.safety_box_z_min = 0.5
        self.safety_box_z_max = 1.5

        self.safety_box_y_min = -self.ego_extent_y * 0.8
        self.safety_box_y_max = self.ego_extent_y * 0.8

        self.safety_box_x_min = self.ego_extent_x
        self.safety_box_x_max = self.ego_extent_x + 2.5

        # Probability 0 - 1. If the confidence in the brake action is higher than this
        # value brake is chosen as the action.
        self.brake_uncertainty_threshold = 0.5
        self.checkpoint_buffer_len = 10  # Number of time steps that we use for route consistency
