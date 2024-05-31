"""
This file implements a lateral PID controller and its super class, which enables furhter lateral
controller implementations.
"""

import numpy as np

class LateralController:
    """
    Base class for lateral controllers.
    """

    def __init__(self, config):
        self.config = config

    def compute_steering(self, route_points, current_speed, vehicle_position, vehicle_heading):
        """
        Computes the steering angle based on the route, current speed, vehicle position, and heading.

        Args:
            route_points (numpy.ndarray): Array of (x, y) coordinates representing the route.
            current_speed (float): Current speed of the vehicle in m/s.
            vehicle_position (numpy.ndarray): Array of (x, y) coordinates representing the vehicle's position.
            vehicle_heading (float): Current heading angle of the vehicle in radians.

        Returns:
            float: Computed steering angle in the range [-1.0, 1.0].
        """
        pass

    def save_state(self):
        """
        Saves the current state of the controller. Useful during forecasting.
        """
        pass

    def load_state(self):
        """
        Loads the previously saved state of the controller. Useful during forecasting.
        """
        pass


class LateralPIDController(LateralController):
    """
    Lateral controller based on a Proportional-Integral-Derivative (PID) controller.
    """

    def __init__(self, config):
        super().__init__(config)

        self.lateral_pid_kp = self.config.lateral_pid_kp
        self.lateral_pid_kd = self.config.lateral_pid_kd
        self.lateral_pid_ki = self.config.lateral_pid_ki

        self.lateral_pid_speed_scale = self.config.lateral_pid_speed_scale
        self.lateral_pid_speed_offset = self.config.lateral_pid_speed_offset
        self.lateral_pid_default_lookahead = self.config.lateral_pid_default_lookahead
        self.lateral_pid_speed_threshold = self.config.lateral_pid_speed_threshold

        self.lateral_pid_window_size = self.config.lateral_pid_window_size
        self.lateral_pid_minimum_lookahead_distance = self.config.lateral_pid_minimum_lookahead_distance
        self.lateral_pid_maximum_lookahead_distance = self.config.lateral_pid_maximum_lookahead_distance

        # The following lists are used as deques
        self.error_history = []  # Sliding window to store past errors
        self.saved_error_history = []  # Saved error history for state loading

    def step(self, route_points, current_speed, vehicle_position, vehicle_heading, inference_mode=False):
        """
        Computes the steering angle based on the route, current speed, vehicle position, and heading.

        Args:
            route_points (numpy.ndarray): Array of (x, y) coordinates representing the route.
            current_speed (float): Current speed of the vehicle in m/s.
            vehicle_position (numpy.ndarray): Array of (x, y) coordinates representing the vehicle's position.
            vehicle_heading (float): Current heading angle of the vehicle in radians.
            inference_mode (bool): Controls whether to TF or PDM-Lite executes this method.

        Returns:
            float: Computed steering angle in the range [-1.0, 1.0].
        """
        current_speed_kph = current_speed * 3.6  # Convert speed from m/s to km/h

        # Compute the lookahead distance based on the current speed
        # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
        if inference_mode:
            lookahead_distance = self.lateral_pid_speed_scale * current_speed + self.lateral_pid_speed_offset
            lookahead_distance = np.clip(lookahead_distance, self.lateral_pid_minimum_lookahead_distance, \
                self.lateral_pid_maximum_lookahead_distance) / self.config.route_points  # range [2.4, 10.5]
            lookahead_distance = lookahead_distance - 2  # range [0.4, 8.5]
        else:
            lookahead_distance = self.lateral_pid_speed_scale * current_speed_kph + self.lateral_pid_speed_offset
            lookahead_distance = np.clip(lookahead_distance, self.lateral_pid_minimum_lookahead_distance,
                                         self.lateral_pid_maximum_lookahead_distance)

        lookahead_distance = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Calculate the desired heading vector from the lookahead point
        desired_heading_vec = route_points[lookahead_distance] - vehicle_position
        desired_heading_angle = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])

        # Calculate the heading error
        heading_error = (desired_heading_angle - vehicle_heading) % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

        # Scale the heading error (leftover from a previous implementation)
        heading_error = heading_error * 180. / np.pi / 90.

        # Update the error history. Only use the last lateral_pid_window_size errors like in a deque.
        self.error_history.append(heading_error)
        self.error_history = self.error_history[-self.lateral_pid_window_size:]

        # Calculate the derivative and integral terms
        derivative = 0.0 if len(self.error_history) == 1 else self.error_history[-1] - self.error_history[-2]
        integral = np.mean(self.error_history)

        # Compute the steering angle using the PID control law
        steering = np.clip(
            self.lateral_pid_kp * heading_error + self.lateral_pid_kd * derivative + self.lateral_pid_ki * integral,
            -1., 1.).item()

        return steering

    def save_state(self):
        """
        Saves the current state of the controller by copying the error history.
        """
        self.saved_error_history = self.error_history.copy()

    def load_state(self):
        """
        Loads the previously saved state of the controller by restoring the saved error history.
        """
        self.error_history = self.saved_error_history.copy()
