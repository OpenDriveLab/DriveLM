"""
This file holds class implementations for a longitudinal PID controller and a linear regression model as
longitudinal controller.
"""

import numpy as np

class LongitudinalController:
    """
    Base class for longitudinal controller.
    """

    def __init__(self, config):
        """
        Constructor of the longitudinal controller, which saves the configuration object for the hyperparameters.

        Args:
            config (GlobalConfig): Object of the config for hyperparameters.
        """
        self.config = config

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        """
        Get the throttle and brake values based on the target speed, current speed, and hazard brake condition.
        This method is used to calculate the throttle / brake values for driving.

        Args:
            hazard_brake (bool): Flag indicating whether to apply hazard braking.
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            tuple: A tuple containing the throttle and brake values.
        """
        pass

    def get_throttle_extrapolation(self, target_speed, current_speed):
        """
        Get the throttle value for the given target speed and current speed, assuming no hazard brake condition.
        This method is used for forecasting.

        Args:
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            float: The throttle value.
        """
        pass

    def save(self):
        """
        Save the current state of the controller.
        """
        pass

    def load(self):
        """
        Load the previously saved state of the controller.
        """
        pass


class LongitudinalPIDController(LongitudinalController):
    """
    This class was used for the ablations. Currently, we use the linear regression controller for longitudinal
    control by default.
    """

    def __init__(self, config):
        super().__init__(config)

        # These parameters are tuned with Bayesian Optimization on a test track
        self.proportional_gain = self.config.longitudinal_pid_proportional_gain
        self.derivative_gain = self.config.longitudinal_pid_derivative_gain
        self.integral_gain = self.config.longitudinal_pid_integral_gain
        self.max_window_length = self.config.longitudinal_pid_max_window_length
        self.speed_error_scaling = self.config.longitudinal_pid_speed_error_scaling
        self.braking_ratio = self.config.longitudinal_pid_braking_ratio
        self.minimum_target_speed = self.config.longitudinal_pid_minimum_target_speed

        self.speed_error_window = []
        self.saved_speed_error_window = []

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        """
        Get the throttle and brake values based on the target speed, current speed, 
        and hazard brake condition using a PID controller.

        Args:
            hazard_brake (bool): Flag indicating whether to apply hazard braking.
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            tuple: A tuple containing the throttle and brake values.
        """
        # If there's a hazard or the target speed is very small, apply braking
        if hazard_brake or target_speed < 1e-5:
            throttle, brake = 0., True
            return throttle, brake

        target_speed = max(self.minimum_target_speed, target_speed)  # Avoid very small target speeds

        current_speed, target_speed = 3.6 * current_speed, 3.6 * target_speed  # Convert to km/h

        # Test if the speed is "much" larger than the target speed
        if current_speed / target_speed > self.braking_ratio:
            self.speed_error_window = [0] * self.max_window_length

            throttle, brake = 0., True
            return throttle, brake

        speed_error = target_speed - current_speed
        speed_error = speed_error + speed_error * current_speed * self.speed_error_scaling

        self.speed_error_window.append(speed_error)
        self.speed_error_window = self.speed_error_window[-self.max_window_length:]

        derivative = 0 if len(
            self.speed_error_window) == 1 else self.speed_error_window[-1] - self.speed_error_window[-2]
        integral = np.mean(self.speed_error_window)

        throttle = self.proportional_gain * speed_error + self.derivative_gain * derivative + \
                                                                                    self.integral_gain * integral
        throttle, brake = np.clip(throttle, 0., 1.), False

        return throttle, brake

    def get_throttle_extrapolation(self, target_speed, current_speed):
        """
        Get the throttle value for the given target speed and current speed, assuming no hazard brake condition.

        Args:
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            float: The throttle value.
        """
        return self.get_throttle(False, target_speed, current_speed)

    def save(self):
        """
        Save the current state of the PID controller.
        """
        self.saved_speed_error_window = self.speed_error_window.copy()

    def load(self):
        """
        Load the previously saved state of the PID controller.
        """
        self.speed_error_window = self.saved_speed_error_window.copy()


class LongitudinalLinearRegressionController(LongitudinalController):
    """
    This class holds the linear regression module used for longitudinal control. It's used by default.
    """

    def __init__(self, config):
        super().__init__(config)

        self.minimum_target_speed = self.config.longitudinal_linear_regression_minimum_target_speed
        self.params = self.config.longitudinal_linear_regression_params
        self.maximum_acceleration = self.config.longitudinal_linear_regression_maximum_acceleration
        self.maximum_deceleration = self.config.longitudinal_linear_regression_maximum_deceleration

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        """
        Get the throttle and brake values based on the target speed, current speed, and hazard brake condition using a
        linear regression model.

        Args:
            hazard_brake (bool): Flag indicating whether to apply hazard braking.
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            tuple: A tuple containing the throttle and brake values.
        """
        if target_speed < 1e-5 or hazard_brake:
            return 0., True
        elif target_speed < self.minimum_target_speed:  # Avoid very small target speeds
            target_speed = self.minimum_target_speed

        current_speed = current_speed * 3.6
        target_speed = target_speed * 3.6
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1., False

        if current_speed / target_speed > params[-1] or hazard_brake:
            throttle, control_brake = 0., True
            return throttle, control_brake

        speed_error_cl = np.clip(speed_error, 0., np.inf) / 100.0
        current_speed /= 100.
        features = np.array([current_speed,\
                            current_speed**2,\
                            100*speed_error_cl,\
                            speed_error_cl**2,\
                            current_speed*speed_error_cl,\
                            current_speed**2*speed_error_cl])

        throttle, control_brake = np.clip(features @ params[:-1], 0., 1.), False

        return throttle, control_brake

    def get_throttle_extrapolation(self, target_speed, current_speed):
        """
        Get the throttle value for the given target speed and current speed, assuming no hazard brake condition. 
        This method is used for forecasting.

        Args:
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            float: The throttle value.
        """
        current_speed = current_speed * 3.6  # Convertion to km/h
        target_speed = target_speed * 3.6  # Convertion to km/h
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.
        # Maximum deceleration -4.82 m/tick
        elif speed_error < self.maximum_deceleration:
            return 0.

        throttle = 0.
        # 0.1 to ensure small distances are overcome fast
        if target_speed < 0.1 or current_speed / target_speed > params[-1]:
            return throttle

        speed_error_cl = np.clip(speed_error, 0., np.inf) / 100.0  # The scaling is a leftover from the optimization
        current_speed /= 100.  # The scaling is a leftover from the optimization
        features = np.array([current_speed,\
                            current_speed**2,\
                            100*speed_error_cl,\
                            speed_error_cl**2,\
                            current_speed*speed_error_cl,\
                            current_speed**2*speed_error_cl]).flatten()

        throttle = np.clip(features @ params[:-1], 0., 1.)

        return throttle
