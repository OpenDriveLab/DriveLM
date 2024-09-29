import numpy as np

class StanleyController():
    def __init__(self):
        self.last_steering = 0

    def run_step(self, ego_vehicle_location, next_100_locations,  ego_vehicle_velocity, ego_rotation, k, factor, smoothing, a, b, c):
        def normalize_angle(angle):
            if angle < -np.pi:
                angle += 2*np.pi
            elif angle > np.pi:
                angle -= 2*np.pi

            return angle

        # ego_rotation = ego_rotation / 180.0 * np.pi # in degrees

	    # https://github.com/Mostafa-wael/Self-Driving-Vehicle-Control-on-CARLA/blob/master/controller2d.py
	    
        # k = 0.1
        # factor = 2
        # smoothing = 0.2
        # a = 1
        # b = 0
        # c = 10
        # [0.2, 1.795215871175866, 0.1969741740760723, 1.0760049582816267, 5, 16]

        n_lookahead = np.clip(a*ego_vehicle_velocity+b, c, 100).astype('int')

        desired_heading_vec = next_100_locations[n_lookahead] - next_100_locations[0]
        
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path - ego_rotation
        heading_error = normalize_angle(factor * heading_error)

        # Cross track error
        yaw_cross_track = np.arctan2(ego_vehicle_location[1] - next_100_locations[0, 1], ego_vehicle_location[0] - next_100_locations[0, 0])
        yaw_diff_of_path_cross_track = yaw_path - yaw_cross_track
        yaw_diff_of_path_cross_track = normalize_angle(yaw_diff_of_path_cross_track)

        crosstrack_error = np.linalg.norm(ego_vehicle_location - next_100_locations[0])
        crosstrack_error = np.abs(crosstrack_error) if yaw_diff_of_path_cross_track > 0 else -np.abs(crosstrack_error)

        steering = heading_error + np.arctan(k * crosstrack_error)
        steering = (normalize_angle(steering)).clip(-1, 1)
	
        steering = smoothing * self.last_steering + (1.0 - smoothing) * steering
        self.last_steering = steering

        return steering