import carla
import subprocess
import time
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner
import socket
from nav_planner import RoutePlanner
from stanley_controller import StanleyController

class Simulator():

    recording_dict = {
        'steering': [],
        'driven_route': [],
        'distances': [],
        'speeds': [],
        'forward_vectors': [],
        'accelerations': []
    }

    def __init__(self):
        self.carla_simulator_path = "/home/jens/Desktop/CARLA_0.9.15/CarlaUE4.sh"

        self.stanley_controller = StanleyController()

        self.best_score = np.inf
        self.best_parameters = None
        self.previous_heading_error = 0.
        self.window_heading_error = []

        self.start_simulator()
        self.configure_simulator()
        self.spawn_car()

        self.track_points = np.array([
            [60.223614, 84.327255],   # target_speed = 20
            [35.776077, 50.112972],   # target_speed = 40
            [54.979015, 5.196576],    # target_speed = 50
            [62.000725, 1.834161],    # target_speed = 50
            [103.447800, -18.999334], # target_speed = 60
            [106.968506, -26.002760], # target_speed = 60
            [37.646976, -110.070084], # target_speed = 70
            [89.984039, -186.83435],  # target_speed = 100
            [189.164719, -30.362225], # target_speed = 100
            [193.130112, -20.527924], # target_speed = 100
            [193.114960, 50.349411],  # target_speed = 100
        ])

        self.interpolate_trace()
        self.configure_route_planner()

    def interpolate_trace(self):
        self.trace = []

        for i in range(self.track_points.shape[0]-1):
            loc, next_loc = self.track_points[i], self.track_points[i+1]
            from_wp = self.map.get_waypoint(carla.Location(x=loc[0], y=loc[1]))
            to_wp = self.map.get_waypoint(carla.Location(x=next_loc[0], y=next_loc[1]))

            self.trace += self.grp.trace_route(from_wp.transform.location, to_wp.transform.location)

        self.trace = [(wp.transform, cmd) for (wp, cmd) in self.trace]

    def configure_route_planner(self):
        self.route_planner = RoutePlanner(append_meters_to_provided_route=0)
        self.route_planner.set_route(self.trace, gps=False, carla_map=self.map)
        self.route_planner.add_further_info_to_route(self.world, self.map)
        self.route_planner.save()

        np.save('route.npy', self.route_planner.route_np)

        for loc_np in self.route_planner.route_np:
            loc = carla.Location(x=loc_np[0], y=loc_np[1], z=0.2+loc_np[2])
            self.world.debug.draw_point(loc, color=carla.Color(0, 1, 0), life_time=0)

    def start_simulator(self):
        for _ in range(5):
            subprocess.Popen('pkill Carla', shell=True)
            time.sleep(0.5)
        print('Start simulator and connect with the client!!!')
        # start the carla simulator
        self.port = self.get_available_port()
        print('PORT: {}'.format(self.port))
        subprocess.Popen('sh {} -RenderOffScreen -carla-streaming-port=0 -carla-rpc-port={} -nosound &'.format(self.carla_simulator_path, self.port), shell=True)
        time.sleep(10)

    def get_available_port(self):
        while True:
            port = np.random.randint(2000, 40_000)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port

    def configure_simulator(self):
        # connect to the carla simulator with a client
        self.client = carla.Client(host='localhost', port=self.port)

        self.client.set_timeout(30)
        self.world = self.client.load_world('Town05')
        self.map = self.world.get_map()
        self.grp = GlobalRoutePlanner(self.map, 1.0)

        self.tm_port= self.get_available_port()
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(np.random.randint(0, 1e8))

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.no_rendering_mode=False
        settings.substepping=True
        settings.max_culling_distance=0
        settings.deterministic_ragdolls=True
        self.world.apply_settings(settings)

        self.spectator = self.world.get_spectator()

    def spawn_car(self):
        blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')

        self.ego_vehicle_bp.set_attribute('role_name', 'hero')
        if self.ego_vehicle_bp.has_attribute('terramechanics'):
            self.ego_vehicle_bp.set_attribute('terramechanics', 'true')
        if self.ego_vehicle_bp.has_attribute('color'):
            color = np.random.choice(self.ego_vehicle_bp.get_attribute('color').recommended_values)
            self.ego_vehicle_bp.set_attribute('color', color)
        if self.ego_vehicle_bp.has_attribute('driver_id'):
            driver_id = np.random.choice(self.ego_vehicle_bp.get_attribute('driver_id').recommended_values)
            self.ego_vehicle_bp.set_attribute('driver_id', driver_id)
        if self.ego_vehicle_bp.has_attribute('is_invincible'):
            self.ego_vehicle_bp.set_attribute('is_invincible', 'true')
        # set the max speed
        if self.ego_vehicle_bp.has_attribute('speed'):
            self.player_max_speed = float(self.ego_vehicle_bp.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(self.ego_vehicle_bp.get_attribute('speed').recommended_values[2])

        self.ego_vehicle = None


    def normalize_angle(self, angle):
        if angle < -np.pi:
            angle += 2*np.pi
        elif angle > np.pi:
            angle -= 2*np.pi

        return angle

    def get_steer_stanley(self, route_np, current_speed, ego_vehicle_location, ego_vehicle_rotation, params):
        cross_track_scale, speed_scale, speed_offset, heading_scale, speed_threshold, default_lookahead = params

        n_lookahead = np.clip(speed_scale * current_speed + speed_offset, 0, 105).astype('int')

        # to make sure it drives not outside the lane during tight turning
        if current_speed < speed_threshold:
            n_lookahead = int(default_lookahead)

        desired_heading_vec = route_np[n_lookahead] - route_np[0]
        
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path - ego_vehicle_rotation
        heading_error = self.normalize_angle(heading_scale * heading_error)

        # Cross track error
        yaw_cross_track = np.arctan2(ego_vehicle_location[1] - route_np[0, 1], ego_vehicle_location[0] - route_np[0, 0])
        yaw_diff_of_path_cross_track = yaw_path - yaw_cross_track
        yaw_diff_of_path_cross_track = self.normalize_angle(yaw_diff_of_path_cross_track)

        crosstrack_error = np.linalg.norm(ego_vehicle_location - route_np[0])
        crosstrack_error = np.abs(crosstrack_error) if yaw_diff_of_path_cross_track > 0 else -np.abs(crosstrack_error)

        steering = heading_error + np.arctan(cross_track_scale * crosstrack_error)
        steering = self.normalize_angle(steering)

        steering = np.clip(steering, -1., 1.).item()

        return steering

    def get_steer_pure_persuit(self, route_np, current_speed, ego_vehicle_location, ego_rotation, params):
        car_length, speed_scale, speed_offset, steer_scale, default_lookahead, speed_threshold = params
        
        n_lookahead = int(min(np.clip(speed_scale * current_speed + speed_offset, 0, 105), route_np.shape[0] - 1))

        # to make sure it drives not outside the lane during tight turning
        if current_speed < speed_threshold:
            n_lookahead = int(default_lookahead)

        target_loc = route_np[n_lookahead]

        rear_axle = ego_vehicle_location[:2] - np.array([np.cos(ego_rotation) * car_length / 2., np.sin(ego_rotation) * car_length / 2])
        diff = target_loc[:2] - rear_axle[:2]
        
        d = np.linalg.norm(diff)
        alpha = np.arctan2(diff[1], diff[0]) - ego_rotation
        steering = np.arctan(2.0 * car_length * np.sin(alpha) / d)
        steering = np.clip(steer_scale * steering, -1., 1.).item()

        return steering

    def get_steer_pid(self, route_np, current_speed, ego_vehicle_location, ego_vehicle_rotation, params):
        k_p, speed_scale, speed_offset, k_d, default_lookahead, speed_threshold, k_i, max_length_window = params
        
        n_lookahead = int(min(np.clip(speed_scale * current_speed + speed_offset, 0, 105), route_np.shape[0] - 1))

        # to make sure it drives not outside the lane in tight turns
        if current_speed < speed_threshold:
            n_lookahead = int(default_lookahead)
            
        desired_heading_vec = route_np[n_lookahead] - ego_vehicle_location

        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = (yaw_path - ego_vehicle_rotation) % (2*np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2*np.pi
        
        # the scaling doesn't deserve any specific purpose but is a leftover from a previous less efficient implementation,
        # on which we optimized the parameters
        heading_error = heading_error * 180. / np.pi / 90.

        derivative = heading_error - self.previous_heading_error
        self.previous_heading_error = heading_error

        self.window_heading_error.append(heading_error)
        self.window_heading_error = self.window_heading_error[-int(max_length_window):]
        integral = np.mean(self.window_heading_error)

        steering = np.clip(k_p * heading_error + k_d * derivative + k_i * integral, -1., 1.).item()

        return steering

    def get_steer(self, route_np, current_speed, vehicle_pos_np, ego_rotation, params):
        # return self.get_steer_pure_persuit(route_np, current_speed, vehicle_pos_np, ego_rotation, params)
        # return self.get_steer_stanley(route_np, current_speed, vehicle_pos_np, ego_rotation, params)
        return self.get_steer_pid(route_np, current_speed, vehicle_pos_np, ego_rotation, params)

    def get_throttle(self, route_idx, current_speed):
        if route_idx<600:
            target_speed = 20
        elif route_idx<1300:
            target_speed = 40
        elif route_idx<2000:
            target_speed = 50
        elif route_idx<3250:
            target_speed = 50
        elif route_idx<4500:
            target_speed = 64
        else:
            target_speed = 100

        return 1. if current_speed < target_speed else 0.

    def drive_track(self, params):
        # reset initial conditions
        self.route_planner.load()
        self.window_heading_error = []
        self.previous_heading_error = 0

        for actor in self.world.get_actors().filter("*vehicle*"):
            if actor is not None and actor.is_alive:
                actor.destroy()

        start_transform = self.route_planner.route_wp[0].transform
        start_transform.location.z += 0.2
        self.ego_vehicle = self.world.try_spawn_actor(self.ego_vehicle_bp, start_transform)
        self.world.tick()

        # the actual driving
        no_movement_counter = 0
        l_distances = []
        while True:         
            vehicle_pos = self.ego_vehicle.get_location()
            ego_rotation = self.ego_vehicle.get_transform().rotation.yaw/180*np.pi
            vehicle_pos_np = np.array([vehicle_pos.x, vehicle_pos.y, vehicle_pos.z])
            route_np, _, _, _, _, _, _, _, _, _ = self.route_planner.run_step(vehicle_pos_np)
            route_idx = self.route_planner.route_idx
            distance = np.linalg.norm(route_np[0, :2] - vehicle_pos_np[:2])
            l_distances.append(distance)

            # there's a lane change
            if self.route_planner.route_idx<1270 and self.route_planner.route_idx>1190 or\
                self.route_planner.route_idx<1920 and self.route_planner.route_idx>1840 or\
                self.route_planner.route_idx<6750 and self.route_planner.route_idx>6570:
                distance = 0

            current_speed = self.ego_vehicle.get_velocity().length()*3.6
            steer = self.get_steer(route_np, current_speed, vehicle_pos_np, ego_rotation, params)
            throttle = self.get_throttle(route_idx, current_speed)
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.))

            forward_vector = self.ego_vehicle.get_transform().rotation.get_forward_vector()
            acceleration = self.ego_vehicle.get_acceleration()
            self.recording_dict['driven_route'].append(vehicle_pos_np)
            self.recording_dict['distances'].append(distance)
            self.recording_dict['steering'].append(steer)
            self.recording_dict['speeds'].append(current_speed)
            self.recording_dict['forward_vectors'].append([forward_vector.x, forward_vector.y, forward_vector.z])
            self.recording_dict['accelerations'].append([acceleration.x, acceleration.y, acceleration.z])

            # self.spectator.set_transform(carla.Transform(vehicle_pos + carla.Location(z=30), carla.Rotation(pitch=-90)))
            self.world.tick()

            if current_speed < 15:
                no_movement_counter += 1
            else:
                no_movement_counter = 0

            if no_movement_counter >= 30 or distance > 4.:
                l_distances = [4.]
                break

            if route_np.shape[0] < 120:
                break

        l_distances = np.array(l_distances)
        l_distances[l_distances<0.3] = 0.
        score = (l_distances**2).mean()

        if score < self.best_score:
            np.save('recording_dict.npy', self.recording_dict)
            self.best_score = score
            self.best_parameters = params

        print(f"Score: {score}\tBest score: {self.best_score}\tBest params: {self.best_parameters}")
            
        return score
