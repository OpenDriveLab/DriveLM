import carla
import numpy as np
from tqdm import tqdm

"""
This script is used to precompute the maps for the speed limits in CARLA.
You have to start the CARLA simulator before executing this script.
"""

def compute_speed_limit_map(carla_port, map_name):
    client = carla.Client('localhost', carla_port)
    client.set_timeout(120)
    carla_world = client.load_world(map_name)
    carla_map = carla_world.get_map()

    # Get all point where the car can drive with 1m distance between each other
    map_waypoints = carla_map.generate_waypoints(1.0)
    map_waypoints_locs = [x.transform.location for x in map_waypoints]
    map_waypoints_locs = np.array([[x.x, x.y, x.z] for x in map_waypoints_locs])

    # Get all locations of the speed signs
    speed_signs = carla_map.get_all_landmarks_of_type(carla.LandmarkType.MaximumSpeed)
    speed_signs_locs = [x.transform.location for x in speed_signs]
    speed_signs_locs = np.array([[x.x, x.y] for x in speed_signs_locs])
            
    # compute speed limits with functions provided by Carla
    speed_limits = -1 * np.ones(map_waypoints_locs.shape[0])
    def compute_speed_limits(wps, speed_limit_indices):
        for wp, speed_limit_idx in zip(wps, speed_limit_indices):
            min_distance, max_distance = 0, 300

            found_max_distance = False
            for dist in range(300, 4000, 300):
                if len(wp.get_landmarks_of_type(dist, carla.LandmarkType.MaximumSpeed)) > 0:
                    min_distance, max_distance = max(0, dist-305), dist
                    found_max_distance = True
                    break

            if found_max_distance:
                while True:
                    mid_distance = (min_distance + max_distance) / 2.
                    landmarks = wp.get_landmarks_of_type(mid_distance, carla.LandmarkType.MaximumSpeed)

                    if len(landmarks) == 1:
                        speed_limits[speed_limit_idx] = landmarks[0].value
                        break
                    elif len(landmarks) == 0:
                        min_distance = mid_distance
                    else:
                        max_distance = mid_distance

                    if max_distance - min_distance < 5:
                        if landmarks:                
                            speed_limits[speed_limit_idx] = landmarks[0].value

                        break

    if len(speed_signs_locs) == 0: # use 50 km/h as default speed if the map contains no speed sign
        speed_limits[:] = 50.
    else:
        compute_speed_limits(map_waypoints, np.arange(speed_limits.shape[0]))

    # compute the speed_limits for points, for which it couldn't be computed earlier by moving them a bit
    indices = np.arange(speed_limits.shape[0])
    wps = np.array(map_waypoints)

    for _ in tqdm(range(10)):
        mask = speed_limits[indices] == -1
        wps = wps[mask]
        if len(wps)==0:
            break

        tmp = []
        for wp in wps:
            nex = wp.next(1)
            if nex:
                tmp.append(np.random.choice(nex))
            else:
                tmp.append(np.random.choice(wp.previous(1)))

        wps = np.array(tmp)
        indices = indices[mask]

        compute_speed_limits(wps, indices)


    # set some variables up to interpolate the rest of the locations, for which we couldn't compute the speed limits earlier
    rounded_map_waypoints_locs = map_waypoints_locs[:, :2].round().astype('int')
    map_offset = rounded_map_waypoints_locs.min()
    rounded_map_waypoints_locs -= map_offset[None]
    map_size = rounded_map_waypoints_locs.max(axis=0) + 1

    map_speed_limits = -np.ones(map_size)
    map_speed_limits[rounded_map_waypoints_locs[:, 0], rounded_map_waypoints_locs[:, 1]] = speed_limits
    mask = speed_limits==-1

    unassigned_waypoints_locs = rounded_map_waypoints_locs[mask] # [13361, 2]

    # interpolate these left locations
    for _ in tqdm(range(unassigned_waypoints_locs.shape[0])):
        dist = 1

        while True:
            arr = np.arange(-dist, dist+1)
            x, y = np.meshgrid(arr, arr)
            indices = np.concatenate([x.flatten()[None], y.flatten()[None]], axis=0).T # [9, 2]
            indices = unassigned_waypoints_locs[:, None, :] + indices[None, :, :] # [13361, 9, 2]
            indices[:, :, 0] = indices[:, :, 0].clip(0, map_size[0] - 1)
            indices[:, :, 1] = indices[:, :, 1].clip(0, map_size[1] - 1)
            close_speed_limits = map_speed_limits[indices[:,:,0], indices[:,:,1]] # [13361, 9]
            close_speed_limits = close_speed_limits.max(axis=1) # [13361]
            max_idx = close_speed_limits.argmax()
            max_speed_limit = close_speed_limits[max_idx]

            found_one = max_speed_limit != -1
            if found_one:
                break
            else:
                dist += 1

        map_speed_limits[unassigned_waypoints_locs[max_idx, 0], unassigned_waypoints_locs[max_idx, 1]] = max_speed_limit
        unassigned_waypoints_locs = np.delete(unassigned_waypoints_locs, max_idx, axis=0)
        
    # filter these locations and only allow speed limits 50, 100 and 120
    filter_dist = 5

    for i in tqdm(range(rounded_map_waypoints_locs.shape[0])):
        rounded_loc = rounded_map_waypoints_locs[i]
        from_loc, to_loc = rounded_loc - filter_dist, rounded_loc + filter_dist + 1

        from_loc[0] = np.clip(from_loc[0], 0, map_size[0]-1)
        from_loc[1] = np.clip(from_loc[1], 0, map_size[1]-1)
        to_loc[0] = np.clip(to_loc[0], 0, map_size[0])
        to_loc[1] = np.clip(to_loc[1], 0, map_size[1])

        filtered_speed_limit = map_speed_limits[from_loc[0]:to_loc[0], from_loc[1]:to_loc[1]].max()

        if filtered_speed_limit <= 50:
            filtered_speed_limit = 50
        elif filtered_speed_limit <= 80:
            filtered_speed_limit = 80
        elif filtered_speed_limit <= 100:
            filtered_speed_limit = 100
        elif filtered_speed_limit <=120:
            filtered_speed_limit = 120
        else:
            raise NotImplementedError()

        speed_limits[i] = filtered_speed_limit

    # save the result
    d = {'speed_limits': speed_limits, 'locations': map_waypoints_locs} 
    np.save(f'{map_name}_speed_limits.npy', d)

for map_name in sorted(['Town07', 'Town06', 'Town04', 'Town02', 'Town10HD', 'Town03', 'Town05', 'Town01', 'Town15', 'Town11', 'Town13', 'Town12']):
    print(f'Compute {map_name}')
    compute_speed_limit_map(2000, map_name)