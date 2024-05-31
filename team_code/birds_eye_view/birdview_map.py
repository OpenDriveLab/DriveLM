"""
Functionality used to store CARLA maps as h5 files.
Code adapted from https://github.com/zhejz/carla-roach
"""

import carla
import pygame
import numpy as np
import h5py
from pathlib import Path
import os
import argparse
import time
import subprocess
from omegaconf import OmegaConf

from traffic_light import TrafficLightHandler
from server_utils import CarlaServerManager

COLOR_WHITE = (255, 255, 255)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class MapImage(object):
  """
  Functionality used to store CARLA maps as h5 files.
  """

  @staticmethod
  def draw_map_image(carla_map_local, pixels_per_meter_local, precision=0.05):

    waypoints = carla_map_local.generate_waypoints(2)
    margin = 100
    max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
    max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
    min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
    min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

    world_offset = np.array([min_x, min_y], dtype=np.float32)
    width_in_meters = max(max_x - min_x, max_y - min_y)
    width_in_pixels = round(pixels_per_meter_local * width_in_meters)

    print(width_in_pixels)
    print(width_in_meters)

    topology = [x[0] for x in carla_map_local.get_topology()]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    print(len(topology))
    road_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    shoulder_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    parking_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    sidewalk_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_yellow_broken_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_yellow_solid_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_white_broken_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_white_solid_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_all_surface = pygame.Surface((width_in_pixels, width_in_pixels))

    
    for i, waypoint in enumerate(topology):
      print(f'{i}/{len(topology)}')
      waypoints = [waypoint]
      # Generate waypoints of a road id. Stop when road id differs
      nxt = waypoint.next(precision)
      if len(nxt) > 0:
        nxt = nxt[0]
        while nxt.road_id == waypoint.road_id:
          waypoints.append(nxt)
          nxt = nxt.next(precision)
          if len(nxt) > 0:
            nxt = nxt[0]
          else:
            break
      # Draw Shoulders, Parkings and Sidewalks
      shoulder = [[], []]
      parking = [[], []]
      sidewalk = [[], []]

      print('Inner loop')

      for w in waypoints:
        # Classify lane types until there are no waypoints by going left
        l = w.get_left_lane()
        while l and l.lane_type != carla.LaneType.Driving:
          if l.lane_type == carla.LaneType.Shoulder:
            shoulder[0].append(l)
          if l.lane_type == carla.LaneType.Parking:
            parking[0].append(l)
          if l.lane_type == carla.LaneType.Sidewalk:
            sidewalk[0].append(l)
          l = l.get_left_lane()
        # Classify lane types until there are no waypoints by going right
        r = w.get_right_lane()
        while r and r.lane_type != carla.LaneType.Driving:
          if r.lane_type == carla.LaneType.Shoulder:
            shoulder[1].append(r)
          if r.lane_type == carla.LaneType.Parking:
            parking[1].append(r)
          if r.lane_type == carla.LaneType.Sidewalk:
            sidewalk[1].append(r)
          r = r.get_right_lane()

      print('draw lanes')

      MapImage.draw_lane(road_surface, waypoints, COLOR_WHITE, pixels_per_meter_local, world_offset)

      MapImage.draw_lane(sidewalk_surface, sidewalk[0], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(sidewalk_surface, sidewalk[1], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(shoulder_surface, shoulder[0], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(shoulder_surface, shoulder[1], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(parking_surface, parking[0], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(parking_surface, parking[1], COLOR_WHITE, pixels_per_meter_local, world_offset)
      print('if statement')

      if not waypoint.is_junction:
        print('In if statement')
        MapImage.draw_lane_marking_single_side(lane_marking_yellow_broken_surface, lane_marking_yellow_solid_surface,
                                               lane_marking_white_broken_surface, lane_marking_white_solid_surface,
                                               lane_marking_all_surface, waypoints, -1, pixels_per_meter_local,
                                               world_offset)
        print('Middle of if statement')
        MapImage.draw_lane_marking_single_side(lane_marking_yellow_broken_surface, lane_marking_yellow_solid_surface,
                                               lane_marking_white_broken_surface, lane_marking_white_solid_surface,
                                               lane_marking_all_surface, waypoints, 1, pixels_per_meter_local,
                                               world_offset)
      print('Loop iteration done')
    print('done with waypoints loop')
    # stoplines
    stopline_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    print('start stoplien loop')
    for stopline_vertices in TrafficLightHandler.list_stopline_vtx:
      for loc_left, loc_right in stopline_vertices:
        stopline_points = [
            MapImage.world_to_pixel(loc_left, pixels_per_meter_local, world_offset),
            MapImage.world_to_pixel(loc_right, pixels_per_meter_local, world_offset)
        ]
        MapImage.draw_line(stopline_surface, stopline_points, 2)
    print('done with stopline loop')
    # np.uint8 mask
    def _make_mask(x):
      return pygame.surfarray.array3d(x)[..., 0].astype(np.uint8)
    print('make a dict')
    # make a dict
    dict_masks_local = {
        'road': _make_mask(road_surface),
        'shoulder': _make_mask(shoulder_surface),
        'parking': _make_mask(parking_surface),
        'sidewalk': _make_mask(sidewalk_surface),
        'lane_marking_yellow_broken': _make_mask(lane_marking_yellow_broken_surface),
        'lane_marking_yellow_solid': _make_mask(lane_marking_yellow_solid_surface),
        'lane_marking_white_broken': _make_mask(lane_marking_white_broken_surface),
        'lane_marking_white_solid': _make_mask(lane_marking_white_solid_surface),
        'lane_marking_all': _make_mask(lane_marking_all_surface),
        'stopline': _make_mask(stopline_surface),
        'world_offset': world_offset,
        'pixels_per_meter': pixels_per_meter_local,
        'width_in_meters': width_in_meters,
        'width_in_pixels': width_in_pixels
    }
    return dict_masks_local

  @staticmethod
  def draw_lane_marking_single_side(lane_marking_yellow_broken_surface, lane_marking_yellow_solid_surface,
                                    lane_marking_white_broken_surface, lane_marking_white_solid_surface,
                                    lane_marking_all_surface, waypoints, sign, pixels_per_meter_local, world_offset):
    """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
        the waypoint based on the sign parameter"""
    lane_marking = None

    previous_marking_type = carla.LaneMarkingType.NONE
    previous_marking_color = carla.LaneMarkingColor.Other
    current_lane_marking = carla.LaneMarkingType.NONE

    markings_list = []
    temp_waypoints = []
    print(f'len(waypoints): {len(waypoints)}')
    for sample in waypoints:
      lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

      if lane_marking is None:
        continue

      if current_lane_marking != lane_marking.type:
        # Get the list of lane markings to draw
        markings = MapImage.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign,
                                              pixels_per_meter_local, world_offset)
        current_lane_marking = lane_marking.type

        # Append each lane marking in the list
        for marking in markings:
          markings_list.append(marking)

        temp_waypoints = temp_waypoints[-1:]

      else:
        temp_waypoints.append((sample))
        previous_marking_type = lane_marking.type
        previous_marking_color = lane_marking.color

    # Add last marking
    last_markings = MapImage.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign,
                                               pixels_per_meter_local, world_offset)
    print(f'last markings loop. len: {len(last_markings)}')
    for marking in last_markings:
      markings_list.append(marking)

    # Once the lane markings have been simplified to Solid or Broken lines, we draw them
    print(f'markings list loop. len: {len(markings_list)}')
    i = 0
    for markings in markings_list:  # markings is a 3-tuple of (carla.libcarla.LaneMarkingType, carla.libcarla.LaneMarkingColor, list of waypoint pairs)
      print(f'markings number: {i}')
      i=i+1
      print(f'len(markings[2]): {len(markings[2])}')
      if markings[1] == carla.LaneMarkingColor.White and markings[0] == carla.LaneMarkingType.Solid:
        print('first line')
        MapImage.draw_line(lane_marking_white_solid_surface, markings[2], 1)
      elif markings[1] == carla.LaneMarkingColor.Yellow and markings[0] == carla.LaneMarkingType.Solid:
        print('2nd line')
        MapImage.draw_line(lane_marking_yellow_solid_surface, markings[2], 1)
      elif markings[1] == carla.LaneMarkingColor.White and markings[0] == carla.LaneMarkingType.Broken:
        print('3rd line')
        MapImage.draw_line(lane_marking_white_broken_surface, markings[2], 1)
      elif markings[1] == carla.LaneMarkingColor.Yellow and markings[0] == carla.LaneMarkingType.Broken:
        print('4th line')
        MapImage.draw_line(lane_marking_yellow_broken_surface, markings[2], 1)

      print('final line')
      for i, chunk in enumerate(chunks(markings[2], 1000)):
        print(f'chunk {i}')
        MapImage.draw_line(lane_marking_all_surface, chunk, 1)


  @staticmethod
  def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign, pixels_per_meter_local, world_offset):
    """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
            as a combination of Broken and Solid lines"""
    margin = 0.25
    marking_1 = [
        MapImage.world_to_pixel(MapImage.lateral_shift(w.transform, sign * w.lane_width * 0.5), pixels_per_meter_local,
                                world_offset) for w in waypoints
    ]

    if lane_marking_type in (carla.LaneMarkingType.Broken, carla.LaneMarkingType.Solid):
      return [(lane_marking_type, lane_marking_color, marking_1)]
    else:
      marking_2 = [
          MapImage.world_to_pixel(MapImage.lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2)),
                                  pixels_per_meter_local, world_offset) for w in waypoints
      ]
      if lane_marking_type == carla.LaneMarkingType.SolidBroken:
        return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
      elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
        return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
      elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
        return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
      elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
        return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
    return [(carla.LaneMarkingType.NONE, lane_marking_color, marking_1)]

  @staticmethod
  def draw_line(surface, points, width):
    """Draws solid lines in a surface given a set of points, width and color"""
    if len(points) >= 2:
      pygame.draw.lines(surface, COLOR_WHITE, False, points, width)

  @staticmethod
  def draw_lane(surface, wp_list, color, pixels_per_meter_local, world_offset):
    """Renders a single lane in a surface and with a specified color"""
    lane_left_side = [MapImage.lateral_shift(w.transform, -w.lane_width * 0.5) for w in wp_list]
    lane_right_side = [MapImage.lateral_shift(w.transform, w.lane_width * 0.5) for w in wp_list]

    polygon = lane_left_side + list(reversed(lane_right_side))
    polygon = [MapImage.world_to_pixel(x, pixels_per_meter_local, world_offset) for x in polygon]

    if len(polygon) > 2:
      pygame.draw.polygon(surface, color, polygon, 5)
      pygame.draw.polygon(surface, color, polygon)

  @staticmethod
  def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()

  @staticmethod
  def world_to_pixel(location, pixels_per_meter_local, world_offset):
    """Converts the world coordinates to pixel coordinates"""
    x = pixels_per_meter_local * (location.x - world_offset[0])
    y = pixels_per_meter_local * (location.y - world_offset[1])
    return [round(y), round(x)]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', default='birds_eye_view/maps')
  parser.add_argument('--pixels_per_meter', type=float, default=4.0)
  parser.add_argument('--carla_sh_path', default='~/software/CARLA_Leaderboard_20/CarlaUE4.sh')
  # Those maps in the repo are generated using calra 0.9.9.4.
  # The maps have been slightly changed in carla 0.9.10/0.9.11

  args = parser.parse_args()

  # kill running carla server
  #with subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True) as kill_process:
  #  kill_process.wait()
  #time.sleep(1)

  env_config = OmegaConf.load('birds_eye_view/endless_all.yaml')
  server_manager = CarlaServerManager(args.carla_sh_path, port=2000, configs=env_config)
  #server_manager.start()

  save_dir = Path(args.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)

  os.environ['SDL_VIDEODRIVER'] = 'dummy'
  pygame.init()
  display = pygame.display.set_mode((320, 320), 0, 32)
  pygame.display.flip()

  pixels_per_meter = float(args.pixels_per_meter)
  for cfg in server_manager.env_configs:
    carla_map = cfg['env_configs']['carla_map']
    hf_file_path = save_dir / (carla_map + '.h5')

    # pass if map h5 already exists
    if hf_file_path.exists():
      map_hf = h5py.File(hf_file_path, 'r')
      hf_pixels_per_meter = float(map_hf.attrs['pixels_per_meter'])
      map_hf.close()
      if np.isclose(hf_pixels_per_meter, pixels_per_meter):
        print(f'{carla_map}.h5 with pixels_per_meter={pixels_per_meter:.2f} already exists.')
        continue

    client = carla.Client('localhost', port=6255)#cfg['port'])
    time.sleep(2)
    client.set_timeout(1000)
    #print(cfg['port'])
    print('Get maps')
    print(client.get_available_maps())
    print(f'Generating {carla_map}.h5 with pixels_per_meter={pixels_per_meter:.2f}.')
    world = client.load_world(carla_map)
    print('Loaded world. Drawing map image')
    dict_masks = MapImage.draw_map_image(world.get_map(), pixels_per_meter)

    with h5py.File(hf_file_path, 'w') as hf:
      hf.attrs['pixels_per_meter'] = pixels_per_meter
      hf.attrs['world_offset_in_meters'] = dict_masks['world_offset']
      hf.attrs['width_in_meters'] = dict_masks['width_in_meters']
      hf.attrs['width_in_pixels'] = dict_masks['width_in_pixels']
      hf.create_dataset('road', data=dict_masks['road'], compression='gzip', compression_opts=9)
      hf.create_dataset('shoulder', data=dict_masks['shoulder'], compression='gzip', compression_opts=9)
      hf.create_dataset('parking', data=dict_masks['parking'], compression='gzip', compression_opts=9)
      hf.create_dataset('sidewalk', data=dict_masks['sidewalk'], compression='gzip', compression_opts=9)
      hf.create_dataset('stopline', data=dict_masks['stopline'], compression='gzip', compression_opts=9)
      hf.create_dataset('lane_marking_all', data=dict_masks['lane_marking_all'], compression='gzip', compression_opts=9)
      hf.create_dataset('lane_marking_yellow_broken',
                        data=dict_masks['lane_marking_yellow_broken'],
                        compression='gzip',
                        compression_opts=9)
      hf.create_dataset('lane_marking_yellow_solid',
                        data=dict_masks['lane_marking_yellow_solid'],
                        compression='gzip',
                        compression_opts=9)
      hf.create_dataset('lane_marking_white_broken',
                        data=dict_masks['lane_marking_white_broken'],
                        compression='gzip',
                        compression_opts=9)
      hf.create_dataset('lane_marking_white_solid',
                        data=dict_masks['lane_marking_white_solid'],
                        compression='gzip',
                        compression_opts=9)

    break
  print('Stopping server')
  #server_manager.stop()
