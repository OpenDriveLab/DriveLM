"""
Utilities to render bird's eye view semantic segmentation maps.
Code adapted from https://github.com/zhejz/carla-roach
"""

import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py
import os

from birds_eye_view.obs_manager import ObsManagerBase
from birds_eye_view.traffic_light import TrafficLightHandler

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREY = (128, 128, 128)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
  r, g, b = color
  r = int(r + (255 - r) * factor)
  g = int(g + (255 - g) * factor)
  b = int(b + (255 - b) * factor)
  r = min(r, 255)
  g = min(g, 255)
  b = min(b, 255)
  return (r, g, b)


class ObsManager(ObsManagerBase):
  """
  Generates bev semantic segmentation maps.
  """

  def __init__(self, obs_configs, config):
    self._width = int(obs_configs['width_in_pixels'])
    self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
    self._pixels_per_meter = obs_configs['pixels_per_meter']
    self._history_idx = obs_configs['history_idx']
    self._scale_bbox = obs_configs.get('scale_bbox', True)
    self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)
    maxlen_queue = max(max(obs_configs['history_idx']) + 1, -min(obs_configs['history_idx']))
    self._history_queue = deque(maxlen=maxlen_queue)
    self.visualize = int(os.environ.get('DEBUG_CHALLENGE', 0)) or int(os.environ.get('TMP_VISU', 0))
    self.config = config

    self._image_channels = 3
    self._masks_channels = 3 + 3 * len(self._history_idx)
    self.vehicle = None
    self._world = None

    if 'map_folder' in obs_configs:
      map_folder = obs_configs['map_folder']
    else:
      map_folder = 'maps'

    self._map_dir = Path(__file__).resolve().parent / map_folder

    super().__init__()

  def _define_obs_space(self):
    self.obs_space = spaces.Dict({
        'rendered': spaces.Box(low=0, high=255, shape=(self._width, self._width, self._image_channels), dtype=np.uint8),
        'masks': spaces.Box(low=0, high=255, shape=(self._masks_channels, self._width, self._width), dtype=np.uint8)
    })

  def attach_ego_vehicle(self, vehicle, criteria_stop):
    self.vehicle = vehicle
    self._world = self.vehicle.get_world()
    self.criteria_stop = criteria_stop

    maps_h5_path = self._map_dir / (self._world.get_map().name.split('/')[
			                                -1] + '.h5')  # splitting because for Town13 the name is 'Carla/Maps/Town13/Town13' instead of 'Town13'
    with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
      self._road = np.array(hf['road'], dtype=np.uint8)
      self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
      self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)
      # self._shoulder = np.array(hf['shoulder'], dtype=np.uint8)
      # self._parking = np.array(hf['parking'], dtype=np.uint8)
      self._sidewalk = np.array(hf['sidewalk'], dtype=np.uint8)
      # self._lane_marking_yellow_broken = np.array(hf['lane_marking_yellow_broken'], dtype=np.uint8)
      # self._lane_marking_yellow_solid = np.array(hf['lane_marking_yellow_solid'], dtype=np.uint8)
      # self._lane_marking_white_solid = np.array(hf['lane_marking_white_solid'], dtype=np.uint8)

      self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
      # in case they aren't close, print them to know what values they should be
      if not np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter'])):
        print(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))
      assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

    self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
    # dilate road mask, lbc draw road polygon with 10px boarder
    # kernel = np.ones((11, 11), np.uint8)
    # self._road = cv.dilate(self._road, kernel, iterations=1)

    TrafficLightHandler.reset(self._world)

  @staticmethod
  def _get_stops(criteria_stop):
    stop_sign = criteria_stop.target_stop_sign
    stops = []
    if (stop_sign is not None) and (not criteria_stop.stop_completed):
      bb_loc = carla.Location(stop_sign.trigger_volume.location)
      bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
      # Workaround since the extents of trigger_volumes of stop signs are often wrong
      # bb_ext.x = max(bb_ext.x, bb_ext.y)
      # bb_ext.y = max(bb_ext.x, bb_ext.y)
      bb_ext.x = 1.5
      bb_ext.y = 1.5
      trans = stop_sign.get_transform()
      stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
    return stops

  def get_road(self):
    """
    :return: Return an image of the road in LiDAR coordinates with alpha channel for visualization
    """
    ev_transform = self.vehicle.get_transform()
    ev_loc = ev_transform.location
    ev_rot = ev_transform.rotation
    m_warp = self._get_warp_transform(ev_loc, ev_rot)
    # road_mask, lane_mask
    road_mask = cv.warpAffine(self._road, m_warp, (self._width, self._width)).astype(np.bool)
    lane_mask_all = cv.warpAffine(self._lane_marking_all, m_warp, (self._width, self._width)).astype(np.bool)
    lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, m_warp,
                                     (self._width, self._width)).astype(np.bool)
    image = np.zeros([self._width, self._width, 4], dtype=np.float32)
    alpha = 0.33
    image[road_mask] = (40, 40, 40, 0.1)
    image[lane_mask_all] = (255, 255, 0, alpha)
    image[lane_mask_broken] = (255, 255, 0, alpha)
    image = np.rot90(image, k=-1)  # Align with LiDAR coordinate system

    return image

  def get_observation(self, close_traffic_lights=None):
    ev_transform = self.vehicle.get_transform()
    ev_loc = ev_transform.location
    ev_rot = ev_transform.rotation
    ev_bbox = self.vehicle.bounding_box

    def is_within_distance(w):
      c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
          and abs(ev_loc.y - w.location.y) < self._distance_threshold \
          and abs(ev_loc.z - w.location.z) < 8.0
      # Cheap trick to remove the ego bounding box
      c_ev = abs(ev_loc.x - w.location.x) < self.config.ego_extent_y and abs(ev_loc.y -
                                                                             w.location.y) < self.config.ego_extent_y
      return c_distance and (not c_ev)

    actors = self._world.get_actors()
    vehicles = actors.filter('*vehicle*')
    walkers = actors.filter('*walker*')

    # This style of generating bounding boxes is more ugly than just calling world.get_level_bbs in carla but it has
    # the advantage of not causing segfaults in the carla library :)
    vehicle_bbox_list = []
    for vehicle in vehicles:
      if vehicle.id == self.vehicle.id:
        continue
      traffic_transform = vehicle.get_transform()

      # Convert the bounding box to global coordinates
      bounding_box = carla.BoundingBox(traffic_transform.location + vehicle.bounding_box.location,
                                       vehicle.bounding_box.extent)
      # Rotations of the bb are 0.
      bounding_box.rotation = carla.Rotation(pitch=vehicle.bounding_box.rotation.pitch +
                                             traffic_transform.rotation.pitch,
                                             yaw=vehicle.bounding_box.rotation.yaw + traffic_transform.rotation.yaw,
                                             roll=vehicle.bounding_box.rotation.roll + traffic_transform.rotation.roll)

      vehicle_bbox_list.append(bounding_box)

    walker_bbox_list = []
    for walker in walkers:
      walker_transform = walker.get_transform()

      walker_location = walker_transform.location
      transform = carla.Transform(walker_location)
      bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
      bounding_box.rotation = carla.Rotation(pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                                             yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw,
                                             roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)

      walker_bbox_list.append(bounding_box)

    if self._scale_bbox:
      vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
      walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
    else:
      vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
      walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

    tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0, self._distance_threshold, close_traffic_lights)
    tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1, self._distance_threshold, close_traffic_lights)
    tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2, self._distance_threshold, close_traffic_lights)
    stops = self._get_stops(self.criteria_stop)

    self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

    m_warp = self._get_warp_transform(ev_loc, ev_rot)

    # objects with history
    vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
        = self._get_history_masks(m_warp)

    # road_mask, lane_mask
    road_mask = cv.warpAffine(self._road, m_warp, (self._width, self._width)).astype(np.bool)
    sidewalk_mask = cv.warpAffine(self._sidewalk, m_warp, (self._width, self._width)).astype(np.bool)
    lane_mask_all = cv.warpAffine(self._lane_marking_all, m_warp, (self._width, self._width)).astype(np.bool)
    lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, m_warp,
                                     (self._width, self._width)).astype(np.bool)

    # render
    if self.visualize:
      # ev_mask
      ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], m_warp)

      image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
      image[road_mask] = COLOR_ALUMINIUM_5
      image[sidewalk_mask] = COLOR_GREY
      image[lane_mask_all] = COLOR_MAGENTA
      image[lane_mask_broken] = COLOR_MAGENTA_2

      h_len = len(self._history_idx) - 1
      for i, mask in enumerate(stop_masks):
        image[mask] = tint(COLOR_YELLOW_2, (h_len - i) * 0.2)
      for i, mask in enumerate(tl_green_masks):
        image[mask] = tint(COLOR_GREEN, (h_len - i) * 0.2)
      for i, mask in enumerate(tl_yellow_masks):
        image[mask] = tint(COLOR_YELLOW, (h_len - i) * 0.2)
      for i, mask in enumerate(tl_red_masks):
        image[mask] = tint(COLOR_RED, (h_len - i) * 0.2)

      for i, mask in enumerate(vehicle_masks):
        image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)
      for i, mask in enumerate(walker_masks):
        image[mask] = tint(COLOR_CYAN, (h_len - i) * 0.2)

      image[ev_mask] = COLOR_WHITE

    # masks
    # 0 = Unlabeled
    c_all = road_mask * 1
    c_all[sidewalk_mask] = 2
    c_all[lane_mask_all] = 3
    c_all[lane_mask_broken] = 4
    c_all[stop_masks[-1]] = 5
    c_all[tl_green_masks[-1]] = 6
    c_all[tl_yellow_masks[-1]] = 7
    c_all[tl_red_masks[-1]] = 8
    c_all[vehicle_masks[-1]] = 9
    c_all[walker_masks[-1]] = 10

    # Align with LiDAR voxelgrid
    c_all = np.rot90(c_all, k=-1)

    obs_dict = {'bev_semantic_classes': c_all}
    if self.visualize:
      # For visualization we don't rotate as
      obs_dict['rendered'] = image

    return obs_dict

  def _get_history_masks(self, m_warp):
    qsize = len(self._history_queue)
    vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
    for idx in self._history_idx:
      idx = max(idx, -1 * qsize)

      vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]

      vehicle_masks.append(self._get_mask_from_actor_list(vehicles, m_warp))
      walker_masks.append(self._get_mask_from_actor_list(walkers, m_warp))
      tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, m_warp))
      tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, m_warp))
      tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, m_warp))
      stop_masks.append(self._get_mask_from_actor_list(stops, m_warp))

    return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

  def _get_mask_from_stopline_vtx(self, stopline_vtx, m_warp):
    mask = np.zeros([self._width, self._width], dtype=np.uint8)
    for sp_locs in stopline_vtx:
      stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
      stopline_warped = cv.transform(stopline_in_pixel, m_warp)
      # Rounding to pixel coordinates is required by newer opencv versions
      pt1 = (stopline_warped[0, 0] + 0.5).astype(np.int)
      pt2 = (stopline_warped[1, 0] + 0.5).astype(np.int)
      cv.line(mask, tuple(pt1), tuple(pt2), color=1, thickness=6)
    return mask.astype(np.bool)

  def _get_mask_from_actor_list(self, actor_list, m_warp):
    mask = np.zeros([self._width, self._width], dtype=np.uint8)
    for actor_transform, bb_loc, bb_ext in actor_list:

      corners = [
          carla.Location(x=-bb_ext.x, y=-bb_ext.y),
          carla.Location(x=bb_ext.x, y=-bb_ext.y),
          carla.Location(x=bb_ext.x, y=0),
          carla.Location(x=bb_ext.x, y=bb_ext.y),
          carla.Location(x=-bb_ext.x, y=bb_ext.y)
      ]
      corners = [bb_loc + corner for corner in corners]

      corners = [actor_transform.transform(corner) for corner in corners]
      corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
      corners_warped = cv.transform(corners_in_pixel, m_warp)

      cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
    return mask.astype(np.bool)

  @staticmethod
  def _get_surrounding_actors(bbox_list, criterium, scale=None):
    actors = []
    for bbox in bbox_list:
      is_within_distance = criterium(bbox)
      if is_within_distance:
        bb_loc = carla.Location()
        bb_ext = carla.Vector3D(bbox.extent)
        if scale is not None:
          bb_ext = bb_ext * scale
          bb_ext.x = max(bb_ext.x, 0.8)
          bb_ext.y = max(bb_ext.y, 0.8)

        actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
    return actors

  def _get_warp_transform(self, ev_loc, ev_rot):
    ev_loc_in_px = self._world_to_pixel(ev_loc)
    yaw = np.deg2rad(ev_rot.yaw)

    forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
    right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

    bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5 * self._width) * right_vec
    top_left = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec - (0.5 * self._width) * right_vec
    top_right = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec + (0.5 * self._width) * right_vec

    src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
    dst_pts = np.array([[0, self._width - 1], [0, 0], [self._width - 1, 0]], dtype=np.float32)
    return cv.getAffineTransform(src_pts, dst_pts)

  def _world_to_pixel(self, location, projective=False):
    """Converts the world coordinates to pixel coordinates"""
    x = self._pixels_per_meter * (location.x - self._world_offset[0])
    y = self._pixels_per_meter * (location.y - self._world_offset[1])

    if projective:
      p = np.array([x, y, 1], dtype=np.float32)
    else:
      p = np.array([x, y], dtype=np.float32)
    return p

  def _world_to_pixel_width(self, width):
    """Converts the world units to pixel units"""
    return self._pixels_per_meter * width

  def clean(self):
    self.vehicle = None
    self._world = None
    self._history_queue.clear()
