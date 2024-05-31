"""
Some utility functions e.g. for normalizing angles
Functions for detecting red lights are adapted from scenario runners
atomic_criteria.py
"""
import math
import carla
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cv2
from collections import deque
from shapely.geometry import Polygon
import shapely
import itertools
from copy import deepcopy


def normalize_angle(x):
  x = x % (2 * np.pi)  # force in range [0, 2 pi)
  if x > np.pi:  # move to [-pi, pi)
    x -= 2 * np.pi
  return x


def normalize_angle_degree(x):
  x = x % 360.0
  if x > 180.0:
    x -= 360.0
  return x


def rotate_point(point, angle):
  """
  rotate a given point by a given angle
  """
  x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
  y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
  return carla.Vector3D(x_, y_, point.z)


def get_traffic_light_waypoints(traffic_light, carla_map):
  """
  get area of a given traffic light
  """
  base_transform = traffic_light.get_transform()
  base_loc = traffic_light.get_location()
  base_rot = base_transform.rotation.yaw
  area_loc = base_transform.transform(traffic_light.trigger_volume.location)

  # Discretize the trigger box into points
  area_ext = traffic_light.trigger_volume.extent
  x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

  area = []
  for x in x_values:
    point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
    point_location = area_loc + carla.Location(x=point.x, y=point.y)
    area.append(point_location)

  # Get the waypoints of these points, removing duplicates
  ini_wps = []
  for pt in area:
    wpx = carla_map.get_waypoint(pt)
    # As x_values are arranged in order, only the last one has to be checked
    if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
      ini_wps.append(wpx)

  # Advance them until the intersection
  wps = []
  eu_wps = []
  for wpx in ini_wps:
    distance_to_light = base_loc.distance(wpx.transform.location)
    eu_wps.append(wpx)
    next_distance_to_light = distance_to_light + 1.0
    while not wpx.is_intersection:
      next_wp = wpx.next(0.5)[0]
      next_distance_to_light = base_loc.distance(next_wp.transform.location)
      if next_wp and not next_wp.is_intersection \
          and next_distance_to_light <= distance_to_light:
        eu_wps.append(next_wp)
        distance_to_light = next_distance_to_light
        wpx = next_wp
      else:
        break

    if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
      wps.append(eu_wps[-4])
    else:
      wps.append(wpx)

  return area_loc, wps


def lidar_to_ego_coordinate(config, lidar):
  """
  Converts the LiDAR points given by the simulator into the ego agents
  coordinate system
  :param config: GlobalConfig, used to read out lidar orientation and location
  :param lidar: the LiDAR point cloud as provided in the input of run_step
  :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
  coordinate system.
  """
  yaw = np.deg2rad(config.lidar_rot[2])
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

  translation = np.array(config.lidar_pos)

  # The double transpose is a trick to compute all the points together.
  ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

  return ego_lidar


def algin_lidar(lidar, translation, yaw):
  """
  Translates and rotates a LiDAR into a new coordinate system.
  Rotation is inverse to translation and yaw
  :param lidar: numpy LiDAR point cloud (N,3)
  :param translation: translations in meters
  :param yaw: yaw angle in radians
  :return: numpy LiDAR point cloud in the new coordinate system.
  """

  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

  aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T

  return aligned_lidar


def inverse_conversion_2d(point, translation, yaw):
  """
  Performs a forward coordinate conversion on a 2D point
  :param point: Point to be converted
  :param translation: 2D translation vector of the new coordinate system
  :param yaw: yaw in radian of the new coordinate system
  :return: Converted point
  """
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

  converted_point = rotation_matrix.T @ (point - translation)
  return converted_point


def preprocess_compass(compass):
  """
  Checks the compass for Nans and rotates it into the default CARLA coordinate
  system with range [-pi,pi].
  :param compass: compass value provided by the IMU, in radian
  :return: yaw of the car in radian in the CARLA coordinate system.
  """
  if math.isnan(compass):  # simulation bug
    compass = 0.0
  # The minus 90.0 degree is because the compass sensor uses a different
  # coordinate system then CARLA. Check the coordinate_sytems.txt file
  compass = normalize_angle(compass - np.deg2rad(90.0))

  return compass


def get_relative_transform(ego_matrix, vehicle_matrix):
  """
  Returns the position of the vehicle matrix in the ego coordinate system.
  :param ego_matrix: ndarray 4x4 Matrix of the ego vehicle in global
  coordinates
  :param vehicle_matrix: ndarray 4x4 Matrix of another actor in global
  coordinates
  :return: ndarray position of the other vehicle in the ego coordinate system
  """
  relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
  rot = ego_matrix[:3, :3].T
  relative_pos = rot @ relative_pos

  return relative_pos


def extract_yaw_from_matrix(matrix):
  """Extracts the yaw from a CARLA world matrix"""
  yaw = math.atan2(matrix[1, 0], matrix[0, 0])
  yaw = normalize_angle(yaw)
  return yaw


# Taken from https://stackoverflow.com/a/47381058/9173068
def trapez(y, y0, w):
  return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
  # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
  # If either of these cases are violated, do some switches.
  if abs(c1 - c0) < abs(r1 - r0):
    # Switch x and y, and switch again when returning.
    xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)  # pylint: disable=locally-disabled, arguments-out-of-order
    return (yy, xx, val)

  # At this point we know that the distance in columns (x) is greater
  # than that in rows (y). Possibly one more switch if c0 > c1.
  if c0 > c1:
    return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)  # pylint: disable=locally-disabled, arguments-out-of-order

  # The following is now always < 1 in abs
  if (c1 - c0) != 0.0:
    slope = (r1 - r0) / (c1 - c0)
  else:
    slope = 0.0

  # Adjust weight by the slope
  w *= np.sqrt(1 + np.abs(slope)) / 2

  # We write y as a function of x, because the slope is always <= 1
  # (in absolute value)
  x = np.arange(c0, c1 + 1, dtype=float)
  if (c1 - c0) != 0.0:
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)
  else:
    y = np.zeros_like(x)

  # Now instead of 2 values for y, we have 2*np.ceil(w/2).
  # All values are 1 except the upmost and bottommost.
  thickness = np.ceil(w / 2)
  yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
  xx = np.repeat(x, yy.shape[1])
  vals = trapez(yy, y.reshape(-1, 1), w).flatten()

  yy = yy.flatten()

  # Exclude useless parts and those outside of the interval
  # to avoid parts outside of the picture
  mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

  return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def draw_line(img, start_row, start_column, end_row, end_column, color=(255, 255, 255), thickness=1, rmax=256):

  if start_row == end_row and start_column == end_column:
    rr, cc, val = start_row, start_column, 1.0
  else:
    rr, cc, val = weighted_line(r0=start_row, c0=start_column, r1=end_row, c1=end_column, w=thickness, rmax=rmax)

  img[rr, cc, 0] = val * color[0] + (1.0 - val) * img[rr, cc, 0]
  img[rr, cc, 1] = val * color[1] + (1.0 - val) * img[rr, cc, 1]
  img[rr, cc, 2] = val * color[2] + (1.0 - val) * img[rr, cc, 2]
  return img


def draw_box(img, box, color=(255, 255, 255), pixel_per_meter=4, thickness=1):
  translation = np.array([[box[0], box[1]]])
  width = box[2]
  height = box[3]
  yaw = box[4]
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
  speed = box[5] * pixel_per_meter
  speed_coords = np.array([[0.0, 0.0], [0.0, speed]])
  corners = np.array([[-width, -height], [width, -height], [width, height], [-width, height]])
  corner_global = (rotation_matrix @ corners.T).T + translation
  speed_coords_global = (rotation_matrix @ speed_coords.T).T + translation
  corner_global = corner_global.astype(np.int64)
  speed_coords_global = speed_coords_global.astype(np.int64)

  # Only the center is guaranteed to be within the image. Need to clip the corner points.
  max_row = img.shape[0]
  max_column = img.shape[1]
  corner_global[:, 0] = np.clip(corner_global[:, 0], a_min=0, a_max=max_row - 1)
  corner_global[:, 1] = np.clip(corner_global[:, 1], a_min=0, a_max=max_column - 1)
  speed_coords_global[:, 0] = np.clip(speed_coords_global[:, 0], a_min=0, a_max=max_row - 1)
  speed_coords_global[:, 1] = np.clip(speed_coords_global[:, 1], a_min=0, a_max=max_column - 1)

  img = draw_line(img,
                  start_row=corner_global[0, 0],
                  start_column=corner_global[0, 1],
                  end_row=corner_global[1, 0],
                  end_column=corner_global[1, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=corner_global[1, 0],
                  start_column=corner_global[1, 1],
                  end_row=corner_global[2, 0],
                  end_column=corner_global[2, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=corner_global[2, 0],
                  start_column=corner_global[2, 1],
                  end_row=corner_global[3, 0],
                  end_column=corner_global[3, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=corner_global[3, 0],
                  start_column=corner_global[3, 1],
                  end_row=corner_global[0, 0],
                  end_column=corner_global[0, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=speed_coords_global[0, 0],
                  start_column=speed_coords_global[0, 1],
                  end_row=speed_coords_global[1, 0],
                  end_column=speed_coords_global[1, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)

  return img


class PIDController(object):
  """
    PID controller that converts waypoints to steer, brake and throttle commands
    """

  def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
    self.k_p = k_p
    self.k_i = k_i
    self.k_d = k_d

    self.window = deque([0 for _ in range(n)], maxlen=n)

  def step(self, error):
    self.window.append(error)

    if len(self.window) >= 2:
      integral = np.mean(self.window)
      derivative = self.window[-1] - self.window[-2]
    else:
      integral = 0.0
      derivative = 0.0

    return self.k_p * error + self.k_i * integral + self.k_d * derivative


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0, reduction='mean'):
  """ Adapted from mmdetection
  Args:
      pred (torch.Tensor): The prediction.
      gaussian_target (torch.Tensor): The learning target of the prediction
          in gaussian distribution.
      alpha (float, optional): A balanced form for Focal Loss.
          Defaults to 2.0.
      gamma (float, optional): The gamma for calculating the modulating
          factor. Defaults to 4.0.
  """
  eps = 1e-12
  pos_weights = gaussian_target.eq(1)
  neg_weights = (1 - gaussian_target).pow(gamma)
  pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
  neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
  loss = pos_loss + neg_loss

  if reduction == 'mean':
    loss = loss.mean()
  elif reduction == 'sum':
    loss = loss.sum()
  # All other reductions will be no reduction.
  return loss


def bb_vehicle_to_image_system(box, pixels_per_meter, min_x, min_y):
  """
  Changed a bounding box from the vehicle x front, y right coordinate system
  to the x back, y right coordinate system of an image, where the center of
  the car is in the center of the image.
  :return:
  """
  # Multiply position and extent by pixels_per_meter to convert the unit from meters to pixels
  box[:4] = box[:4] * pixels_per_meter
  # Pixel coordinates is y front, x right. CARLA is x front, y right.
  # So we need to swap the axes to convert the coordinates.
  box[0], box[1] = box[1], box[0]
  box[2], box[3] = box[3], box[2]
  # Compute pixel location that represents 0/0 in the image
  translation = np.array([-(min_x * pixels_per_meter), -(min_y * pixels_per_meter)])
  # Shift the coordinates so that the ego_vehicle is at the center of the image
  box[:2] = box[:2] + translation
  box[4] = -box[4]
  return box


def bb_image_to_vehicle_system(box, pixels_per_meter, min_x, min_y):
  """
  Changed a bounding box from the vehicle x front, y right coordinate system
  to the x back, y right coordinate system of an image, where the center of
  the car is in the center of the image.
  :return:
  """
  box[4] = -box[4]
  # Compute pixel location that represents 0/0 in the image
  translation = np.array([-(min_x * pixels_per_meter), -(min_y * pixels_per_meter)])
  # Shift the coordinates so that the ego_vehicle is at [0,0]
  box[:2] = box[:2] - translation
  # Pixel coordinates is y front, x right. CARLA is x front, y right.
  # So we need to swap the axes to convert the coordinates.
  box[0], box[1] = box[1], box[0]
  box[2], box[3] = box[3], box[2]
  # Divide position and extent by pixels_per_meter to convert the unit from pixels to meters
  box[:4] = box[:4] / pixels_per_meter
  return box


def non_maximum_suppression(bounding_boxes, iou_treshhold):
  filtered_boxes = []
  bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

  if bounding_boxes.size == 0:  #If no bounding boxes are detected can't do NMS
    return filtered_boxes

  confidences_indices = np.argsort(bounding_boxes[:, -1])
  while len(confidences_indices) > 0:
    idx = confidences_indices[-1]
    current_bb = bounding_boxes[idx]
    filtered_boxes.append(current_bb)
    # Remove last element from the list
    confidences_indices = confidences_indices[:-1]

    if len(confidences_indices) == 0:
      break

    for idx2 in deepcopy(confidences_indices):
      if iou_bbs(current_bb, bounding_boxes[idx2]) > iou_treshhold:  # Remove BB from list
        confidences_indices = confidences_indices[confidences_indices != idx2]

  return filtered_boxes


def rect_polygon(x, y, width, height, angle):
  """Return a shapely Polygon describing the rectangle with centre at
  (x, y) and the given width and height, rotated by angle quarter-turns.

  """
  p = Polygon([(-width, -height), (width, -height), (width, height), (-width, height)])
  # Shapely is very inefficient at these operations, worth rewriting
  return shapely.affinity.translate(shapely.affinity.rotate(p, angle, use_radians=True), x, y)


def iou_bbs(bb1, bb2):
  a = rect_polygon(bb1[0], bb1[1], bb1[2], bb1[3], bb1[4])
  b = rect_polygon(bb2[0], bb2[1], bb2[2], bb2[3], bb2[4])
  intersection_area = a.intersection(b).area
  union_area = a.union(b).area
  iou = intersection_area / union_area
  return iou


def dot_product(vector1, vector2):
  return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z


def cross_product(vector1, vector2):
  return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y,
                        y=vector1.z * vector2.x - vector1.x * vector2.z,
                        z=vector1.x * vector2.y - vector1.y * vector2.x)


def get_separating_plane(r_pos, plane, obb1, obb2):
  ''' Checks if there is a seperating plane
      rPos Vec3
      plane Vec3
      obb1  Bounding Box
      obb2 Bounding Box
      '''
  return (abs(dot_product(r_pos, plane)) >
          (abs(dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
           abs(dot_product((obb1.rotation.get_right_vector() * obb1.extent.y), plane)) +
           abs(dot_product((obb1.rotation.get_up_vector() * obb1.extent.z), plane)) +
           abs(dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
           abs(dot_product((obb2.rotation.get_right_vector() * obb2.extent.y), plane)) +
           abs(dot_product((obb2.rotation.get_up_vector() * obb2.extent.z), plane))))


def check_obb_intersection(obb1, obb2):
  '''
  Checks whether two bounding boxes intersect
  Rather complex looking because it is the general algorithm for 3D oriented bounding boxes.
  '''
  r_pos = obb2.location - obb1.location
  return not (
      get_separating_plane(r_pos, obb1.rotation.get_forward_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb1.rotation.get_right_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb1.rotation.get_up_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb2.rotation.get_forward_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb2.rotation.get_right_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb2.rotation.get_up_vector(), obb1, obb2) or get_separating_plane(
          r_pos, cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_up_vector()), obb1, obb2))


def command_to_one_hot(command):
  if command < 0:
    command = 4
  command -= 1
  if command not in [0, 1, 2, 3, 4, 5]:
    command = 3
  cmd_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  cmd_one_hot[command] = 1.0

  return np.array(cmd_one_hot)


class InfoDummy(object):
  """ Info dummy that tries to mimic TIMMs info class"""

  def __init__(self, info):
    super().__init__()
    self.info = info


def calculate_intrinsic_matrix(fov, height, width):
  """ Intrinsics and extrinsics for a single camera.
  adapted from MILE
  https://github.com/wayveai/mile/blob/247280758b40ae999a5de14a8423f0d4db2655ac/mile/data/dataset.py#L194
  """

  # Intrinsics
  f = width / (2.0 * np.tan(fov * np.pi / 360.0))
  cx = width / 2.0
  cy = height / 2.0
  intrinsics = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

  return intrinsics


def normalize_imagenet(x):
  """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
  x = x.clone()
  x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
  x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
  x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
  return x


class CarlaActorDummy(object):
  """
  Actor dummy structure used to simulate a CARLA actor for data augmentation
  """
  world = None
  bounding_box = None
  transform = None
  id = None

  def __init__(self, world, bounding_box, transform, id):  # pylint: disable=locally-disabled, redefined-builtin
    self.world = world
    self.bounding_box = bounding_box
    self.transform = transform
    self.id = id

  def get_world(self):
    return self.world

  def get_transform(self):
    return self.transform

  def get_bounding_box(self):
    return self.bounding_box


def convert_depth(data):
  """
  Computes the normalized depth from a CARLA depth map.
  """
  #data = np.transpose(data, (1, 2, 0))
  data = data.astype(np.float32)

  normalized = np.dot(data, [65536.0, 256.0, 1.0])
  normalized /= (256 * 256 * 256 - 1)
  # in_meters = 1000 * normalized
  # clip to 50 meters
  normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
  normalized = normalized * 20.0  # Rescale map to lie in [0,1]

  return normalized


def create_projection_grid(config):
  """
  Creates a voxel grid around the car with each voxel containing the pixel index indicating the pixel
  it would land on if you project it into the camera of the car with a pinhole camera model.
  Also returns a valid mask indicating which voxels are visible from the camera.
  Because the coordinates are in normalized display coordinates, the image can also be a down-sampled version.
  :return: grid: voxel grid around the car. Each voxel contains the index of the corresponding camera pixel (x, y, 0).
           Coordinates are in normalized display coordinates [-1, 1].
           (-1,-1) is the top left pixel, (1,1) is the bottom right pixel .
           all_valid: The same voxel grid containing a bool that indicates whether the voxel is visible from the
           camera.
  """
  meters_per_pixel = 1.0 / config.pixels_per_meter
  # + half a pixel because we want the center of the voxel.
  widths = torch.arange(config.min_x, config.max_x, meters_per_pixel) + (meters_per_pixel * 0.5)
  depths = torch.arange(config.min_y, config.max_y, meters_per_pixel) + (meters_per_pixel * 0.5)
  meters_per_pixel_height = meters_per_pixel * config.bev_grid_height_downsample_factor
  heights = torch.arange(config.min_z_projection, config.max_z_projection,
                         meters_per_pixel_height) + (meters_per_pixel_height * 0.5)

  depths, widths, heights = torch.meshgrid(depths, widths, heights, indexing='ij')
  test_cloud = torch.stack((depths, widths, heights), dim=0)  # CARLA coordinate system
  _, d, w, h = test_cloud.shape  # channel, depth, width, height
  # If you rotate the camera adjust the rotation matrix here
  assert config.camera_rot_0[0] == config.camera_rot_0[1] == config.camera_rot_0[2] == 0.0
  rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  t = torch.tensor(config.camera_pos).unsqueeze(1)
  test_cloud2 = (rotation_matrix.T @ test_cloud.view(3, -1)) - (rotation_matrix.T @ t)

  # Convert CARLA coordiante system x front, y right, z up to pinhole coordinate system: x right, y down, z front
  test_cloud2 = torch.stack((test_cloud2[1], test_cloud2[2], test_cloud2[0]))

  # Apply intrinsic camera matrix
  intrinsic_matrix = torch.from_numpy(
      calculate_intrinsic_matrix(fov=config.camera_fov, height=config.camera_height,
                                 width=config.camera_width)).to(dtype=torch.float32)
  test_cloud2 = intrinsic_matrix @ test_cloud2

  depths = test_cloud2[2:3]
  grid = torch.zeros_like(test_cloud2).to(dtype=torch.float32)
  # Project to image coordinates using pinhole camera model
  # The depth grid is designed, so that the smallest number is +-0.125. Prevent division by 0 if you change it.
  grid[:2] = test_cloud2[:2] / depths

  # Note that the points themselfs are in pinhole camera coordinates, but the index in the grid represents the voxel
  # in the 3D volume
  grid = grid.view(3, d, w, h)

  width_valid = grid[0:1] >= 0.0
  witdh_valid2 = grid[0:1] < config.camera_width
  width_valid = torch.logical_and(width_valid, witdh_valid2)

  height_valid = grid[1:2] >= 0.0
  height_valid2 = grid[1:2] < config.camera_height
  height_valid = torch.logical_and(height_valid, height_valid2)

  depths = depths.view(1, d, w, h)
  depth_valid = depths > 0.0

  all_valid = torch.logical_and(width_valid, height_valid)
  all_valid = torch.logical_and(all_valid, depth_valid)

  # Normalizes pixel values to [-1, 1] normalized display coordinates
  grid[0:1] = (grid[0:1] / (0.5 * config.camera_width - 0.5)) - 1.0
  grid[1:2] = (grid[1:2] / (0.5 * config.camera_height - 0.5)) - 1.0

  grid = torch.reshape(grid, [1, 3, d, w, h, 1])
  grid = torch.transpose(grid, 1, 5).squeeze(1)

  return grid, all_valid.to(dtype=torch.float32)


class PerspectiveDecoder(nn.Module):
  """
  Decodes a low resolution perspective grid to a full resolution output. E.g. semantic segmentation, depth
  """

  def __init__(self, in_channels, out_channels, inter_channel_0, inter_channel_1, inter_channel_2, scale_factor_0,
               scale_factor_1):
    super().__init__()
    self.scale_factor_0 = scale_factor_0
    self.scale_factor_1 = scale_factor_1

    self.deconv1 = nn.Sequential(
        nn.Conv2d(in_channels, inter_channel_0, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(inter_channel_0, inter_channel_1, 3, 1, 1),
        nn.ReLU(True),
    )
    self.deconv2 = nn.Sequential(
        nn.Conv2d(inter_channel_1, inter_channel_2, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(inter_channel_2, inter_channel_2, 3, 1, 1),
        nn.ReLU(True),
    )
    self.deconv3 = nn.Sequential(
        nn.Conv2d(inter_channel_2, inter_channel_2, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(inter_channel_2, out_channels, 3, 1, 1),
    )

  def forward(self, x):
    x = self.deconv1(x)
    x = F.interpolate(x, scale_factor=self.scale_factor_0, mode='bilinear', align_corners=False)
    x = self.deconv2(x)
    x = F.interpolate(x, scale_factor=self.scale_factor_1, mode='bilinear', align_corners=False)
    x = self.deconv3(x)

    return x


def draw_probability_boxes(img, speed_prob, target_speeds, color=(128, 128, 128), color_selected=(255, 165, 0)):
  speed_index = np.argmax(speed_prob)
  colors = [color for _ in range(len(speed_prob))]
  colors[speed_index] = color_selected
  start_x = 0
  start_y = 719  # 1024-155-150  # start_x and start_y specify position of upper left corner of box
  width_bar = 20 * 4
  width_space = 10
  cv2.rectangle(img, (start_x, start_y), (1024, start_y + 155), (255, 255, 255), cv2.FILLED)

  for idx, s in enumerate(speed_prob):
    start = start_x + idx * (width_space + width_bar)
    cv2.rectangle(img, (start, start_y + 130), (start + width_bar, start_y + 130 - int(s * 100)), colors[idx], cv2.FILLED)
    cv2.putText(img, f'{s:.2f}', (int(start + 0.33 * width_bar), start_y + 127 - int(s * 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0), 1, cv2.LINE_AA)

    # 3.6 is conversion from m/s to km/h
    cv2.putText(img, f'{int(round(target_speeds[idx] * 3.6)):02d}', (int(start + 0.33 * width_bar), start_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

  cv2.putText(img, 'km/h', (start + width_bar + width_space, start_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
              cv2.LINE_AA)


def plant_quant_to_box(config, pred_bounding_boxes):
  """Convert a plant auxiliary class to an x,y location of a box"""
  pred_bb_x = F.softmax(pred_bounding_boxes[0][0], dim=1)
  pred_bb_y = F.softmax(pred_bounding_boxes[1][0], dim=1)
  pred_bb_ext_x = F.softmax(pred_bounding_boxes[2][0], dim=1)
  pred_bb_ext_y = F.softmax(pred_bounding_boxes[3][0], dim=1)
  pred_bb_yaw = F.softmax(pred_bounding_boxes[4][0], dim=1)
  pred_bb_speed = F.softmax(pred_bounding_boxes[5][0], dim=1)

  pred_bb_x = torch.argmax(pred_bb_x, dim=1)
  pred_bb_y = torch.argmax(pred_bb_y, dim=1)
  pred_bb_ext_x = torch.argmax(pred_bb_ext_x, dim=1)
  pred_bb_ext_y = torch.argmax(pred_bb_ext_y, dim=1)
  pred_bb_yaw = torch.argmax(pred_bb_yaw, dim=1)
  pred_bb_speed = torch.argmax(pred_bb_speed, dim=1)

  x_step = (config.max_x - config.min_x) / pow(2, config.plant_precision_pos)
  y_step = (config.max_y - config.min_y) / pow(2, config.plant_precision_pos)
  extent_step = (30) / pow(2, config.plant_precision_pos)
  yaw_step = (2 * np.pi) / pow(2, config.plant_precision_angle)
  speed_step = (config.plant_max_speed_pred / 3.6) / pow(2, config.plant_precision_speed)

  pred_bb_x = pred_bb_x * x_step - config.max_x
  pred_bb_y = pred_bb_y * y_step - config.max_y
  pred_bb_ext_x = pred_bb_ext_x * extent_step
  pred_bb_ext_y = pred_bb_ext_y * extent_step
  pred_bb_yaw = pred_bb_yaw * yaw_step - np.pi
  pred_bb_speed = pred_bb_speed * speed_step
  pred_bb_center = torch.stack((pred_bb_x, pred_bb_y, pred_bb_ext_x, pred_bb_ext_y, pred_bb_yaw, pred_bb_speed), dim=1)

  return pred_bb_center


def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
  """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

  :param circle_center: The (x, y) location of the circle center
  :param circle_radius: The radius of the circle
  :param pt1: The (x, y) location of the first point of the segment
  :param pt2: The (x, y) location of the second point of the segment
  :param full_line: True to find intersections along full line - not just in the segment.
                    False will just return intersections within the segment.
  :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a
                      tangent
  :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the
                                         circle intercepts a line segment.

  Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
  Credit: https://stackoverflow.com/a/59582674/9173068
  """

  if np.linalg.norm(pt1 - pt2) < 0.000000001:
    print('Problem')

  (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
  (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
  dx, dy = (x2 - x1), (y2 - y1)
  dr = (dx**2 + dy**2)**.5
  big_d = x1 * y2 - x2 * y1
  discriminant = circle_radius**2 * dr**2 - big_d**2

  if discriminant < 0:  # No intersection between circle and line
    return []
  else:  # There may be 0, 1, or 2 intersections with the segment
    # This makes sure the order along the segment is correct
    intersections = [(cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr**2,
                      cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr**2)
                     for sign in ((1, -1) if dy < 0 else (-1, 1))]
    if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
      fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
      intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
    # If line is tangent to circle, return just one point (as both intersections have same location)
    if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
      return [intersections[0]]
    else:
      return intersections


def crop_array(config, images_i):   # images_i must have dimensions (H,W,C) or (H,W)
    """
    Crop rgb images to the desired height and width
    """
    if config.crop_image:
      # crops rgb/depth/semantics from the bottom to cropped_height and symetrically from both sides to cropped_width
      assert config.cropped_height <= images_i.shape[0]
      assert config.cropped_width <= images_i.shape[1]
      side_crop_amount = (images_i.shape[1] - config.cropped_width) // 2
      if len(images_i.shape) > 2: # for rgb, we have 3 channels
        return images_i[0:config.cropped_height, side_crop_amount:images_i.shape[1]-side_crop_amount, :]
      else:  # for depth and semantics, there is no channel dimension
        return images_i[0:config.cropped_height, side_crop_amount:images_i.shape[1]-side_crop_amount]
    else: 
      return images_i