"""
Transformation utility functions.
Code adapted from https://github.com/zhejz/carla-roach
"""

import numpy as np
import carla


def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
  """
    :param target_loc_in_global: carla.Location in global coordinate (world, actor)
    :param ref_trans_in_global: carla.Transform in global coordinate (world, actor)
    :return: carla.Location in ref coordinate
    """
  x = target_loc_in_global.x - ref_trans_in_global.location.x
  y = target_loc_in_global.y - ref_trans_in_global.location.y
  z = target_loc_in_global.z - ref_trans_in_global.location.z
  vec_in_global = carla.Vector3D(x=x, y=y, z=z)
  vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)

  target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
  return target_loc_in_ref


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
  """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
  rotation = carla_rot_to_mat(ref_rot_in_global)
  np_vec_in_global = np.array([[target_vec_in_global.x], [target_vec_in_global.y], [target_vec_in_global.z]])
  np_vec_in_ref = rotation.T.dot(np_vec_in_global)
  target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
  return target_vec_in_ref


def rot_global_to_ref(target_rot_in_global, ref_rot_in_global):
  target_roll_in_ref = cast_angle(target_rot_in_global.roll - ref_rot_in_global.roll)
  target_pitch_in_ref = cast_angle(target_rot_in_global.pitch - ref_rot_in_global.pitch)
  target_yaw_in_ref = cast_angle(target_rot_in_global.yaw - ref_rot_in_global.yaw)

  target_rot_in_ref = carla.Rotation(roll=target_roll_in_ref, pitch=target_pitch_in_ref, yaw=target_yaw_in_ref)
  return target_rot_in_ref


def rot_ref_to_global(target_rot_in_ref, ref_rot_in_global):
  target_roll_in_global = cast_angle(target_rot_in_ref.roll + ref_rot_in_global.roll)
  target_pitch_in_global = cast_angle(target_rot_in_ref.pitch + ref_rot_in_global.pitch)
  target_yaw_in_global = cast_angle(target_rot_in_ref.yaw + ref_rot_in_global.yaw)

  target_rot_in_global = carla.Rotation(roll=target_roll_in_global,
                                        pitch=target_pitch_in_global,
                                        yaw=target_yaw_in_global)
  return target_rot_in_global


def carla_rot_to_mat(carla_rotation):
  """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation
    :return: np.array rotation matrix
  """
  roll = np.deg2rad(carla_rotation.roll)
  pitch = np.deg2rad(carla_rotation.pitch)
  yaw = np.deg2rad(carla_rotation.yaw)

  yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
  pitch_matrix = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
  roll_matrix = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])

  rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
  return rotation_matrix


def get_loc_rot_vel_in_ev(actor_list, ev_transform):
  location, rotation, absolute_velocity = [], [], []
  for actor in actor_list:
    # location
    location_in_world = actor.get_transform().location
    location_in_ev = loc_global_to_ref(location_in_world, ev_transform)
    location.append([location_in_ev.x, location_in_ev.y, location_in_ev.z])
    # rotation
    rotation_in_world = actor.get_transform().rotation
    rotation_in_ev = rot_global_to_ref(rotation_in_world, ev_transform.rotation)
    rotation.append([rotation_in_ev.roll, rotation_in_ev.pitch, rotation_in_ev.yaw])
    # velocity
    vel_in_world = actor.get_velocity()
    vel_in_ev = vec_global_to_ref(vel_in_world, ev_transform.rotation)
    absolute_velocity.append([vel_in_ev.x, vel_in_ev.y, vel_in_ev.z])
  return location, rotation, absolute_velocity


def cast_angle(x):
  # cast angle to [-180, +180)
  return (x + 180.0) % 360.0 - 180.0
