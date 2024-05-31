"""
Creates log files during evaluation with which we can visualize failures
"""

import os
import json
import carla
import gzip
import numpy as np
from rdp import rdp


class ScenarioLogger():
  """
  Creates log files during evaluation with which we can visualize failures
  """

  def __init__(self, save_path, route_index, logging_freq, log_only, route_only, roi=30, rdp_epsilon=0.5) -> None:
    """
        """
    # logger_settings
    self.logging_freq = logging_freq
    self.log_only = log_only
    self.route_only = route_only

    self.roi = roi  # radius around ego agent in meters
    self.rdp_epsilon = rdp_epsilon  # hyperparameter for RDP simplification

    # meta data
    self.save_path = save_path
    self.route_index = route_index

    # simulation objects
    self.world = None
    self.ego_vehicle = None
    self.step = 0

    # logging objects
    self.states = []
    self.lights = []
    self.route_boxes = []
    self.ego_actions = []
    self.adv_actions = []

    self.ego_pos = None
    self.ego_yaw = None
    self.ego_vel = None
    self.ego_extent = None
    self.ego_id = None
    self.ego_type = None
    self.ego_color = None
    self.ego_height = None
    self.ego_pitch = None
    self.ego_roll = None

    self.bg_vehicles = []
    self.bg_pos = None
    self.bg_yaw = None
    self.bg_vel = None
    self.bg_extent = None
    self.bg_id = None
    self.bg_type = None
    self.bg_color = None
    self.bg_height = None
    self.bg_pitch = None
    self.bg_roll = None

    self.bg_steer = None
    self.bg_throttle = None
    self.bg_brake = None

    self.tlights = []
    self.tl_pos = None
    self.tl_yaw = None
    self.tl_state = None
    self.tl_extent = None

    self.route_pos = None
    self.route_yaw = None
    self.route_id = None
    self.route_extent = None

  def _initialize_bg_agents(self):
    """
    _initialize_bg_agents
    """
    actors = self.world.get_actors()

    vehicles = actors.filter("*vehicle*")
    self.bg_vehicles = []
    for vehicle in vehicles:
      if vehicle.id != self.ego_vehicle.id:
        vehicle_location = vehicle.get_transform().location
        if vehicle_location.distance(self.ego_location) < self.roi:
          self.bg_vehicles.append(vehicle)

    tlights = actors.filter("*traffic_light*")
    self.tlights = []
    for tlight in tlights:
      if tlight.state != carla.libcarla.TrafficLightState.Green:
        trigger_box_global_pos = tlight.get_transform().transform(tlight.trigger_volume.location)
        trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x,
                                                y=trigger_box_global_pos.y,
                                                z=trigger_box_global_pos.z)
        if trigger_box_global_pos.distance(self.ego_location) < self.roi:
          self.tlights.append(tlight)

  def fetch_bg_state(self):
    """
    fetch_bg_state
    """
    # vehicles
    positions = []
    yaws = []
    velocities = []
    extents = []
    ids = []
    types = []
    colors = []
    heigths = []
    pitchs = []
    rolls = []

    for vehicle in self.bg_vehicles:
      # shape is batch_size x num_agents x state_dims
      positions.append(np.array([[[vehicle.get_location().x, vehicle.get_location().y]]]))
      yaws.append(np.array([[[vehicle.get_transform().rotation.yaw]]]))
      velocities.append(np.array([[[vehicle.get_velocity().x, vehicle.get_velocity().y]]]))
      extents.append(
          np.array([[[
              [vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x],
              [vehicle.bounding_box.extent.y, -vehicle.bounding_box.extent.x],
              [-vehicle.bounding_box.extent.y, -vehicle.bounding_box.extent.x],
              [-vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x],
          ]]]))
      ids.append(np.array([[[vehicle.id]]]))

      types.append(np.array([[[vehicle.type_id]]]))
      try:
        colors.append(np.array([[[vehicle.attributes["color"]]]]))
      except KeyError:
        colors.append(np.array([[["0,0,0"]]]))

      heigths.append(np.array([[[vehicle.get_location().z]]]))

      pitchs.append(np.array([[[vehicle.get_transform().rotation.pitch]]]))

      rolls.append(np.array([[[vehicle.get_transform().rotation.roll]]]))

    if self.bg_pos is None and len(positions) > 0:
      self.bg_pos = np.concatenate(positions, axis=1)
    if self.bg_yaw is None and len(yaws) > 0:
      self.bg_yaw = np.concatenate(np.radians(yaws), axis=1)
    if self.bg_vel is None and len(velocities) > 0:
      self.bg_vel = np.concatenate(velocities, axis=1)
    if self.bg_extent is None and len(extents) > 0:
      self.bg_extent = np.concatenate(extents, axis=1)
    if self.bg_id is None and len(ids) > 0:
      self.bg_id = np.concatenate(ids, axis=1)
    if self.bg_type is None and len(types) > 0:
      self.bg_type = np.concatenate(types, axis=1)
    if self.bg_color is None and len(colors) > 0:
      self.bg_color = np.concatenate(colors, axis=1)
    if self.bg_height is None and len(heigths) > 0:
      self.bg_height = np.concatenate(heigths, axis=1)
    if self.bg_pitch is None and len(pitchs) > 0:
      self.bg_pitch = np.concatenate(pitchs, axis=1)
    if self.bg_roll is None and len(rolls) > 0:
      self.bg_roll = np.concatenate(rolls, axis=1)

    # traffic lights
    tl_positions = []
    tl_yaws = []
    tl_states = []
    tl_extents = []
    for tlight in self.tlights:
      if tlight.state == carla.libcarla.TrafficLightState.Red:
        state = 0
      elif tlight.state == carla.libcarla.TrafficLightState.Yellow:
        state = 1
      else:  # unknown
        state = -1
        continue

      center_bounding_box = tlight.get_transform().transform(tlight.trigger_volume.location)
      center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
      length_bounding_box = carla.Vector3D(tlight.trigger_volume.extent.x, tlight.trigger_volume.extent.y,
                                           tlight.trigger_volume.extent.z)
      # can only create a bounding box from a transform.location, not from a
      # location
      transform = carla.Transform(center_bounding_box)
      bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

      gloabl_rot = tlight.get_transform().rotation
      bounding_box.rotation = carla.Rotation(pitch=tlight.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                             yaw=tlight.trigger_volume.rotation.yaw + gloabl_rot.yaw,
                                             roll=tlight.trigger_volume.rotation.roll + gloabl_rot.roll)

      # shape is batch_size x num_agents x state_dims
      tl_positions.append(np.array([[[center_bounding_box.x, center_bounding_box.y]]]))
      tl_yaws.append(np.array([[[bounding_box.rotation.yaw]]]))
      tl_states.append(np.array([[[state]]]))
      tl_extents.append(
          np.array([[[
              [bounding_box.extent.y, bounding_box.extent.x],
              [bounding_box.extent.y, -bounding_box.extent.x],
              [-bounding_box.extent.y, -bounding_box.extent.x],
              [-bounding_box.extent.y, bounding_box.extent.x],
          ]]]))

    if self.tl_pos is None and len(tl_positions) > 0:
      self.tl_pos = np.concatenate(tl_positions, axis=1)
    if self.tl_yaw is None and len(tl_yaws) > 0:
      self.tl_yaw = np.concatenate(np.radians(tl_yaws), axis=1)
    if self.tl_state is None and len(tl_states) > 0:
      self.tl_state = np.concatenate(tl_states, axis=1)
    if self.tl_extent is None and len(tl_extents) > 0:
      self.tl_extent = np.concatenate(tl_extents, axis=1)

  def log_step(self, route, ego_control=None):
    """
    log_step
    """
    self.step += 1

    # only save sim state to log every k frames
    if self.log_only and self.step % self.logging_freq != 0:
      return

    # we check if the logging objects are already set to avoid redundant
    # computation in case they can be set from the agent
    # shape is batch_size x num_agents x state_dims so e.g. 1 x 1 x 2 for
    # the ego pos (since bs is always 1 and we just have a single ego agent)
    if not self.ego_extent:
      ego_extent = self.ego_vehicle.bounding_box.extent
      self.ego_extent = np.array([[[
          [ego_extent.y, ego_extent.x],
          [ego_extent.y, -ego_extent.x],
          [-ego_extent.y, -ego_extent.x],
          [-ego_extent.y, ego_extent.x],
      ]]])

    if not self.ego_pos:
      self.ego_pos = self.ego_vehicle.get_location()
      self.ego_pos = np.array([[[self.ego_pos.x, self.ego_pos.y]]])

    if not self.ego_yaw:
      self.ego_orientation = self.ego_vehicle.get_transform()
      self.ego_yaw = np.array([[[np.radians(self.ego_orientation.rotation.yaw)]]])

    if not self.ego_vel:
      self.ego_vel = self.ego_vehicle.get_velocity()
      self.ego_vel = np.array([[[self.ego_vel.x, self.ego_vel.y]]])

    if not self.ego_id:
      self.ego_id = np.array([[[self.ego_vehicle.id]]])

    if not self.ego_type:
      self.ego_type = np.array([[[self.ego_vehicle.type_id]]])

    if not self.ego_color:
      self.ego_color = np.array([[[self.ego_vehicle.attributes["color"]]]])

    if not self.ego_height:
      self.ego_height = np.array([[[self.ego_vehicle.get_location().z]]])

    if not self.ego_pitch:
      self.ego_orientation = self.ego_vehicle.get_transform()
      self.ego_pitch = np.array([[[np.radians(self.ego_orientation.rotation.pitch)]]])

    if not self.ego_roll:
      self.ego_orientation = self.ego_vehicle.get_transform()
      self.ego_roll = np.array([[[np.radians(self.ego_orientation.rotation.roll)]]])

    self.ego_location = self.ego_vehicle.get_location()

    # fetch relevant (nearby) background agents
    if not self.route_only:
      self._initialize_bg_agents()

    # fetch and concat adv states
    self.fetch_bg_state()

    # vehicles
    if len(self.bg_vehicles) > 0 and self.bg_color is not None:
      state = {
          "pos": np.concatenate([self.ego_pos, self.bg_pos], axis=1).tolist(),
          "yaw": np.concatenate([self.ego_yaw, self.bg_yaw], axis=1).tolist(),
          "vel": np.concatenate([self.ego_vel, self.bg_vel], axis=1).tolist(),
          "extent": np.concatenate([self.ego_extent, self.bg_extent], axis=1).tolist(),
          "id": np.concatenate([self.ego_id, self.bg_id], axis=1).tolist(),
          "type": np.concatenate([self.ego_type, self.bg_type], axis=1).tolist(),
          "color": np.concatenate([self.ego_color, self.bg_color], axis=1).tolist(),
          "height": np.concatenate([self.ego_height, self.bg_height], axis=1).tolist(),
          "pitch": np.concatenate([self.ego_pitch, self.bg_pitch], axis=1).tolist(),
          "roll": np.concatenate([self.ego_roll, self.bg_roll], axis=1).tolist(),
      }
    else:
      state = {
          "pos": self.ego_pos.tolist(),
          "yaw": self.ego_yaw.tolist(),
          "vel": self.ego_vel.tolist(),
          "extent": self.ego_extent.tolist(),
          "id": self.ego_id.tolist(),
          "type": self.ego_type.tolist(),
          "color": self.ego_color.tolist(),
          "height": self.ego_height.tolist(),
          "pitch": self.ego_pitch.tolist(),
          "roll": self.ego_roll.tolist(),
      }

    # traffic lights
    if len(self.tlights) > 0:
      lights = {
          "pos": self.tl_pos.tolist(),
          "yaw": self.tl_yaw.tolist(),
          "state": self.tl_state.tolist(),
          "extent": self.tl_extent.tolist(),
      }
    else:
      lights = {
          "pos": [],
          "yaw": [],
          "state": [],
          "extent": [],
      }

    # route
    self.route_as_boxes(route)
    route_boxes = {
        "pos": self.route_pos.tolist(),
        "yaw": self.route_yaw.tolist(),
        "id": self.route_id.tolist(),
        "extent": self.route_extent.tolist(),
    }

    # actions
    # ego action logging only if provided
    if ego_control is not None:
      ego_steer = [[[ego_control.steer]]]
      ego_throttle = [[[ego_control.throttle]]]
      ego_brake = [[[ego_control.brake]]]

      ego_actions = {"steer": ego_steer, "throttle": ego_throttle, "brake": ego_brake}

    if len(self.bg_vehicles) > 0:
      self.fetch_bg_actions()
      bg_actions = {
          "steer": self.bg_steer.tolist(),
          "throttle": self.bg_throttle.tolist(),
          "brake": self.bg_brake.tolist(),
      }
    else:
      bg_actions = {
          "steer": [],
          "throttle": [],
          "brake": [],
      }

    # only save sim state to log every k frames
    if not self.route_only and self.step % self.logging_freq == 0:
      self.states.append(state)
      self.lights.append(lights)
      self.route_boxes.append(route_boxes)
      if ego_control:
        self.ego_actions.append(ego_actions)
      self.adv_actions.append(bg_actions)

    # reset logging objects after storing current timestep
    self.ego_pos = None
    self.ego_yaw = None
    self.ego_vel = None
    self.ego_extent = None
    self.ego_id = None
    self.ego_type = None
    self.ego_color = None
    self.ego_height = None
    self.ego_pitch = None
    self.ego_roll = None

    self.bg_pos = None
    self.bg_yaw = None
    self.bg_vel = None
    self.bg_extent = None
    self.bg_id = None
    self.bg_type = None
    self.bg_color = None
    self.bg_height = None
    self.bg_pitch = None
    self.bg_roll = None

    self.bg_steer = None
    self.bg_throttle = None
    self.bg_brake = None

    self.tl_pos = None
    self.tl_yaw = None
    self.tl_state = None
    self.tl_extent = None

    self.route_pos = None
    self.route_yaw = None
    self.route_id = None
    self.route_extent = None

    return state, lights, route_boxes, bg_actions

  def route_as_boxes(self, route):
    """
    route_as_boxes
    """
    shortened_route = rdp(route, epsilon=self.rdp_epsilon)

    # convert points to vectors
    vectors = shortened_route[1:] - shortened_route[:-1]
    midpoints = shortened_route[:-1] + vectors / 2.
    norms = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    route_positions = []
    route_yaws = []
    route_ids = []
    route_extents = []
    for i, midpoint in enumerate(midpoints):
      # represent the route element as a bounding box
      center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
      transform = carla.Transform(center_bounding_box)

      # only store route boxes that are near the ego vehicle
      start_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
      if 0 < i < 10 and start_bounding_box.distance(self.ego_location) > self.roi:
        continue

      length_bounding_box = carla.Vector3D(norms[i] / 2., self.ego_vehicle.bounding_box.extent.y,
                                           self.ego_vehicle.bounding_box.extent.z)
      bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
      bounding_box.rotation = carla.Rotation(pitch=0.0, yaw=angles[i] * 180 / np.pi, roll=0.0)

      # shape is batch_size x num_agents x state_dims
      route_positions.append(np.array([[[center_bounding_box.x, center_bounding_box.y]]]))
      route_yaws.append(np.array([[[bounding_box.rotation.yaw]]]))
      route_ids.append(np.array([[[i]]]))
      route_extents.append(
          np.array([[[
              [bounding_box.extent.y, bounding_box.extent.x],
              [bounding_box.extent.y, -bounding_box.extent.x],
              [-bounding_box.extent.y, -bounding_box.extent.x],
              [-bounding_box.extent.y, bounding_box.extent.x],
          ]]]))

    if self.route_pos is None and len(route_positions) > 0:
      self.route_pos = np.concatenate(route_positions, axis=1)
    if self.route_yaw is None and len(route_yaws) > 0:
      self.route_yaw = np.concatenate(np.radians(route_yaws), axis=1)
    if self.route_id is None and len(route_ids) > 0:
      self.route_id = np.concatenate(route_ids, axis=1)
    if self.route_extent is None and len(route_extents) > 0:
      self.route_extent = np.concatenate(route_extents, axis=1)

  def fetch_bg_actions(self):
    """
    fetch_bg_actions
    """
    steers = []
    throttles = []
    brakes = []
    for vehicle in self.bg_vehicles:
      # shape is batch_size x num_agents x state_dims
      steers.append(np.array([[[vehicle.get_control().steer]]]))
      throttles.append(np.array([[[vehicle.get_control().throttle]]]))
      brakes.append(np.array([[[vehicle.get_control().brake]]]))

    if self.bg_steer is None:
      self.bg_steer = np.concatenate(steers, axis=1)
    if self.bg_throttle is None:
      self.bg_throttle = np.concatenate(throttles, axis=1)
    if self.bg_brake is None:
      self.bg_brake = np.concatenate(brakes, axis=1)

  def dump_to_json(self):
    """
    dump_to_json
    """
    if not self.route_only:
      self.records_file_path = os.path.join(self.save_path, "records.json.gz")
      if not os.path.exists(self.save_path):
        os.mkdir(os.path.join(self.save_path))

      if self.world is not None:
        town_name = self.world.get_map().name
      else:
        town_name = "Unkown"

      meta_data = {
          "index": self.route_index,
          "town": town_name,
      }
      records_dict = {
          "meta_data": meta_data,
          "states": [],
          "lights": [],
          "route": [],
          "ego_actions": [],
          "adv_actions": [],
      }

      for i in range(len(self.states)):
        records_dict["states"].append(self.states[i])
        records_dict["lights"].append(self.lights[i])
        records_dict["route"].append(self.route_boxes[i])
        records_dict["adv_actions"].append(self.adv_actions[i])

      # ego actions only logged if provided
      for i in range(len(self.ego_actions)):
        records_dict["ego_actions"].append(self.ego_actions[i])

      with gzip.open(self.records_file_path, "wt", encoding="utf-8") as f:
        json.dump(records_dict, f)
