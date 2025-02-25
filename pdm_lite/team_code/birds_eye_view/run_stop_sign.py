"""
Tests if a stop sign is relevant for the a particular vehicle.
Code adapted from https://github.com/zhejz/carla-roach
"""

import carla
import numpy as np


class RunStopSign:
    """
    Criteria to test if a stop sign affects a vehicle.
    """

    def __init__(
        self,
        carla_world,
        proximity_threshold=50.0,
        speed_threshold=0.1,
        waypoint_step=1.0,
    ):
        self._map = carla_world.get_map()
        self._proximity_threshold = proximity_threshold
        self._speed_threshold = speed_threshold
        self._waypoint_step = waypoint_step

        all_actors = carla_world.get_actors()
        self._list_stop_signs = []
        for actor in all_actors:
            if "traffic.stop" in actor.type_id:
                self._list_stop_signs.append(actor)

        self.target_stop_sign = None
        self.stop_completed = False
        self.affected_by_stop = False

    def tick(self, vehicle):
        ev_loc = vehicle.get_location()

        if self.target_stop_sign is None:
            self.target_stop_sign = self._scan_for_stop_sign(vehicle.get_transform())
        else:
            # we were in the middle of dealing with a stop sign
            if not self.stop_completed:
                # did the ego-vehicle stop?
                current_speed = self._calculate_speed(vehicle.get_velocity())
                if current_speed < self._speed_threshold:
                    self.stop_completed = True

            if not self.affected_by_stop:
                stop_t = self.target_stop_sign.get_transform()
                transformed_tv = stop_t.transform(
                    self.target_stop_sign.trigger_volume.location
                )
                stop_extent = self.target_stop_sign.trigger_volume.extent
                if self.point_inside_boundingbox(ev_loc, transformed_tv, stop_extent):
                    self.affected_by_stop = True

            if not self.is_affected_by_stop(ev_loc, self.target_stop_sign):
                # is the vehicle out of the influence of this stop sign now?
                # reset state
                self.target_stop_sign = None
                self.stop_completed = False
                self.affected_by_stop = False

    def _scan_for_stop_sign(self, vehicle_transform):
        target_stop_sign = None

        ve_dir = vehicle_transform.get_forward_vector()

        wp = self._map.get_waypoint(vehicle_transform.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self.is_affected_by_stop(vehicle_transform.location, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

    def is_affected_by_stop(self, vehicle_loc, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        stop_t = stop.get_transform()
        stop_location = stop_t.location
        if stop_location.distance(vehicle_loc) > self._proximity_threshold:
            return affected

        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [vehicle_loc]
        waypoint = self._map.get_waypoint(vehicle_loc)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(self._waypoint_step)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(
                actor_location, transformed_tv, stop.trigger_volume.extent
            ):
                affected = True

        return affected

    @staticmethod
    def _calculate_speed(carla_velocity):
        return np.linalg.norm([carla_velocity.x, carla_velocity.y])

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        """
        X
        :param point:
        :param bb_center:
        :param bb_extent:
        :return:
        """
        # bugfix slim bbox
        bb_extent.x = max(bb_extent.x, bb_extent.y)
        bb_extent.y = max(bb_extent.x, bb_extent.y)

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return 0 < am_ab < ab_ab and 0 < am_ad < ad_ad
