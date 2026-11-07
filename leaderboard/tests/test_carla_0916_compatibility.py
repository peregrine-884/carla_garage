"""Offline regression tests using the installed CARLA PythonAPI (no server).

From the workspace root, with CARLA/leaderboard/scenario_runner on PYTHONPATH:
    python -m unittest discover -s external/carla_garage/leaderboard/tests -v
"""

import unittest
from unittest.mock import Mock, patch

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

from leaderboard.autoagents.agent_wrapper_local import AgentWrapper
from leaderboard.envs.sensor_interface import (
    BaseReader, OpenDriveMapReader, SpeedometerReader,
)
from leaderboard.utils import route_manipulation as leaderboard_routes
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ChangeActorWaypointsToReachPosition, calculate_distance,
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTimeToArrivalToOSCPosition, InTimeToArrivalToVehicle,
    InTriggerDistanceToOSCPosition,
)
from srunner.tools import route_manipulation as scenario_runner_routes


# Two linked roads give CARLA a connected topology, entirely in memory.
STRAIGHT_ROAD = """<?xml version="1.0"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="offline" version="1.00"
          north="0" south="0" east="160" west="0">
    <geoReference><![CDATA[+proj=tmerc +lat_0=42 +lon_0=2 +k=1
      +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference>
  </header>
  <road name="straight" length="80" id="1" junction="-1">
    <link><successor elementType="road" elementId="2" contactPoint="start"/></link>
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="80"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link><successor id="-1"/></link>
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <road name="continuation" length="80" id="2" junction="-1">
    <link><predecessor elementType="road" elementId="1" contactPoint="end"/></link>
    <planView>
      <geometry s="0" x="80" y="0" hdg="0" length="80"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link><predecessor id="-1"/></link>
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


class RoutePlannerCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carla_map = carla.Map("offline", STRAIGHT_ROAD)
        cls.locations = [
            cls.carla_map.get_waypoint_xodr(1, -1, s).transform.location
            for s in (1.0, 20.0, 40.0)
        ]

    @staticmethod
    def route_signature(route):
        return [
            (transform.location.x, transform.location.y, transform.location.z,
             transform.rotation.yaw, option)
            for transform, option in route
        ]

    def test_interpolation_preserves_segments_resolution_and_gps(self):
        world = Mock()
        world.get_map.return_value = self.carla_map
        for module in (leaderboard_routes, scenario_runner_routes):
            for resolution in (0.5, 1.0, 2.0):
                with self.subTest(module=module.__name__, resolution=resolution):
                    planner = GlobalRoutePlanner(self.carla_map, resolution)
                    expected = []
                    for start, end in zip(self.locations, self.locations[1:]):
                        expected.extend(
                            (waypoint.transform, option)
                            for waypoint, option in planner.trace_route(start, end)
                        )

                    gps, route = module.interpolate_trajectory(
                        world, self.locations, hop_resolution=resolution)

                    self.assertGreater(len(route), len(self.locations))
                    self.assertEqual(self.route_signature(route),
                                     self.route_signature(expected))
                    self.assertTrue(all(option == RoadOption.LANEFOLLOW
                                        for _, option in route))
                    self.assertLess(route[-1][0].location.distance(self.locations[-1]),
                                    2 * resolution)
                    self.assertEqual(gps, module.location_route_to_gps(expected, 42.0, 2.0))

        world.tick.assert_not_called()

    def test_along_route_conditions_build_working_planners(self):
        actor, other_actor = Mock(), Mock()
        with patch.object(CarlaDataProvider, "get_map", return_value=self.carla_map):
            conditions = [
                InTriggerDistanceToOSCPosition(actor, "position", 50, along_route=True),
                InTimeToArrivalToOSCPosition(actor, "position", 5, along_route=True),
                InTimeToArrivalToVehicle(actor, other_actor, 5, along_route=True),
            ]

        for condition in conditions:
            with self.subTest(condition=type(condition).__name__):
                self.assertEqual(condition._grp._sampling_resolution, 0.5)
                distance = calculate_distance(
                    self.locations[0], self.locations[-1], condition._grp)
                self.assertGreater(distance, 38.0)
                self.assertLessEqual(distance, 39.0)

    def test_actor_waypoint_behavior_generates_route(self):
        actor = Mock(id=123)
        with patch.object(CarlaDataProvider, "get_map", return_value=self.carla_map):
            behavior = ChangeActorWaypointsToReachPosition(
                actor, carla.Transform(self.locations[-1]))

        self.assertEqual(behavior._grp._sampling_resolution, 2)
        with patch.object(CarlaDataProvider, "get_location", return_value=self.locations[0]), \
                patch("srunner.scenariomanager.scenarioatomics.atomic_behaviors."
                      "ChangeActorWaypoints.initialise"):
            behavior.initialise()

        self.assertGreater(len(behavior._waypoints), 10)
        # The planner chooses the nearest point on its 2 m sampling grid.
        self.assertLess(behavior._waypoints[0].location.distance(self.locations[0]), 2.0)
        self.assertLess(behavior._waypoints[-1].location.distance(self.locations[-1]), 4.0)


class AgentWrapperCleanupTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = AgentWrapper(Mock())
        self.world_access = patch.object(CarlaDataProvider, "get_world")
        self.get_world = self.world_access.start()
        self.addCleanup(self.world_access.stop)

    def tearDown(self):
        # Cleanup must not introduce a synchronous tick or even fetch the world.
        self.get_world.assert_not_called()

    def add_sensor(self, wrapper=None):
        wrapper = self.wrapper if wrapper is None else wrapper
        # Pseudo sensors do not expose CARLA actor fields such as id/is_alive.
        sensor = Mock(spec=["stop", "destroy"])
        wrapper._sensors_list.append(sensor)
        wrapper.sensor_list_names.append(["sensor", sensor])
        return sensor

    def assert_cleaned(self, sensors):
        for sensor in sensors:
            sensor.stop.assert_called_once_with()
            sensor.destroy.assert_called_once_with()
        self.assertEqual(self.wrapper._sensors_list, [])
        self.assertEqual(self.wrapper.sensor_list_names, [])
        self.assertTrue(self.wrapper._cleanup_done)

    def test_normal_cleanup(self):
        sensors = [self.add_sensor(), self.add_sensor()]
        self.wrapper._sensors_list.insert(1, None)
        self.wrapper.cleanup()
        self.assert_cleaned(sensors)

    def test_stop_failure_still_destroys_all_sensors(self):
        sensors = [self.add_sensor(), self.add_sensor()]
        sensors[0].stop.side_effect = RuntimeError("stop failed")
        with self.assertLogs(level="WARNING") as logs:
            self.wrapper.cleanup()
        self.assert_cleaned(sensors)
        self.assertIn("stop failed", logs.output[0])

    def test_destroy_failure_still_cleans_remaining_sensors(self):
        sensors = [self.add_sensor(), self.add_sensor()]
        sensors[0].destroy.side_effect = RuntimeError("destroy failed")
        with self.assertLogs(level="WARNING") as logs:
            self.wrapper.cleanup()
        self.assert_cleaned(sensors)
        self.assertIn("destroy failed", logs.output[0])

    def test_both_failures_are_reported_and_cleanup_is_idempotent(self):
        sensors = [self.add_sensor(), self.add_sensor()]
        sensors[0].stop.side_effect = RuntimeError("stop failed")
        sensors[0].destroy.side_effect = RuntimeError("destroy failed")
        with self.assertLogs(level="WARNING") as logs:
            self.wrapper.cleanup()
            self.wrapper.cleanup()
        self.assertEqual(len(logs.output), 2)
        self.assertIn("stop failed", logs.output[0])
        self.assertIn("destroy failed", logs.output[1])
        self.assert_cleaned(sensors)

    def test_second_cleanup_does_not_touch_sensors(self):
        sensor = self.add_sensor()
        self.wrapper.cleanup()
        self.wrapper.cleanup()
        self.assert_cleaned([sensor])

    def test_instances_have_independent_sensor_lists(self):
        other = AgentWrapper(Mock())
        first_sensor = self.add_sensor()
        other_sensor = self.add_sensor(other)
        self.assertIsNot(self.wrapper._sensors_list, other._sensors_list)
        self.assertIsNot(self.wrapper.sensor_list_names, other.sensor_list_names)

        self.wrapper.cleanup()

        self.assert_cleaned([first_sensor])
        self.assertEqual(other._sensors_list, [other_sensor])
        self.assertEqual(other.sensor_list_names, [["sensor", other_sensor]])
        self.assertFalse(other._cleanup_done)
        other_sensor.stop.assert_not_called()
        other_sensor.destroy.assert_not_called()
        other.cleanup()
        other_sensor.stop.assert_called_once_with()
        other_sensor.destroy.assert_called_once_with()

    def test_pseudo_sensor_setup_and_cleanup_preserve_tick_behavior(self):
        world = Mock()
        world.get_settings.return_value.fixed_delta_seconds = 0.05
        self.wrapper._agent.sensors.return_value = [
            {"type": "sensor.opendrive_map", "id": "map", "reading_frequency": 1},
            {"type": "sensor.speedometer", "id": "speed"},
        ]
        # Avoid starting reader threads; exercise their real lifecycle methods.
        with patch.object(CarlaDataProvider, "get_world", return_value=world), \
                patch.object(BaseReader, "run"):
            self.wrapper.setup_sensors(Mock())
        readers = list(self.wrapper._sensors_list)
        self.assertIsInstance(readers[0], OpenDriveMapReader)
        self.assertIsInstance(readers[1], SpeedometerReader)
        self.assertEqual(readers[1]._reading_frequency, 20.0)
        self.assertTrue(all(reader._run_ps for reader in readers))
        self.assertTrue(all(reader._callback is not None for reader in readers))
        world.spawn_actor.assert_not_called()
        world.tick.assert_called_once_with()

        self.wrapper.cleanup()
        self.wrapper.cleanup()

        self.assertTrue(all(not reader._run_ps for reader in readers))
        self.assertEqual(self.wrapper._sensors_list, [])
        self.assertEqual(self.wrapper.sensor_list_names, [])
        world.tick.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
