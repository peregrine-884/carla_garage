"""Research opt-in and failure cleanup, without a CARLA server or watchdog thread."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from leaderboard.autoagents.agent_wrapper_local import AgentWrapper, RESEARCH_CAMERA_IDS, SENSORS_LIMITS
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.utils.research_metadata import save_research_metadata
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def cameras():
    return [{'id': name, 'type': 'sensor.camera.rgb', 'x': 0., 'y': 0., 'z': 1.,
             'roll': 0., 'pitch': 0., 'yaw': 0., 'width': 1600, 'height': 900,
             'fov': 65., 'sensor_tick': .05} for name in RESEARCH_CAMERA_IDS]


class ResearchSensorTests(unittest.TestCase):
    def test_official_default_rejects_six_in_both_tracks(self):
        for track in (Track.SENSORS, Track.MAP):
            with self.subTest(track=track), self.assertRaisesRegex(SensorConfigurationInvalid, 'Maximum number allowed is 4'):
                AgentWrapper.validate_sensor_configuration(cameras(), track, track.value)

    def test_explicit_research_accepts_existing_six_without_changing_defaults(self):
        before = dict(SENSORS_LIMITS)
        for track in (Track.SENSORS, Track.MAP):
            AgentWrapper.validate_sensor_configuration(cameras(), track, track.value, research_six_camera=True)
        self.assertEqual(SENSORS_LIMITS, before)
        AgentWrapper.validate_sensor_configuration(cameras()[:4], Track.SENSORS, 'SENSORS')

    def test_research_rejects_different_order_and_seventh_camera(self):
        for sensors in (list(reversed(cameras())), cameras() + [{'id': 'extra', 'type': 'sensor.camera.rgb'}]):
            with self.assertRaises(SensorConfigurationInvalid):
                AgentWrapper.validate_sensor_configuration(sensors, Track.SENSORS, 'SENSORS', True)

    def test_other_limits_and_track_restrictions_still_apply(self):
        for kind, count in [('sensor.other.gnss', 2), ('sensor.lidar.ray_cast', 2),
                            ('sensor.other.imu', 2), ('sensor.other.radar', 3),
                            ('sensor.camera.depth', 5)]:
            with self.subTest(kind=kind), self.assertRaisesRegex(SensorConfigurationInvalid, 'Too many'):
                sensors = cameras() + [{'id': str(i), 'type': kind} for i in range(count)]
                AgentWrapper.validate_sensor_configuration(sensors, Track.SENSORS, 'SENSORS', True)
        with self.assertRaisesRegex(SensorConfigurationInvalid, 'Illegal sensor'):
            AgentWrapper.validate_sensor_configuration(cameras() + [{'id': 'map', 'type': 'sensor.opendrive_map'}],
                                                       Track.SENSORS, 'SENSORS', True)

    def test_extrinsic_limit_not_relaxed(self):
        sensors = cameras()
        sensors[0]['x'] = 11.
        with self.assertRaisesRegex(SensorConfigurationInvalid, 'extrinsics'):
            AgentWrapper.validate_sensor_configuration(sensors, Track.SENSORS, 'SENSORS', True)

    def test_sensor_tick_applied_and_callback_failure_keeps_actor_tracked(self):
        sensor = Mock()
        sensor.listen.side_effect = RuntimeError('callback registration failed')
        world = Mock()
        world.spawn_actor.return_value = sensor
        agent = Mock()
        agent.sensors.return_value = cameras()[:1]
        wrapper = AgentWrapper(agent)
        with patch.object(CarlaDataProvider, 'get_world', return_value=world):
            with self.assertRaisesRegex(RuntimeError, 'callback registration failed'):
                wrapper.setup_sensors(Mock())
            self.assertEqual(wrapper._sensors_list, [sensor])
            wrapper.cleanup()
        world.get_blueprint_library.return_value.find.return_value.set_attribute.assert_any_call('sensor_tick', '0.05')
        sensor.stop.assert_called_once_with()
        sensor.destroy.assert_called_once_with()
        world.tick.assert_not_called()


class ScenarioCleanupTests(unittest.TestCase):
    def manager(self, status=True):
        manager = ScenarioManager.__new__(ScenarioManager)
        manager._watchdog = Mock()
        manager._watchdog.get_status.return_value = status
        manager._agent = Mock()
        manager.scenario = Mock()
        manager.analyze_scenario = Mock()
        manager.start_system_time = 1.
        manager.start_game_time = 0.
        manager._running = True
        return manager

    def test_normal_stop_keeps_termination_cleanup_analysis_order(self):
        manager = self.manager()
        calls = Mock()
        calls.attach_mock(manager.scenario.terminate, 'terminate')
        calls.attach_mock(manager._agent.cleanup, 'cleanup')
        calls.attach_mock(manager.analyze_scenario, 'analyze')
        manager.stop_scenario()
        self.assertEqual([call[0] for call in calls.mock_calls], ['terminate', 'cleanup', 'analyze'])
        self.assertIsNone(manager._agent)

    def test_watchdog_failure_still_cleans_without_analyzing_as_success(self):
        manager = self.manager(False)
        agent = manager._agent
        manager.stop_scenario()
        agent.cleanup.assert_called_once_with()
        manager.scenario.terminate.assert_not_called()
        manager.analyze_scenario.assert_not_called()
        self.assertFalse(manager.get_running_status())
        self.assertIsNotNone(manager.end_system_time)

    def test_termination_error_is_preserved_and_sensors_still_cleaned(self):
        manager = self.manager()
        agent = manager._agent
        error = RuntimeError('scenario termination failure')
        manager.scenario.terminate.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            manager.stop_scenario()
        self.assertIs(raised.exception, error)
        agent.cleanup.assert_called_once_with()
        manager.analyze_scenario.assert_not_called()

    def test_evaluator_cleanup_after_setup_failure_and_repeated_cleanup(self):
        manager = self.manager(False)
        agent = manager._agent
        with patch.object(CarlaDataProvider, 'get_world') as world:
            manager.cleanup()
            manager.cleanup()
        agent.cleanup.assert_called_once_with()
        world.assert_not_called()
        self.assertFalse(manager.get_running_status())

    def test_cleanup_exception_is_reported_without_replacing_watchdog_failure(self):
        manager = self.manager(False)
        manager._agent.cleanup.side_effect = RuntimeError('sensor cleanup failure')
        with self.assertLogs(level='WARNING') as logs:
            manager.cleanup()
        self.assertIn('sensor cleanup failure', logs.output[0])
        self.assertFalse(manager.get_running_status())
        self.assertIsNone(manager._agent)


class MetadataTests(unittest.TestCase):
    def test_research_metadata_leaves_scores_and_route_records_identical(self):
        original = {'_checkpoint': {'records': [{'scores': {'score_composed': 12.345}, 'status': 'Failed'}]},
                    'values': ['12.345'], 'entry_status': 'Finished with agent errors'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps(original))
            save_research_metadata(str(path), cameras(), {'route_id': '0', 'sensors_effective': []})
            result = json.loads(path.read_text())
        metadata = result.pop('evaluation_metadata')
        self.assertEqual(result, original)
        self.assertFalse(metadata['official_sensor_rules_compliant'])
        self.assertEqual(metadata['sensors_requested'], cameras())


if __name__ == '__main__':
    unittest.main()
