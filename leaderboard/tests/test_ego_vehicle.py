"""Tests for the configurable LB1 ego vehicle (no CARLA server needed)."""

import os
import unittest
from unittest.mock import patch

from leaderboard.utils.ego_vehicle import DEFAULT_EGO_VEHICLE, EGO_VEHICLE_ENV, ego_vehicle_blueprint


class EgoVehicleTest(unittest.TestCase):

    def test_default_matches_training_vehicle(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(EGO_VEHICLE_ENV, None)
            self.assertEqual(ego_vehicle_blueprint(), 'vehicle.nissan.micra')
            self.assertEqual(DEFAULT_EGO_VEHICLE, 'vehicle.nissan.micra')

    def test_empty_value_falls_back_to_default(self):
        with patch.dict(os.environ, {EGO_VEHICLE_ENV: '  '}):
            self.assertEqual(ego_vehicle_blueprint(), DEFAULT_EGO_VEHICLE)

    def test_override(self):
        with patch.dict(os.environ, {EGO_VEHICLE_ENV: 'vehicle.lincoln.mkz_2017'}):
            self.assertEqual(ego_vehicle_blueprint(), 'vehicle.lincoln.mkz_2017')

    def test_rejects_non_vehicle_blueprint(self):
        with patch.dict(os.environ, {EGO_VEHICLE_ENV: 'walker.pedestrian.0001'}):
            with self.assertRaises(ValueError):
                ego_vehicle_blueprint()

    def test_route_scenarios_use_configured_vehicle(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ('route_scenario_local.py', 'route_scenario.py'):
            path = os.path.join(here, '..', 'leaderboard', 'scenarios', name)
            with open(path) as handle:
                source = handle.read()
            self.assertNotIn("'vehicle.lincoln.mkz_2017'", source, name)
            self.assertEqual(source.count('ego_vehicle_blueprint()'), 2, name)


if __name__ == '__main__':
    unittest.main()
