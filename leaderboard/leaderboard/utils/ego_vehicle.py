"""Ego vehicle blueprint used by the LB1 route scenarios.

The research workspace collects its training data with ``vehicle.nissan.micra``
(the carla_garage ``evaluation`` branch autopilot) and mounts its cameras relative
to that body, so closed-loop evaluation spawns the same vehicle by default.
Set ``LB1_EGO_VEHICLE`` to another blueprint id (for example the official LB1
``vehicle.lincoln.mkz_2017``) to override it.
"""

import os

DEFAULT_EGO_VEHICLE = 'vehicle.nissan.micra'
EGO_VEHICLE_ENV = 'LB1_EGO_VEHICLE'


def ego_vehicle_blueprint():
    """Return the blueprint id for the hero vehicle."""
    value = os.environ.get(EGO_VEHICLE_ENV, '').strip()
    if not value:
        return DEFAULT_EGO_VEHICLE
    if not value.startswith('vehicle.'):
        raise ValueError('{} must be a vehicle blueprint id, got {!r}'.format(EGO_VEHICLE_ENV, value))
    return value
