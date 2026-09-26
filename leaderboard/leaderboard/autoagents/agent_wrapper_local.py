#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os
import time

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.ros_base_agent import ROSBaseAgent

DATAGEN = int(os.environ.get('DATAGEN', 0))

DATASET_COLLECTION_ENV_VAR = 'CARLA_RAW_COLLECTION'
COLLECTION_SENSOR_ATTRIBUTES = {
    'sensor.camera': {
        'width': 'image_size_x', 'height': 'image_size_y', 'fov': 'fov',
        'sensor_tick': 'sensor_tick',
    },
    'sensor.lidar.ray_cast': {
        'range': 'range', 'rotation_frequency': 'rotation_frequency',
        'channels': 'channels', 'upper_fov': 'upper_fov', 'lower_fov': 'lower_fov',
        'horizontal_fov': 'horizontal_fov', 'points_per_second': 'points_per_second',
        'sensor_tick': 'sensor_tick',
    },
    'sensor.other.radar': {
        'range': 'range', 'horizontal_fov': 'horizontal_fov',
        'vertical_fov': 'vertical_fov', 'points_per_second': 'points_per_second',
        'sensor_tick': 'sensor_tick',
    },
    'sensor.other.gnss': {'sensor_tick': 'sensor_tick'},
    'sensor.other.imu': {'sensor_tick': 'sensor_tick'},
}


def _is_dataset_collection():
    """Read the mode at use time so import order cannot freeze its value."""
    return os.environ.get(DATASET_COLLECTION_ENV_VAR, '0') == '1'


def _collection_attribute_map(sensor_type):
    if sensor_type.startswith('sensor.camera'):
        return COLLECTION_SENSOR_ATTRIBUTES['sensor.camera']
    return COLLECTION_SENSOR_ATTRIBUTES.get(sensor_type, {})


def _requested_collection_attributes(sensor_spec):
    mapping = _collection_attribute_map(sensor_spec['type'])
    return {target: str(sensor_spec[source]) for source, target in mapping.items()
            if source in sensor_spec}


def _carla_attribute_value(attribute):
    """Return a JSON-safe ActorAttribute value instead of its debug repr."""
    attribute_type = getattr(attribute, 'type', None)
    carla_attribute_type = getattr(carla, 'ActorAttributeType', None)
    converters = []
    if carla_attribute_type is not None:
        converters = [
            (getattr(carla_attribute_type, 'Bool', None), 'as_bool'),
            (getattr(carla_attribute_type, 'Int', None), 'as_int'),
            (getattr(carla_attribute_type, 'Float', None), 'as_float'),
            (getattr(carla_attribute_type, 'String', None), 'as_str'),
        ]
    for expected_type, method_name in converters:
        if expected_type is not None and attribute_type == expected_type:
            return getattr(attribute, method_name)()

    try:
        return attribute.as_str()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        value = getattr(attribute, 'value', None)
        if value is not None:
            return value
        raise SensorConfigurationInvalid(
            "Could not read CARLA ActorAttribute value: {!r}".format(attribute))


def _attribute_values_match(requested, effective):
    if isinstance(effective, bool):
        if isinstance(requested, bool):
            return requested is effective
        if isinstance(requested, str):
            normalized = requested.strip().lower()
            if normalized in ('true', '1'):
                return effective is True
            if normalized in ('false', '0'):
                return effective is False
        return False
    if isinstance(effective, int):
        try:
            return int(requested) == effective
        except (TypeError, ValueError):
            return False
    if isinstance(effective, float):
        try:
            return math.isclose(float(requested), effective, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if requested == effective:
        return True
    try:
        return math.isclose(float(requested), float(effective), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False

MAX_ALLOWED_RADIUS_SENSOR = 70.0
QUALIFIER_SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1
}
SENSORS_LIMITS = {
    'sensor.camera.rgb': 8,
    'sensor.lidar.ray_cast': 2,
    'sensor.other.radar': 4,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1,
    'sensor.camera.depth': 8, # for data generation
    'sensor.camera.semantic_segmentation': 8 # for data generation
}
ALLOWED_SENSORS = SENSORS_LIMITS.keys()


class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)


class AgentWrapperFactory(object):

    @staticmethod
    def get_wrapper(agent):
        if isinstance(agent, ROSBaseAgent):
            return ROSAgentWrapper(agent)
        else:
            return AgentWrapper(agent)


def validate_sensor_configuration(sensors, agent_track, selected_track):
    """
    Ensure that the sensor configuration is valid, in case the challenge mode is used
    Returns true on valid configuration, false otherwise
    """
    if Track(selected_track) != agent_track:
        raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

    sensor_count = {}
    sensor_ids = []

    for sensor in sensors:

        # Check if the is has been already used
        sensor_id = sensor['id']
        if sensor_id in sensor_ids:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
        else:
            sensor_ids.append(sensor_id)

        # Check if the sensor is valid
        if agent_track == Track.SENSORS:
            if sensor['type'].startswith('sensor.opendrive_map'):
                raise SensorConfigurationInvalid("Illegal sensor 'sensor.opendrive_map' used for Track [{}]!".format(agent_track))

        # Check the sensors validity
        collection_only_sensor = (
            _is_dataset_collection()
            and sensor['type'] == 'sensor.camera.instance_segmentation'
        )
        if sensor['type'] not in ALLOWED_SENSORS and not collection_only_sensor:
            raise SensorConfigurationInvalid("Illegal sensor '{}' used for Track [{}]!".format(sensor['type'], agent_track))

        # Check the extrinsics of the sensor
        if 'x' in sensor and 'y' in sensor and 'z' in sensor:
            if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                raise SensorConfigurationInvalid(
                    "Illegal sensor extrinsics used for sensor '{}'. Max allowed radius is {}m".format(sensor['id'], MAX_ALLOWED_RADIUS_SENSOR))

        # Check the amount of sensors
        if sensor['type'] in sensor_count:
            sensor_count[sensor['type']] += 1
        else:
            sensor_count[sensor['type']] = 1

    if agent_track in (Track.SENSORS_QUALIFIER, Track.MAP_QUALIFIER):
        sensor_limits = QUALIFIER_SENSORS_LIMITS
    else:
        sensor_limits = SENSORS_LIMITS

    for sensor_type, max_instances_allowed in sensor_limits.items():
        if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
            raise SensorConfigurationInvalid(
                "Too many {} used! "
                "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                              max_instances_allowed,
                                                                              sensor_count[sensor_type]))


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """
    _agent = None
    _sensors_list = []
    sensor_list_names = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent
        self._sensors_list = []
        self.sensor_list_names = []
        self._cleanup_done = False

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent(self.sensor_list_names)
        # return self._agent()

    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_.startswith('sensor.camera'):
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)
            if DATAGEN==1:
                attributes['rotation_frequency'] = str(sensor_spec['rotation_frequency'])
                attributes['points_per_second'] = str(sensor_spec['points_per_second'])
            else:
                attributes['rotation_frequency'] = str(10)
                attributes['points_per_second'] = str(600000)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'  # meters

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            if DATAGEN==0:
                attributes['noise_alt_stddev'] = str(0.000005)
                attributes['noise_lat_stddev'] = str(0.000005)
                attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        if _is_dataset_collection():
            requested_attributes = _requested_collection_attributes(sensor_spec)
            if type_ == 'sensor.lidar.ray_cast':
                missing = set(COLLECTION_SENSOR_ATTRIBUTES[type_].values()) - set(requested_attributes)
                if missing:
                    raise SensorConfigurationInvalid(
                        "Dataset collection LiDAR spec is missing CARLA attributes: {}".format(
                            ', '.join(sorted(missing))))
            attributes.update(requested_attributes)
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()
        effective_configs = []
        for sensor_spec in self._agent.sensors():
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            # These are the pseudosensors (not spawned)
            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    if not bp.has_attribute(str(key)):
                        raise SensorConfigurationInvalid(
                            "CARLA blueprint {} does not support requested attribute {}".format(type_, key))
                    bp.set_attribute(str(key), str(value))

                if _is_dataset_collection():
                    requested_attributes = _requested_collection_attributes(sensor_spec)
                    effective_attributes = {
                        key: _carla_attribute_value(bp.get_attribute(str(key))) for key in attributes
                    }
                    mismatches = {
                        key: (requested, effective_attributes.get(key))
                        for key, requested in requested_attributes.items()
                        if not _attribute_values_match(requested, effective_attributes.get(key))
                    }
                    if mismatches:
                        raise SensorConfigurationInvalid(
                            "Dataset collection sensor {} effective attributes differ from requested: {}".format(
                                id_, mismatches))
                    effective_configs.append({
                        'id': id_, 'type': type_,
                        'requested_carla_attributes': requested_attributes,
                        'effective_carla_attributes': effective_attributes,
                    })
                    if type_ == 'sensor.lidar.ray_cast':
                        ordered_keys = COLLECTION_SENSOR_ATTRIBUTES[type_].values()
                        print('[LIDAR effective config] {}'.format(' '.join(
                            '{}={}'.format(key, effective_attributes[key]) for key in ordered_keys)))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)

            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)
            self.sensor_list_names.append([sensor_spec['id'], sensor])

        if effective_configs and hasattr(self._agent, 'record_effective_sensor_configs'):
            self._agent.record_effective_sensor_configs(effective_configs)

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        for _ in range(10):
            world.tick()

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        if self._cleanup_done:
            return
        self._cleanup_done = True

        sensors = list(self._sensors_list)

        for sensor in sensors:
            if sensor is None:
                continue
            try:
                sensor.stop()
            except Exception as e:
                print("WARNING: failed to stop sensor {}: {}".format(getattr(sensor, 'id', sensor), e))

        time.sleep(0.05)
        world = CarlaDataProvider.get_world()
        if world is not None:
            try:
                world.tick()
            except Exception as e:
                print("WARNING: failed to tick world after stopping sensors: {}".format(e))

        for sensor in sensors:
            if sensor is None:
                continue
            try:
                if hasattr(sensor, 'is_alive') and not sensor.is_alive:
                    continue
                sensor.destroy()
            except Exception as e:
                print("WARNING: failed to destroy sensor {}: {}".format(getattr(sensor, 'id', sensor), e))

        self._sensors_list.clear()
        self.sensor_list_names.clear()

        # Tick once to destroy the sensors
        if world is not None:
            try:
                world.tick()
            except Exception as e:
                print("WARNING: failed to tick world after destroying sensors: {}".format(e))


class ROSAgentWrapper(AgentWrapper):

    SENSOR_TYPE_REMAPS = {
        "sensor.opendrive_map": "sensor.pseudo.opendrive_map",
        "sensor.speedometer": "sensor.pseudo.speedometer"
    }

    def __init__(self, agent):
        super(ROSAgentWrapper, self).__init__(agent)

    def _preprocess_sensor_spec(self, sensor_spec):
        type_, id_, sensor_transform, attributes = super(ROSAgentWrapper, self)._preprocess_sensor_spec(sensor_spec)
        new_type = self.SENSOR_TYPE_REMAPS.get(type_, type_)
        return new_type, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        for sensor_spec in self._agent.sensors():
            type_, id_, transform, attributes = self._preprocess_sensor_spec(sensor_spec)
            uid = self._agent.spawn_object(type_, id_, transform, attributes, attach_to=vehicle.id)
            self._sensors_list.append(uid)

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()

    def cleanup(self):
        if self._cleanup_done:
            return
        self._cleanup_done = True

        for uid in self._sensors_list:
            try:
                self._agent.destroy_object(uid)
            except Exception as e:
                print("WARNING: failed to destroy ROS sensor {}: {}".format(uid, e))
        self._sensors_list.clear()

        # Tick once to destroy the sensors
        world = CarlaDataProvider.get_world()
        if world is not None:
            try:
                world.tick()
            except Exception as e:
                print("WARNING: failed to tick world after destroying ROS sensors: {}".format(e))
        self._sensors_list.clear()
        self.sensor_list_names.clear()
