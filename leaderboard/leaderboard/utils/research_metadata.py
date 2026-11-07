"""Attach research provenance without changing route records or score values."""
from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict
from leaderboard.utils.ego_vehicle import ego_vehicle_blueprint


def save_research_metadata(endpoint, sensors, agent_metadata=None):
    data = fetch_dict(endpoint)
    metadata = dict(agent_metadata or {})
    metadata.update(
        evaluation_mode='research_vadv2_six_camera',
        official_sensor_rules_compliant=False,
        official_rgb_camera_limit=4,
        research_rgb_camera_limit=6,
        sensors_requested=sensors,
        ego_vehicle=ego_vehicle_blueprint(),
    )
    data['evaluation_metadata'] = metadata
    save_dict(endpoint, data)
