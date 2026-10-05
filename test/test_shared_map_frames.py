"""Two robots that start apart and look at the same wall must put it in the same place
of the shared obstacle map.

The chain under test is the real one: Habitat's GPS and compass sensors (relative to
the episode start, which is R1's start for both robots), then VLFM's
_cache_observations and ObstacleMap.update_map. Only the depth image is synthetic: it is
ray cast against a wall defined in Habitat's world frame.
"""

from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import quaternion
import torch
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.habitat_policies import HabitatITMPolicyV2

# Camera of Habitat's objectnav_hm3d benchmark, as used in the experiments
WIDTH, HEIGHT, HFOV = 640, 480, 79.0
MIN_DEPTH, MAX_DEPTH, CAMERA_HEIGHT = 0.5, 5.0, 0.88
FX = FY = WIDTH / (2 * np.tan(np.deg2rad(HFOV) / 2))

FLOOR_Y = 0.1
R1_START = np.array([2.0, FLOOR_Y, -1.0])
R1_YAW = np.deg2rad(30.0)  # rotation about Habitat's up axis (+y), counterclockwise


def yaw_quaternion(yaw: float) -> quaternion.quaternion:
    return quaternion.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)


def forward(yaw: float) -> np.ndarray:
    """Habitat agents look along -z."""
    return np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])


def left(yaw: float) -> np.ndarray:
    return np.array([-np.cos(yaw), 0.0, np.sin(yaw)])


def shared_xy_to_world(x: float, y: float) -> np.ndarray:
    """A point given in W (R1's start frame: x forward, y left) in Habitat's world."""
    return R1_START + x * forward(R1_YAW) + y * left(R1_YAW)


# A 3 m wide, 2.5 m high wall, 3 m in front of R1's start: x = 3, -1.5 <= y <= 1.5 in W
WALL_CENTER = shared_xy_to_world(3.0, 0.0)
WALL_ALONG = left(R1_YAW)
WALL_HALF_WIDTH, WALL_HEIGHT = 1.5, 2.5


def render_depth(position: np.ndarray, yaw: float) -> np.ndarray:
    """Normalized depth image of the wall, as Habitat's depth sensor returns it."""
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    # Ray directions in the camera frame (x right, y up, looking along -z), scaled so
    # that the distance along a ray equals the depth along the optical axis
    rays = np.stack([(u - WIDTH / 2) / FX, -(v - HEIGHT / 2) / FY, -np.ones_like(u)], axis=-1)
    rotation = quaternion.as_rotation_matrix(yaw_quaternion(yaw))
    rays_world = rays @ rotation.T
    origin = position + np.array([0.0, CAMERA_HEIGHT, 0.0])

    normal = np.cross(WALL_ALONG, [0.0, 1.0, 0.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = ((WALL_CENTER - origin) @ normal) / (rays_world @ normal)
    hits = origin + depth[..., None] * rays_world
    along = (hits - WALL_CENTER) @ WALL_ALONG
    on_wall = (
        (depth > 0) & (np.abs(along) <= WALL_HALF_WIDTH) & (hits[..., 1] >= FLOOR_Y) & (hits[..., 1] <= FLOOR_Y + WALL_HEIGHT)
    )
    depth = np.where(on_wall & (depth >= MIN_DEPTH) & (depth < MAX_DEPTH), depth, MAX_DEPTH)
    return ((depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)).astype(np.float32)[..., None]


def habitat_gps_compass(
    position: np.ndarray, yaw: float, start_position: np.ndarray, start_yaw: float
) -> Tuple[np.ndarray, float]:
    """GPS and compass from Habitat's own sensors, for an episode starting at the
    given pose."""
    sim = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=position, rotation=yaw_quaternion(yaw)))
    start_rotation = yaw_quaternion(start_yaw)
    episode = SimpleNamespace(
        start_position=start_position.tolist(),
        start_rotation=[start_rotation.x, start_rotation.y, start_rotation.z, start_rotation.w],
    )
    gps_sensor = object.__new__(EpisodicGPSSensor)
    gps_sensor._sim, gps_sensor._dimensionality = sim, 2
    compass_sensor = object.__new__(EpisodicCompassSensor)
    compass_sensor._sim = sim
    gps = gps_sensor.get_observation(None, episode)
    compass = compass_sensor.get_observation(None, episode)
    return np.asarray(gps, dtype=np.float32), float(np.asarray(compass).reshape(-1)[0])


def new_obstacle_map() -> ObstacleMap:
    # VLFMConfig's values
    obstacle_map = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18, area_thresh=1.5)
    obstacle_map.reset()
    return obstacle_map


def observe(
    obstacle_map: ObstacleMap, position: np.ndarray, yaw: float, start_position: np.ndarray, start_yaw: float
) -> None:
    """One step of VLFM's Habitat policy: cache the observations, which updates the
    obstacle map."""
    policy = object.__new__(HabitatITMPolicyV2)  # skips the model clients
    policy._observations_cache = {}
    policy._compute_frontiers = True
    policy._obstacle_map = obstacle_map
    policy._robot_id = 0
    policy._camera_height = CAMERA_HEIGHT
    policy._min_depth, policy._max_depth = MIN_DEPTH, MAX_DEPTH
    policy._fx = policy._fy = FX
    policy._camera_fov = np.deg2rad(HFOV)

    gps, compass = habitat_gps_compass(position, yaw, start_position, start_yaw)
    observations = {
        "rgb": torch.zeros((1, HEIGHT, WIDTH, 3), dtype=torch.uint8),
        "depth": torch.from_numpy(render_depth(position, yaw))[None],
        "gps": torch.from_numpy(gps)[None],
        "compass": torch.tensor([[compass]]),
        "heading": torch.tensor([[0.0]]),
    }
    policy._cache_observations(observations)


def wall_pixels(obstacle_map: ObstacleMap) -> np.ndarray:
    return obstacle_map._map.copy()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


# R2 starts 1 m ahead and 2.5 m to the left of R1's start, turned to face the wall
R2_START = shared_xy_to_world(1.0, 2.5)
R2_YAW = R1_YAW + np.arctan2(0.0 - 2.5, 3.0 - 1.0)


def map_of(robots: List[Tuple[np.ndarray, float, np.ndarray, float]]) -> np.ndarray:
    obstacle_map = new_obstacle_map()
    for robot in robots:
        observe(obstacle_map, *robot)
    return wall_pixels(obstacle_map)


def test_r1_sees_the_wall_where_it_is() -> None:
    r1 = map_of([(R1_START, R1_YAW, R1_START, R1_YAW)])
    rows, cols = np.nonzero(r1)
    # x = 3 m -> row 500 + 3 * 20; y in [-1.5, 1.5] m -> columns 500 -+ 30
    assert set(rows.tolist()) == {560}
    assert cols.min() >= 470 and cols.max() <= 530 and len(cols) >= 55


def test_both_robots_put_the_wall_in_the_same_pixels() -> None:
    r1 = map_of([(R1_START, R1_YAW, R1_START, R1_YAW)])
    # R2's sensors are relative to the episode start, which is R1's start
    r2 = map_of([(R2_START, R2_YAW, R1_START, R1_YAW)])
    assert r2.sum() > 0
    assert iou(r1, r2) > 0.9

    # Both robots writing into one shared map: no second copy of the wall
    shared = map_of([(R1_START, R1_YAW, R1_START, R1_YAW), (R2_START, R2_YAW, R1_START, R1_YAW)])
    assert iou(shared, r1) > 0.9


def test_a_robot_using_its_own_start_as_origin_would_misplace_the_wall() -> None:
    # The mistake the test above guards against: R2's GPS relative to R2's own start
    r1 = map_of([(R1_START, R1_YAW, R1_START, R1_YAW)])
    r2_wrong = map_of([(R2_START, R2_YAW, R2_START, R2_YAW)])
    assert r2_wrong.sum() > 0
    assert iou(r1, r2_wrong) < 0.1
