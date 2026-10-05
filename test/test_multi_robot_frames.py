import numpy as np
import pytest

from vlfm.multi_robot.frames import shared_to_world, world_to_shared

IDENTITY = [0.0, 0.0, 0.0, 1.0]


def yaw_quat(yaw: float) -> list:
    """Rotation by yaw about the simulator's up (y) axis, as [x, y, z, w]."""
    return [0.0, float(np.sin(yaw / 2)), 0.0, float(np.cos(yaw / 2))]


def test_start_pose_is_the_origin() -> None:
    start = np.array([3.0, 0.5, -2.0])
    xyz, yaw = world_to_shared(start, yaw_quat(1.0), start, yaw_quat(1.0))
    assert xyz == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_axes_with_identity_start() -> None:
    origin = np.zeros(3)
    # An agent with identity rotation faces -z, so -z is forward (x in W)
    assert world_to_shared(np.array([0.0, 0.0, -2.0]), IDENTITY, origin, IDENTITY)[0] == pytest.approx([2.0, 0.0, 0.0])
    # +x in the simulator is to the agent's right, so -x is left (y in W)
    assert world_to_shared(np.array([-2.0, 0.0, 0.0]), IDENTITY, origin, IDENTITY)[0] == pytest.approx([0.0, 2.0, 0.0])
    # The simulator's y is up (z in W)
    assert world_to_shared(np.array([0.0, 2.0, 0.0]), IDENTITY, origin, IDENTITY)[0] == pytest.approx([0.0, 0.0, 2.0])


def test_left_turn_is_positive_yaw() -> None:
    _, yaw = world_to_shared(np.zeros(3), yaw_quat(np.pi / 6), np.zeros(3), IDENTITY)
    assert yaw == pytest.approx(np.pi / 6)
    # Yaw is relative to the start heading
    _, yaw = world_to_shared(np.zeros(3), yaw_quat(np.pi / 2), np.zeros(3), yaw_quat(np.pi / 3))
    assert yaw == pytest.approx(np.pi / 6)


def test_start_rotation_rotates_the_axes() -> None:
    # The episode starts facing the simulator's -x (a 90 degree left turn from -z)
    start_rotation = yaw_quat(np.pi / 2)
    xyz, _ = world_to_shared(np.array([-2.0, 0.0, 0.0]), IDENTITY, np.zeros(3), start_rotation)
    assert xyz == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)  # straight ahead


def test_round_trip() -> None:
    rng = np.random.default_rng(0)
    start_position, start_rotation = rng.normal(size=3), yaw_quat(0.7)
    for position in rng.normal(size=(5, 3)):
        xyz, _ = world_to_shared(position, IDENTITY, start_position, start_rotation)
        assert shared_to_world(xyz, start_position, start_rotation)[0] == pytest.approx(position, abs=1e-9)
