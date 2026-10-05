import numpy as np

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap, get_random_subarray


def draws(object_map: ObjectPointCloudMap) -> np.ndarray:
    points = np.arange(300).reshape(100, 3)
    return get_random_subarray(points, 10, object_map._rng)


def test_seeding_restarts_the_stream() -> None:
    object_map = ObjectPointCloudMap(erosion_size=5)
    object_map.seed([123, 0])
    first = draws(object_map)
    draws(object_map)  # consume more
    object_map.seed([123, 0])
    assert np.array_equal(draws(object_map), first)


def test_robots_have_different_streams() -> None:
    r1, r2 = ObjectPointCloudMap(erosion_size=5), ObjectPointCloudMap(erosion_size=5)
    r1.seed([123, 0])
    r2.seed([123, 1])
    assert not np.array_equal(draws(r1), draws(r2))


def test_global_random_state_does_not_affect_the_map() -> None:
    object_map = ObjectPointCloudMap(erosion_size=5)
    object_map.seed([123, 0])
    first = draws(object_map)
    object_map.seed([123, 0])
    np.random.randint(0, 256, size=50)  # e.g. box colors drawn while rendering a video
    np.random.seed(7)
    assert np.array_equal(draws(object_map), first)


def test_reset_keeps_the_stream() -> None:
    # The policy seeds the map when the episode starts; the map is reset afterwards,
    # on the robot's first step, and that must not undo the seeding
    object_map = ObjectPointCloudMap(erosion_size=5)
    object_map.seed([123, 0])
    first = draws(object_map)
    object_map.seed([123, 0])
    object_map.reset()
    assert np.array_equal(draws(object_map), first)
