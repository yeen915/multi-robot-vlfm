from typing import Any, Dict

import numpy as np
import pytest

from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.utils.hw_corruption import is_corrupted_index_error


def index_error(index: int, size: int) -> IndexError:
    try:
        np.zeros(size)[index]
    except IndexError as e:
        return e
    raise AssertionError("no IndexError")


def test_stray_high_bit_is_corruption() -> None:
    # The indices seen on the evaluation machine
    for index, size in [(2**53 + 128, 480), (2**51 + 935, 1000), (2**51 + 1131, 640), (2**55 + 851, 1000)]:
        assert is_corrupted_index_error(index_error(index, size))


def test_running_off_the_map_is_not_corruption() -> None:
    assert not is_corrupted_index_error(index_error(1000, 1000))
    assert not is_corrupted_index_error(index_error(1400, 1000))
    assert not is_corrupted_index_error(index_error(-1001, 1000))
    assert not is_corrupted_index_error(IndexError("list index out of range"))


class _Policy(BaseObjectNavPolicy):
    """Skips __init__ (which loads models); only _pre_step is exercised."""

    def __init__(self, error: IndexError) -> None:
        self._error = error
        self._did_reset = True
        self._hw_corruption = False
        self._policy_info: Dict[str, Any] = {"stop_called": False}

    def _cache_observations(self, observations: Any) -> None:
        raise self._error


def test_pre_step_flags_corruption_and_stops() -> None:
    policy = _Policy(index_error(2**51 + 1131, 640))
    with pytest.raises(StopIteration):
        policy._pre_step({}, np.ones((1, 1)))
    assert policy._hw_corruption
    assert policy._policy_info == {"stop_called": False, "hw_corruption": True}


def test_pre_step_keeps_map_edge_behaviour() -> None:
    policy = _Policy(index_error(1000, 1000))
    with pytest.raises(StopIteration):
        policy._pre_step({}, np.ones((1, 1)))
    assert not policy._hw_corruption
    assert "hw_corruption" not in policy._policy_info
