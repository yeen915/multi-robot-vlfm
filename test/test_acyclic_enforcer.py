# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer


def _visit(enforcer: AcyclicEnforcer) -> bool:
    position, frontier, values = np.array([1.0, 2.0]), np.array([3.0, 4.0]), (0.2, 0.1)
    enforcer.add_state_action(position, frontier, values)
    return enforcer.check_cyclic(position.copy(), frontier.copy(), values)


def test_default_never_detects_cycles() -> None:
    # Matches the published VLFM behavior.
    assert not _visit(AcyclicEnforcer())


def test_fix_detects_repeated_state_action() -> None:
    enforcer = AcyclicEnforcer(fix=True)
    assert _visit(enforcer)
    assert not enforcer.check_cyclic(np.array([1.0, 2.0]), np.array([9.0, 9.0]), (0.2, 0.1))


def test_history_is_per_instance() -> None:
    first = AcyclicEnforcer(fix=True)
    _visit(first)
    assert len(AcyclicEnforcer(fix=True).history) == 0
