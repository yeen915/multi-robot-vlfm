# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Set

import numpy as np


class StateAction:
    def __init__(self, position: np.ndarray, action: Any, other: Any = None):
        self.position = position
        self.action = action
        self.other = other

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.action}_{self.other}"
        return hash(string_repr)


class AcyclicEnforcer:
    def __init__(self, fix: bool = False) -> None:
        """
        Args:
            fix: StateAction defines __hash__ but not __eq__, so by default a lookup
                in self.history never matches and no cycle is ever detected. That is
                the behavior of the published VLFM, so it is kept as the default.
                Set fix=True to compare state-actions by value instead.
        """
        self.history: Set[Any] = set()
        self._fix = fix

    def _key(self, position: np.ndarray, action: Any, other: Any) -> Any:
        state_action = StateAction(position, action, other)
        return hash(state_action) if self._fix else state_action

    def check_cyclic(self, position: np.ndarray, action: Any, other: Any = None) -> bool:
        cyclic = self._key(position, action, other) in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: Any, other: Any = None) -> None:
        self.history.add(self._key(position, action, other))
