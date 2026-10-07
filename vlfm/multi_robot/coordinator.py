"""Assigns frontiers to robots that share their maps (design doc 3.4).

Utility of frontier f for robot i, with fixed scales (no per-step min-max):

    U(i, f) = V(f) / s_V - lambda * A(i, f) / s_A + sum_k w_k * T_k(i, f)

V is the value map's value at f (as VLFM computes it), A the path cost in actions
(path_cost.py). The T_k are extra terms registered in UTILITY_TERMS; with weight 0 they
are not evaluated.

Assigners:
    independent_utility: each robot takes its own best frontier by U; no spacing.
    greedy: each robot takes its nearest frontier (by A, ignoring values) that is at
        least d_min from its teammates' goals.
    joint: the pair of frontiers with the highest U(1, f1) + U(2, f2) such that
        |f1 - f2| >= d_min (straight-line distance).

All three keep a robot's current frontier unless the new choice scores more than tau
higher (for joint: the pair's total), so that the robots do not switch back and forth.
The same tau is used by all three, so independent_utility and joint differ only in
deciding jointly.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from vlfm.multi_robot.path_cost import PathPlanner

ASSIGNERS = ("independent_utility", "greedy", "joint")
# A frontier within this distance of the previous goal is the same frontier (as in
# VLFM's rule for sticking to its last frontier)
SAME_FRONTIER_DISTANCE = 0.5  # meters


@dataclass
class CoordinatorConfig:
    assigner: str = "joint"
    lambda_path: float = 1.0
    d_min: float = 2.0  # meters, straight-line distance between two robots' frontiers
    hysteresis_tau: float = 0.1
    value_scale: float = 0.0070  # s_V
    path_scale: float = 11.8  # s_A, in actions
    unreachable_penalty: float = 3.0
    term_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class TermContext:
    """What an extra utility term may look at."""

    robot: int
    poses: Dict[int, Tuple[np.ndarray, float]]
    obstacle_map: Any


def _not_implemented(name: str) -> Callable[..., float]:
    def term(frontier: np.ndarray, context: TermContext) -> float:
        raise NotImplementedError(f"Utility term {name!r} is planned for phase 6")

    return term


# Extra utility terms: name -> function(frontier_xy, context) -> value. Reserved for
# phase 6; they are only evaluated when given a non-zero weight.
UTILITY_TERMS: Dict[str, Callable[[np.ndarray, TermContext], float]] = {
    "info_gain": _not_implemented("info_gain"),  # non-overlapping information gain
    "trajectory_reuse": _not_implemented("trajectory_reuse"),  # penalty for the teammate's path
}


def match_frontier(goal: Optional[np.ndarray], frontiers: np.ndarray) -> Optional[int]:
    """Index of the frontier that is the same as `goal`, if any."""
    if goal is None or len(frontiers) == 0:
        return None
    distances = np.linalg.norm(frontiers - goal, axis=1)
    index = int(np.argmin(distances))
    return index if distances[index] <= SAME_FRONTIER_DISTANCE else None


def far_enough(frontiers: np.ndarray, others: Sequence[np.ndarray], d_min: float) -> np.ndarray:
    """Mask of the frontiers at least d_min from every point of `others`."""
    mask = np.ones(len(frontiers), dtype=bool)
    for other in others:
        mask &= np.linalg.norm(frontiers - other, axis=1) >= d_min
    return mask


class Coordinator:
    def __init__(self, config: CoordinatorConfig) -> None:
        if config.assigner not in ASSIGNERS:
            raise ValueError(f"Unknown assigner {config.assigner!r}")
        for name, weight in config.term_weights.items():
            if name not in UTILITY_TERMS:
                raise ValueError(f"Unknown utility term {name!r}")
        self.config = config
        self.reset()

    def reset(self) -> None:
        # Latest (x, y) and yaw of each robot, in the shared frame
        self.poses: Dict[int, Tuple[np.ndarray, float]] = {}
        # Each robot's current frontier
        self.goals: Dict[int, Optional[np.ndarray]] = {}
        self.stats = {"decisions": 0, "switches": 0, "path_costs": 0, "unreachable_paths": 0}

    # Utilities ----------------------------------------------------------------------

    def utilities(
        self, robots: Sequence[int], frontiers: np.ndarray, values: np.ndarray, planner: PathPlanner
    ) -> Dict[int, np.ndarray]:
        c = self.config
        result = {}
        for robot in robots:
            xy, yaw = self.poses[robot]
            costs = planner.costs(xy, yaw, list(frontiers))
            self.stats["path_costs"] += len(costs)
            self.stats["unreachable_paths"] += sum(not cost.reachable for cost in costs)
            actions = np.array([cost.actions for cost in costs])
            if c.assigner == "greedy":
                utility = -actions / c.path_scale  # nearest first, values ignored
            else:
                utility = values / c.value_scale - c.lambda_path * actions / c.path_scale
            context = TermContext(robot, self.poses, planner.obstacle_map)
            for name, weight in c.term_weights.items():
                if weight != 0:
                    utility = utility + weight * np.array([UTILITY_TERMS[name](f, context) for f in frontiers])
            result[robot] = utility
        return result

    # Assignment ---------------------------------------------------------------------

    def choose(
        self,
        robot: int,
        frontiers: np.ndarray,
        values: np.ndarray,
        obstacle_map: Any,
        exploring: Set[int],
        occupied: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """The frontier `robot` should head for now, or None to turn in place.

        Args:
            robot: The robot that is exploring and asks.
            frontiers: (N, 2) frontiers of the shared obstacle map.
            values: (N,) their values V(f).
            obstacle_map: The shared ObstacleMap, for path costs.
            exploring: The robots exploring now (including `robot`); the others are
                turning in place at the start, or navigating to the target.
            occupied: Goals of robots that are not exploring (e.g. the target the
                navigating robot goes to); exploring robots keep away from them.
        """
        frontiers = np.asarray(frontiers, dtype=float).reshape(-1, 2)
        values = np.asarray(values, dtype=float)
        occupied = {r: g for r, g in (occupied or {}).items() if r not in exploring and g is not None}
        # A teammate whose pose is not known yet cannot be planned for
        exploring = {r for r in exploring if r in self.poses} | {robot}
        self.stats["decisions"] += 1
        if len(frontiers) == 0:
            self.goals = {r: None for r in exploring}
            return None

        planner = PathPlanner(obstacle_map, self.config.unreachable_penalty)
        if self.config.assigner == "joint" and len(exploring) > 1:
            robots = sorted(exploring)
            utility = self.utilities(robots, frontiers, values, planner)
            self._choose_joint(robots, frontiers, utility, list(occupied.values()))
        else:
            utility = self.utilities([robot], frontiers, values, planner)[robot]
            avoid: List[np.ndarray] = list(occupied.values())
            if self.config.assigner != "independent_utility":
                # Keep away from what the teammates head for
                avoid += [g for r, g in self.goals.items() if r != robot and r in exploring and g is not None]
            self._choose_single(robot, frontiers, utility, avoid)
        return self.goals.get(robot)

    def _set_goal(self, robot: int, goal: Optional[np.ndarray]) -> None:
        previous = self.goals.get(robot)
        if previous is not None and (goal is None or np.linalg.norm(goal - previous) > SAME_FRONTIER_DISTANCE):
            self.stats["switches"] += 1
        self.goals[robot] = None if goal is None else np.array(goal)

    def _choose_single(self, robot: int, frontiers: np.ndarray, utility: np.ndarray, avoid: List[np.ndarray]) -> None:
        """One robot's choice with its own hysteresis (independent_utility, greedy, or
        joint while its teammate is not exploring)."""
        allowed = far_enough(frontiers, avoid, self.config.d_min)
        if not allowed.any():
            allowed[:] = True  # nothing is far enough: ignore the spacing
        candidates = np.where(allowed, utility, -np.inf)
        best = int(np.argmax(candidates))
        current = match_frontier(self.goals.get(robot), frontiers)
        if current is not None and allowed[current] and candidates[best] <= utility[current] + self.config.hysteresis_tau:
            best = current
        self._set_goal(robot, frontiers[best])

    def _best_pair(
        self, robots: List[int], frontiers: np.ndarray, utility: Dict[int, np.ndarray], avoid: List[np.ndarray]
    ) -> Dict[int, Optional[int]]:
        """Frontier indices maximizing the total utility of two robots, at least d_min
        apart, with the fallbacks of design doc 3.4."""
        r1, r2 = robots
        allowed = far_enough(frontiers, avoid, self.config.d_min)
        if not allowed.any():
            allowed[:] = True
        u1 = np.where(allowed, utility[r1], -np.inf)
        u2 = np.where(allowed, utility[r2], -np.inf)
        candidates = np.flatnonzero(allowed)
        if len(candidates) == 1:
            # One frontier: the robot that values it more goes; the other turns in place
            only = int(candidates[0])
            return {r1: only, r2: None} if u1[only] >= u2[only] else {r1: None, r2: only}
        distance = np.linalg.norm(frontiers[:, None] - frontiers[None], axis=2)
        total = u1[:, None] + u2[None, :]
        feasible = (distance >= self.config.d_min) & np.isfinite(total)
        if feasible.any():
            i, j = np.unravel_index(np.argmax(np.where(feasible, total, -np.inf)), total.shape)
            return {r1: int(i), r2: int(j)}
        # All frontiers within d_min of each other: the robot with the better option takes
        # it, the other takes its own best of the rest
        if u1.max() >= u2.max():
            first, second, u_first, u_second = r1, r2, u1, u2
        else:
            first, second, u_first, u_second = r2, r1, u2, u1
        taken = int(np.argmax(u_first))
        rest = u_second.copy()
        rest[taken] = -np.inf
        return {first: taken, second: int(np.argmax(rest))}

    def _choose_joint(
        self, robots: List[int], frontiers: np.ndarray, utility: Dict[int, np.ndarray], avoid: List[np.ndarray]
    ) -> None:
        current = {r: match_frontier(self.goals.get(r), frontiers) for r in robots}
        kept = [r for r in robots if current[r] is not None]
        if len(kept) == 1:
            # The other robot's frontier is gone (or it had none): keep this one's and
            # choose for the other, away from it
            keeper = kept[0]
            other = robots[0] if robots[1] == keeper else robots[1]
            self._set_goal(keeper, frontiers[current[keeper]])
            self._choose_single(other, frontiers, utility[other], avoid + [frontiers[current[keeper]]])
            return
        new = self._best_pair(robots, frontiers, utility, avoid)
        if len(kept) == 2:
            new_total = sum(utility[r][i] for r, i in new.items() if i is not None)
            current_total = sum(utility[r][current[r]] for r in robots)
            if new_total <= current_total + self.config.hysteresis_tau:
                new = current
        for r in robots:
            self._set_goal(r, None if new[r] is None else frontiers[new[r]])


def values_in_order(sorted_frontiers: np.ndarray, sorted_values: Sequence[float], frontiers: np.ndarray) -> np.ndarray:
    """Puts values that come sorted with their frontiers (as ValueMap.sort_waypoints
    returns them) back in the order of `frontiers`."""
    sorted_frontiers = np.asarray(sorted_frontiers, dtype=float).reshape(-1, 2)
    values = np.empty(len(frontiers))
    for i, frontier in enumerate(np.asarray(frontiers, dtype=float).reshape(-1, 2)):
        values[i] = sorted_values[int(np.argmin(np.linalg.norm(sorted_frontiers - frontier, axis=1)))]
    return values


def iqr(samples: Sequence[float]) -> float:
    q25, q75 = np.percentile(np.asarray(samples, dtype=float), [25, 75])
    return float(q75 - q25)


class UtilityScaleLog:
    """Values and path costs of all frontiers seen at exploration steps, to set the
    utility's fixed scales (value_scale, path_scale) from real runs.

    Besides the pooled percentiles, it keeps the spread within each step: the
    interquartile range of the values and of the (reachable) path costs among the
    frontiers the robot chose from. The scales are the median of these over all steps
    with at least MIN_FRONTIERS frontiers, so that V / s_V and A / s_A vary by similar
    amounts between the frontiers of one decision (design doc 3.4)."""

    MIN_FRONTIERS = 3

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._values: List[float] = []
        self._actions: List[float] = []
        self._unreachable = 0
        self._step_iqrs: List[List[float]] = []  # [frontiers, IQR of V, IQR of A] per step

    def add(self, values: Sequence[float], costs: Sequence[Any]) -> None:
        self._values.extend(float(v) for v in values)
        reachable = [float(c.actions) for c in costs if c.reachable]
        self._actions.extend(reachable)
        self._unreachable += sum(not c.reachable for c in costs)
        if len(values) >= self.MIN_FRONTIERS and len(reachable) >= self.MIN_FRONTIERS:
            self._step_iqrs.append([len(values), round(iqr(values), 6), round(iqr(reachable), 4)])

    def summary(self) -> Dict[str, float]:
        def percentiles(name: str, samples: List[float]) -> Dict[str, float]:
            if not samples:
                return {f"{name}_p{p}": -1.0 for p in (25, 50, 75)}
            return {f"{name}_p{p}": float(np.percentile(samples, p)) for p in (25, 50, 75)}

        return {
            "frontiers": len(self._values),
            "unreachable": self._unreachable,
            **percentiles("value", self._values),
            **percentiles("actions", self._actions),
            "step_iqrs": self._step_iqrs,
        }
