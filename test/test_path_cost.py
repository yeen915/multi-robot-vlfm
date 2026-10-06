from typing import List, Tuple

import numpy as np
import pytest

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.multi_robot.path_cost import PathPlanner, polyline_turns, straight_line_cost


def make_map(free_rects_xy: List[Tuple[float, float, float, float]]) -> ObstacleMap:
    """A map whose explored, navigable area is the union of (x0, x1, y0, y1) rectangles
    in meters."""
    obstacle_map = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18)
    obstacle_map.reset()
    free = np.zeros_like(obstacle_map.explored_area, dtype=bool)
    for x0, x1, y0, y1 in free_rects_xy:
        corners = obstacle_map._xy_to_px(np.array([[x0, y0], [x1, y1]]))
        (c_a, r_a), (c_b, r_b) = corners
        free[min(r_a, r_b) : max(r_a, r_b) + 1, min(c_a, c_b) : max(c_a, c_b) + 1] = True
    obstacle_map.explored_area = free.copy()
    obstacle_map._navigable_map = free.copy()
    return obstacle_map


CORRIDOR = [(-0.5, 3.5, -0.5, 0.5)]  # along x
L_SHAPE = [(-0.5, 3.5, -0.5, 0.5), (2.5, 3.5, -0.5, 3.5)]  # along x, then along +y


def test_straight_corridor_costs_one_action_per_quarter_meter() -> None:
    cost = PathPlanner(make_map(CORRIDOR)).costs(np.array([0.0, 0.0]), 0.0, [np.array([3.0, 0.0])])[0]
    assert cost.reachable
    assert cost.forward == pytest.approx(12.0, abs=0.3)
    assert cost.turns == pytest.approx(0.0, abs=0.1)


def test_facing_away_adds_a_half_turn() -> None:
    cost = PathPlanner(make_map(CORRIDOR)).costs(np.array([0.0, 0.0]), np.pi, [np.array([3.0, 0.0])])[0]
    assert cost.turns == pytest.approx(6.0, abs=0.1)  # 180 degrees in 30 degree turns


def test_corner_adds_a_quarter_turn() -> None:
    cost = PathPlanner(make_map(L_SHAPE)).costs(np.array([0.0, 0.0]), 0.0, [np.array([3.0, 3.0])])[0]
    assert cost.reachable
    # 3 m + 3 m, cutting the inside of the corner a little
    assert 21.0 < cost.forward <= 24.5
    # At least the 90 degree corner (3 turns). The shortest path in a 1 m wide corridor
    # cuts the corner diagonally and ends with a diagonal to the goal on the center
    # line, so it turns 0 -> 45 -> 90 -> 45 degrees: 4.5 turns
    assert 3.0 <= cost.turns <= 5.0
    assert cost.actions == pytest.approx(cost.forward + cost.turns)


def test_unreachable_goal_is_a_penalized_straight_line() -> None:
    two_rooms = [(-0.5, 1.0, -0.5, 0.5), (3.0, 4.0, -0.5, 0.5)]
    planner = PathPlanner(make_map(two_rooms), unreachable_penalty=3.0)
    cost = planner.costs(np.array([0.0, 0.0]), 0.0, [np.array([3.5, 0.0])])[0]
    assert not cost.reachable
    assert cost.actions == pytest.approx(3.0 * 3.5 / 0.25)


def test_goal_outside_the_explored_area_is_snapped_into_it() -> None:
    planner = PathPlanner(make_map(CORRIDOR))
    # A frontier just past the end of the corridor, in unknown space
    snapped = planner.snap(np.array([4.0, 0.0]))
    assert snapped[0] == pytest.approx(3.5, abs=0.06) and snapped[1] == pytest.approx(0.0, abs=0.06)
    cost = planner.costs(np.array([0.0, 0.0]), 0.0, [np.array([4.0, 0.0])])[0]
    assert cost.reachable and cost.forward == pytest.approx(14.0, abs=0.3)


def test_robot_off_the_traversable_area_is_snapped_too() -> None:
    cost = PathPlanner(make_map(CORRIDOR)).costs(np.array([0.0, 0.8]), 0.0, [np.array([3.0, 0.0])])[0]
    assert cost.reachable


def test_empty_map_estimates_everything() -> None:
    planner = PathPlanner(make_map([]), unreachable_penalty=2.0)
    cost = planner.costs(np.array([0.0, 0.0]), 0.0, [np.array([1.0, 0.0])])[0]
    assert not cost.reachable and cost.actions == pytest.approx(8.0)


def test_straight_line_and_polyline_turns() -> None:
    cost = straight_line_cost(np.array([0.0, 0.0]), 0.0, np.array([0.0, 1.0]))
    assert cost.forward == pytest.approx(4.0) and cost.turns == pytest.approx(3.0)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert polyline_turns(square, 0.0) == pytest.approx(6.0)  # two 90 degree corners
