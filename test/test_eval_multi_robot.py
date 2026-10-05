import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import eval_multi_robot as ev  # noqa: E402

EPISODES = [("sceneA", "1"), ("sceneA", "2"), ("sceneB", "1")]


def write(directory: Any, episode: ev.Episode, **fields: Any) -> None:
    with open(ev.result_path(str(directory), episode), "w") as f:
        json.dump({"success": 0.0, "num_steps": 10, **fields}, f)


def test_only_episodes_without_a_result_are_pending(tmp_path: Any) -> None:
    write(tmp_path, EPISODES[0])
    with open(ev.result_path(str(tmp_path), EPISODES[1]), "w") as f:
        f.write("")  # a run that died while writing
    assert ev.pending_episodes(str(tmp_path), EPISODES) == EPISODES[1:]


def test_set_aside_makes_an_episode_pending_and_keeps_every_copy(tmp_path: Any) -> None:
    for _ in range(2):
        write(tmp_path, EPISODES[0], hw_corruption=True)
        ev.set_aside(str(tmp_path), EPISODES[0], "hw_corruption")
    assert ev.pending_episodes(str(tmp_path), EPISODES[:1]) == EPISODES[:1]
    assert sorted(os.listdir(tmp_path / "corrupted")) == ["1_sceneA.hw_corruption0.json", "1_sceneA.hw_corruption1.json"]


def test_differing_fields() -> None:
    a = {"success": 1.0, "num_steps": 50, "multi_robot": {"joint_steps": 25}, "failure_cause": "x"}
    assert ev.differing_fields(a, dict(a)) == []
    # Fields that are not compared (e.g. free text) do not count
    assert ev.differing_fields(a, {**a, "failure_cause": "y"}) == []
    b = {**a, "num_steps": 51, "multi_robot": {"joint_steps": 26}}
    assert ev.differing_fields(a, b) == ["num_steps", "multi_robot"]


def test_duplicates_are_a_fixed_sample() -> None:
    episodes = [("scene", str(i)) for i in range(200)]
    chosen = ev.choose_duplicates(episodes, 0.2, seed=0)
    assert len(chosen) == 40
    assert len(set(chosen)) == 40
    assert chosen == ev.choose_duplicates(episodes, 0.2, seed=0)
    assert chosen != ev.choose_duplicates(episodes, 0.2, seed=1)
    assert len(ev.choose_duplicates(episodes[:3], 0.2, seed=0)) == 1  # rounds up


def test_every_method_has_overrides() -> None:
    assert {"single", "independent", "shared_independent_utility", "joint", "greedy", "shared_vlfm"} == set(ev.METHODS)


def test_flagged_results_are_set_aside_and_logged(tmp_path: Any) -> None:
    write(tmp_path, EPISODES[0], hw_corruption=True)
    write(tmp_path, EPISODES[1], hw_corruption=False)
    log: Any = {"hw_corruption": []}
    ev.set_aside_flagged(str(tmp_path), EPISODES, log, attempt=2)
    assert ev.pending_episodes(str(tmp_path), EPISODES) == [EPISODES[0], EPISODES[2]]
    assert log["hw_corruption"] == [{"pass": tmp_path.name, "episode": EPISODES[0], "attempt": 2}]
