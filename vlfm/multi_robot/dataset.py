"""ObjectNav dataset restricted to a given list of episodes.

Used to evaluate on a fixed split and to rerun individual episodes. The list is passed
through an environment variable because the dataset config has a fixed schema.
"""

import json
import os
from typing import Any, Optional, Set, Tuple

from habitat import registry
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1

EPISODE_LIST_ENV = "VLFM_EPISODE_LIST"


def scene_name(scene_id: str) -> str:
    """'hm3d/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb' -> '5cdEh9F2hJL'"""
    return os.path.basename(scene_id).split(".")[0]


def load_episode_list(path: str) -> Set[Tuple[str, str]]:
    """Reads a JSON list of {"scene": ..., "episode_id": ...} entries."""
    with open(path) as f:
        return {(entry["scene"], str(entry["episode_id"])) for entry in json.load(f)}


@registry.register_dataset(name="ObjectNavSubset-v1")
class ObjectNavSubsetDataset(ObjectNavDatasetV1):
    """ObjectNav-v1, keeping only the episodes listed in the file that the
    VLFM_EPISODE_LIST environment variable points to. Without it, all episodes are kept.

    Episode ids are the ones Habitat assigns on loading: the index of the episode in
    its scene file.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        path = os.environ.get(EPISODE_LIST_ENV)
        if config is None or not path:
            return
        wanted = load_episode_list(path)
        self.episodes = [
            episode for episode in self.episodes if (scene_name(episode.scene_id), str(episode.episode_id)) in wanted
        ]
