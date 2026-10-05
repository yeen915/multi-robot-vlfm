"""Detects indices that can only come from corrupted memory.

On the evaluation machine, numpy indices occasionally come back with a stray high bit
set (for example 2**51 + 1131 for an axis of size 640). VLFM treats any IndexError
while updating its maps as "reached the edge of the map" and ends the episode, which
would silently count such an episode as a failure.
"""

import re

# An index this many times larger than the axis cannot be a real map overflow: the
# maps are 1000 px wide and the robot moves 5 px per step.
CORRUPTION_FACTOR = 1000

_INDEX_ERROR_PATTERN = re.compile(r"index (-?\d+) is out of bounds for axis \d+ with size (\d+)")


def is_corrupted_index_error(error: BaseException) -> bool:
    """Whether an IndexError reports an index that is orders of magnitude outside the
    array, as opposed to one that is just past its edge."""
    match = _INDEX_ERROR_PATTERN.search(str(error))
    if match is None:
        return False
    index, size = int(match.group(1)), int(match.group(2))
    return abs(index) > CORRUPTION_FACTOR * max(size, 1)
