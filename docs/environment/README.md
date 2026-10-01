# Environment snapshot (single-robot VLFM reproduction)

Snapshot of the `vlfm` conda env on 2026-10-01, after a 20-episode HM3D val
smoke test ran end to end (SR 35%, SPL 0.149, all from scene 00853).

- `pip_freeze.txt`: `python -m pip freeze`
- `conda_list.txt`: `conda list -n vlfm --export`

Key versions: python 3.9, torch 1.12.1+cu113, habitat-sim 0.2.4,
habitat-lab 0.2.4, numpy 1.26.4, numba 0.60.0, opencv-python 4.5.5.64.

External repos cloned into the repo root (ignored by git):
- `GroundingDINO/` at eeba084 (IDEA-Research/GroundingDINO)
- `yolov7/` at a207844 (WongKinYiu/yolov7)

## Fixes needed to get a run working

1. `env.reset()` crashed with numba `can't unbox array from PyObject`.
   numpy 1.26.4 (conda) and 2.0.2 (pip) were installed over each other, so
   `numpy/` held files from both. Fix: `pip uninstall -y numpy` twice, then
   `pip install --no-deps numpy==1.26.4`.
2. Video writing failed with `No ffmpeg exe could be found`. The env's
   `bin/ffmpeg` could not start because `libiconv.so.2` was missing.
   Fix: `conda install -n vlfm -c conda-forge "libiconv>=1.17,<2"`.

## Limits found on this machine (one GTX 1080 Ti, 11 GB)

- The four model servers use about 9 GB; one Habitat process adds about 1 GB.
- `habitat_baselines.num_environments > 1` is not supported by the VLFM policy.
- Two Habitat processes sharing the servers ran Mobile SAM out of GPU memory.
- One episode takes about 65 s.
