"""Runs the unit tests without pytest, which is not installed on the server.

Usage (from the repo root):
    $VLFM_PYTHON scripts/dev/run_tests.py                     # all of test/
    $VLFM_PYTHON scripts/dev/run_tests.py test_coordinator    # some modules

scripts/dev/pytest_shim/pytest.py stands in for the parts of pytest the tests use
(approx, raises, mark.parametrize, the tmp_path fixture).
"""
import importlib, inspect, os, sys, tempfile, traceback
from pathlib import Path
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest_shim"), "test", "."]
mods = sys.argv[1:] or sorted(p[:-3] for p in os.listdir("test") if p.startswith("test_") and p.endswith(".py"))
passed = failed = 0
for m in mods:
    mod = importlib.import_module(m)
    for name, f in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or f.__module__ != mod.__name__:
            continue
        names, values = getattr(f, "_params", ([], [None]))
        for v in values:
            kw = {}
            if names:
                v = v if isinstance(v, (tuple, list)) and len(names) > 1 else (v,)
                kw = dict(zip(names, v))
            if "tmp_path" in inspect.signature(f).parameters:
                kw["tmp_path"] = Path(tempfile.mkdtemp())
            try:
                f(**kw); passed += 1
            except Exception:
                failed += 1; print(f"FAIL {m}::{name}{kw if names else ''}"); traceback.print_exc()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
