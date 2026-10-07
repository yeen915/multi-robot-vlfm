"""Minimal stand-in for the parts of pytest the tests use (pytest is not installed)."""
import math
from contextlib import contextmanager
import numpy as np

class approx:
    __array_ufunc__ = None  # make numpy defer == to us
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs
    def _eq(self, a, e):
        if isinstance(e, (list, tuple, np.ndarray)):
            a = list(np.asarray(a).ravel()) if isinstance(a, np.ndarray) else list(a)
            e = list(np.asarray(e).ravel()) if isinstance(e, np.ndarray) else list(e)
            return len(a) == len(e) and all(self._eq(x, y) for x, y in zip(a, e))
        if isinstance(e, dict):
            return set(a) == set(e) and all(self._eq(a[k], e[k]) for k in e)
        return math.isclose(float(a), float(e), rel_tol=self.rel, abs_tol=self.abs)
    def __eq__(self, other):
        return self._eq(other, self.expected)
    def __repr__(self):
        return f"approx({self.expected!r})"

@contextmanager
def raises(exc, match=None):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"DID NOT RAISE {exc}")

class _Mark:
    @staticmethod
    def parametrize(names, values):
        def deco(f):
            f._params = ([n.strip() for n in names.split(",")], values)
            return f
        return deco
mark = _Mark()
