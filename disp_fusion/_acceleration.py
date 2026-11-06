"""Optional native compilation; mathematical parameters are never changed."""
import os
import numpy as np


ENABLED = os.environ.get('DISP_FUSION_ACCELERATION', '1') != '0'
if ENABLED:
    try:
        from numba import njit
    except ImportError:
        ENABLED = False


def compiled(function):
    """Keep a NumPy fallback and do not enable unsafe fast-math transforms."""
    if not ENABLED:
        return function
    try:
        return njit(cache=True, nogil=True, fastmath=False)(function)
    except RuntimeError as error:
        # Read-only/sandboxed installations may have no writable cache locator.
        # Caching is an execution convenience, never an import requirement.
        if 'cannot cache function' not in str(error):
            raise
        return njit(cache=False, nogil=True, fastmath=False)(function)


def window_workers():
    """Only native, GIL-free jobs benefit from the small thread pool."""
    if not ENABLED:
        return 1
    value = int(os.environ.get('DISP_FUSION_WORKERS', min(4, os.cpu_count() or 1)))
    if value < 1:
        raise ValueError('DISP_FUSION_WORKERS must be a positive integer')
    return value


@compiled
def linear_percentile(values, percentile):
    """NumPy's default linear interpolation for nonempty, finite float64 data.

    Numba's built-in percentile uses a different interpolation expression.
    Keeping NumPy's expression avoids moving a strict fit gate by an ULP.
    """
    rank = (len(values)-1)*(percentile/100.)
    left = int(np.floor(rank))
    right = min(left+1, len(values)-1)
    ordered = np.partition(values, (left, right))
    fraction = rank-left
    difference = ordered[right]-ordered[left]
    if fraction >= .5:
        return ordered[right]-difference*(1.-fraction)
    return ordered[left]+difference*fraction
