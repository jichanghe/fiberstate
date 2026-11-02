"""fiberstate: sequence-aware single-molecule chromatin state calling for Fiber-seq."""
import os as _os

# fiberstate parallelises across fibers with one process per core. numpy's BLAS library
# (OpenBLAS / MKL) would otherwise start its own thread pool in every worker for the small
# dot products in the per-fiber EM, and those pools fight each other for the CPU
# (seen as 4x slower workers on a 14-core laptop). One BLAS thread per process avoids it.
# Set FIBERSTATE_BLAS_THREADS to override.
_n = _os.environ.get("FIBERSTATE_BLAS_THREADS", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, _n)

__version__ = "0.5.1"
from .model import Model, Params, ReadCall  # noqa: E402,F401
