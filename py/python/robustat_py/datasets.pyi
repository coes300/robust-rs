"""Type stubs for the ``robustat_py.datasets`` submodule."""

import numpy as np
from numpy.typing import NDArray

def stackloss() -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def stars_cyg() -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
