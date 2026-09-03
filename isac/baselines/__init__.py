"""Internal OFDM-ISAC baselines (uniform, water-filling, conventional S2, exact EFIM).

Published-paper reproductions are intentionally absent.  Do not label a curve
with a paper citation unless that method has been implemented under this model.
"""

from isac.baselines.base import BaselineResult
from isac.baselines.internal import (
    run_conventional_s2,
    run_exact_efim,
    run_uniform,
    run_water_filling,
)

__all__ = [
    "BaselineResult",
    "run_uniform",
    "run_water_filling",
    "run_conventional_s2",
    "run_exact_efim",
]
