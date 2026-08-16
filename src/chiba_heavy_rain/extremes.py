"""GEV fitting and return-period uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genextreme


@dataclass(frozen=True)
class GEVFit:
    shape_xi: float
    location: float
    scale: float

    @property
    def scipy_shape(self) -> float:
        # scipy.stats.genextreme uses c = -xi.
        return -self.shape_xi


def fit_gev(values: np.ndarray) -> GEVFit:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("GEV fit requires at least three finite one-dimensional values")
    c, loc, scale = genextreme.fit(x)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid GEV scale estimate")
    return GEVFit(shape_xi=float(-c), location=float(loc), scale=float(scale))


def return_period(value: float, fit: GEVFit) -> float:
    log_p = genextreme.logsf(value, fit.scipy_shape, loc=fit.location, scale=fit.scale)
    if not np.isfinite(log_p):
        return float("inf")
    if log_p < np.log(np.finfo(float).tiny):
        return float("inf")
    return float(np.exp(-log_p))


def return_level(period: np.ndarray, fit: GEVFit) -> np.ndarray:
    periods = np.asarray(period, dtype=float)
    return genextreme.ppf(
        1.0 - 1.0 / periods, fit.scipy_shape, loc=fit.location, scale=fit.scale
    )


def bootstrap_return_period(
    values: np.ndarray,
    event_value: float,
    samples: int = 2000,
    seed: int = 20260813,
) -> dict[str, float | int]:
    """Nonparametric station-wise bootstrap percentile interval."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    failures = 0
    for _ in range(samples):
        draw = rng.choice(x, size=len(x), replace=True)
        try:
            estimate = return_period(event_value, fit_gev(draw))
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
            continue
        estimates.append(estimate)
    finite = np.asarray([v for v in estimates if np.isfinite(v)], dtype=float)
    infinite = sum(not np.isfinite(v) for v in estimates)
    if len(finite) == 0:
        return {
            "ci_low": float("nan"),
            "ci_median": float("nan"),
            "ci_high": float("inf"),
            "bootstrap_valid": len(estimates),
            "bootstrap_infinite": infinite,
            "bootstrap_failures": failures,
        }
    # Treat any infinite estimates as an unbounded upper percentile.
    infinite_fraction = infinite / max(len(estimates), 1)
    upper = float("inf") if infinite_fraction >= 0.025 else float(np.quantile(finite, 0.975))
    return {
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_median": float(np.quantile(finite, 0.5)),
        "ci_high": upper,
        "bootstrap_valid": len(estimates),
        "bootstrap_infinite": infinite,
        "bootstrap_failures": failures,
    }

