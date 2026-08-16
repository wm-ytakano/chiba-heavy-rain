"""GEV fitting and return-period uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma
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
    """Estimate stationary GEV parameters with sample L-moments.

    Probability-weighted moments are converted to L1, L2, and L-skewness.
    The GEV shape is obtained by numerically inverting the theoretical
    L-skewness relation. This avoids the pathological unconstrained MLE shapes
    that can occur in short annual-maximum records.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("GEV fit requires at least three finite one-dimensional values")
    ordered = np.sort(x)
    n = len(ordered)
    index = np.arange(n, dtype=float)
    b0 = float(np.mean(ordered))
    b1 = float(np.sum((index / (n - 1)) * ordered) / n)
    b2 = float(
        np.sum((index * (index - 1) / ((n - 1) * (n - 2))) * ordered) / n
    )
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    if not np.isfinite(l2) or l2 <= 0:
        raise RuntimeError("GEV L-moment fit requires positive L2")
    tau3 = l3 / l2
    if not -1.0 < tau3 < 1.0:
        raise RuntimeError(f"GEV L-skewness outside (-1, 1): {tau3}")

    def theoretical_tau3(xi: float) -> float:
        if abs(xi) < 1e-8:
            return 2.0 * np.log(3.0) / np.log(2.0) - 3.0
        numerator = np.expm1(xi * np.log(3.0))
        denominator = np.expm1(xi * np.log(2.0))
        return 2.0 * numerator / denominator - 3.0

    shape_xi = float(brentq(lambda xi: theoretical_tau3(xi) - tau3, -10.0, 0.999999))
    if abs(shape_xi) < 1e-7:
        scale = float(l2 / np.log(2.0))
        location = float(l1 - np.euler_gamma * scale)
        shape_xi = 0.0
    else:
        gamma_term = float(gamma(1.0 - shape_xi))
        scale = float(
            l2
            * shape_xi
            / (gamma_term * np.expm1(shape_xi * np.log(2.0)))
        )
        location = float(l1 - scale * (gamma_term - 1.0) / shape_xi)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(location):
        raise RuntimeError("Invalid GEV L-moment parameter estimate")
    return GEVFit(shape_xi=shape_xi, location=location, scale=scale)


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
