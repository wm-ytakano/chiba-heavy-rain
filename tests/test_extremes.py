import numpy as np
import pandas as pd
from scipy.stats import genextreme

from chiba_heavy_rain.analysis import _historical_period_results
from chiba_heavy_rain.extremes import GEVFit, fit_gev, return_level, return_period


def test_return_period_and_level_are_inverse() -> None:
    fit = GEVFit(shape_xi=0.1, location=100.0, scale=25.0)
    periods = np.array([2.0, 10.0, 100.0, 1000.0])
    levels = return_level(periods, fit)
    actual = np.array([return_period(x, fit) for x in levels])
    np.testing.assert_allclose(actual, periods, rtol=1e-10)


def test_scipy_shape_sign_conversion() -> None:
    fit = GEVFit(shape_xi=0.2, location=0.0, scale=1.0)
    assert fit.scipy_shape == -0.2
    expected = genextreme.isf(0.01, -0.2)
    assert np.isclose(return_level(np.array([100.0]), fit)[0], expected)


def test_fit_gev_returns_valid_scale() -> None:
    rng = np.random.default_rng(4)
    values = genextreme.rvs(-0.1, loc=120, scale=25, size=80, random_state=rng)
    fit = fit_gev(values)
    assert fit.scale > 0
    assert 80 < fit.location < 160


def test_centered_1995_period_is_1976_through_2014_without_quality_filter() -> None:
    current = pd.DataFrame(
        [{"station": "test", "block_no": "0000", "eligible": True, "event_24h_mm": 180.0}]
    )
    annual = pd.DataFrame(
        {
            "station": ["test"] * 41,
            "year": range(1975, 2016),
            "max_24h_mm": np.linspace(80, 160, 41),
            "usable": [False] * 41,
        }
    )
    period, series = _historical_period_results(current, annual)
    assert period.loc[0, "start_year"] == 1976
    assert period.loc[0, "end_year"] == 2014
    assert period.loc[0, "n_years"] == 39
    assert len(series["test"]) == 39
