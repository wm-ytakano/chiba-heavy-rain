from datetime import date

import numpy as np
from scipy.io import netcdf_file

from chiba_heavy_rain.extremes import fit_gev, return_period
from chiba_heavy_rain.nusdas_rain import (
    JMA_RAINFALL_RGB,
    RollingMaximum,
    _dissolved_prefecture_segments,
    _geometry_segments,
    _map_display_values,
    _rainfall_color_scale,
    _return_period_color_scale,
    _write_field,
    _write_final,
    aggregate_period,
    fit_gev_grid,
)


def test_polygon_boundaries_include_holes_and_multipolygon_parts() -> None:
    square = [[0, 0], [1, 0], [1, 1], [0, 0]]
    hole = [[0.2, 0.2], [0.3, 0.2], [0.2, 0.2]]
    polygon = {"type": "Polygon", "coordinates": [square, hole]}
    multi = {"type": "MultiPolygon", "coordinates": [[square], [hole]]}
    assert len(_geometry_segments(polygon)) == 2
    assert len(_geometry_segments(multi)) == 2


def test_prefecture_dissolve_removes_municipal_divider() -> None:
    def feature(code, west, east):
        return {
            "properties": {"regioncode": code},
            "geometry": {"type": "Polygon", "coordinates": [[
                [west, 0], [east, 0], [east, 1], [west, 1], [west, 0],
            ]]},
        }

    segments, count = _dissolved_prefecture_segments(
        [feature("1200001", 0, 1), feature("1200002", 1, 2), feature("1300001", 2, 3)],
        (0, 3, 0, 1),
    )
    assert count == 2
    assert any(np.allclose(segment[:, 0], 2) for segment in segments)
    assert not any(np.allclose(segment[:, 0], 1) for segment in segments)


def test_return_period_color_scale_has_white_under_and_extended_upper_bin() -> None:
    cmap, norm, bounds = _return_period_color_scale()
    assert len(bounds) == 10
    np.testing.assert_allclose(cmap(norm(9.9)), [1, 1, 1, 1])
    assert not np.allclose(cmap(norm(10)), [1, 1, 1, 1])
    assert norm(10000) < cmap.N
    assert norm(np.nextafter(10000.0, np.inf)) == cmap.N
    masked = np.ma.masked_invalid(np.array([np.nan]))
    np.testing.assert_allclose(cmap(norm(masked))[0], [1, 1, 1, 1])


def test_infinite_return_period_uses_upper_extension_instead_of_missing_white() -> None:
    cmap, norm, _ = _return_period_color_scale()
    display = _map_display_values(np.array([np.nan, 5.0, np.inf]), norm)
    assert display.mask.tolist() == [True, False, False]
    np.testing.assert_allclose(cmap(norm(display))[0], [1, 1, 1, 1])
    np.testing.assert_allclose(cmap(norm(display))[1], [1, 1, 1, 1])
    np.testing.assert_allclose(cmap(norm(display))[2], cmap.get_over())


def test_jma_rainfall_colors_and_24h_thresholds() -> None:
    cmap, norm, bounds = _rainfall_color_scale()
    np.testing.assert_array_equal(bounds[:-1], [1, 10, 20, 50, 100, 200, 300])
    np.testing.assert_allclose(cmap(norm(0.9)), [1, 1, 1, 1])
    np.testing.assert_allclose(cmap(norm(1))[:3], np.array(JMA_RAINFALL_RGB[0]) / 255)
    np.testing.assert_allclose(cmap(norm(400))[:3], np.array(JMA_RAINFALL_RGB[-2]) / 255)
    np.testing.assert_allclose(cmap(norm(401))[:3], np.array(JMA_RAINFALL_RGB[-1]) / 255)
    masked = np.ma.masked_invalid(np.array([np.nan]))
    np.testing.assert_allclose(cmap(norm(masked))[0], [1, 1, 1, 1])


def test_rolling_maximum_requires_24_consecutive_valid_hours() -> None:
    rolling = RollingMaximum((1, 2))
    for hour in range(48):
        field = np.array([[1.0, 2.0]], dtype=np.float32)
        if hour == 12:
            field[0, 1] = np.nan
        rolling.add(field)
    np.testing.assert_allclose(rolling.maximum, [[24.0, 48.0]])
    np.testing.assert_array_equal(rolling.valid_windows, [[25, 12]])
    assert rolling.argmax_hour[0, 1] == 37


def test_missing_and_failed_files_keep_annual_maximum(tmp_path) -> None:
    db = tmp_path / "db" / "_SRF" / "LL" / "AA" / "PR10"
    db.mkdir(parents=True)
    for day in (date(2006, 1, 2), date(2006, 1, 3)):
        (db / day.strftime("%Y%m%d0000")).touch()

    class Reader:
        shape = (1, 1)

        def read(self, day, hour):
            if day == date(2006, 1, 2) and hour == 5:
                raise OSError("corrupt record")
            return np.array([[float(day.day)]], dtype=np.float32)

    status = tmp_path / "status.csv"
    missing = tmp_path / "missing.csv"
    actual = aggregate_period(
        Reader(), tmp_path / "db", date(2006, 1, 1), date(2006, 1, 3),
        status, missing,
    )
    assert actual.maximum[0, 0] == 72.0
    assert actual.valid_windows[0, 0] == 20
    assert "read_failed" in missing.read_text()
    assert "file_missing" in missing.read_text()
    assert "2006-01-01" in status.read_text()


def test_grid_gev_matches_existing_scalar_fit_and_excludes_event() -> None:
    values = np.array(
        [80, 91, 85, 106, 112, 103, 132, 97, 120, 140,
         151, 124, 134, 160, 142, 149, 173, 154, 181, 190],
        dtype=np.float32,
    )
    annual = values[:, None, None]
    event = np.array([[250.0]], dtype=np.float32)
    fit = fit_gev(values)
    result = fit_gev_grid(annual, event)
    np.testing.assert_allclose(result["shape_xi"], fit.shape_xi, atol=1e-6)
    np.testing.assert_allclose(result["location"], fit.location, rtol=1e-6)
    np.testing.assert_allclose(result["scale"], fit.scale, rtol=1e-6)
    np.testing.assert_allclose(
        result["return_period_years"], return_period(250, fit), rtol=1e-5
    )
    annual[0, 0, 0] = np.nan
    assert np.isnan(fit_gev_grid(annual, event)["return_period_years"][0, 0])
    annual[0, 0, 0] = 80.0
    no_event = fit_gev_grid(annual, np.array([[np.nan]], dtype=np.float32))
    assert np.isfinite(no_event["location"][0, 0])
    assert np.isnan(no_event["return_period_years"][0, 0])


def test_cached_annual_fields_reappear_in_final_netcdf(tmp_path) -> None:
    class Reader:
        shape = (1, 2)
        lon = np.array([139.0, 140.0])
        lat = np.array([35.0])

    reader = Reader()
    paths = []
    for index in range(20):
        path = tmp_path / f"annual_{2006 + index}.nc"
        _write_field(
            path,
            np.array([[80.0 + index * 5, np.nan]], dtype=np.float32),
            np.array([[100, 0]], dtype=np.uint32),
            reader.lon, reader.lat,
        )
        paths.append(path)
    event = RollingMaximum(reader.shape)
    event.maximum[0] = [150.0, 200.0]
    event.argmax_hour[0] = [48, 0]
    output = tmp_path / "event.nc"
    _write_final(output, paths, event, reader, date(2026, 8, 13), date(2026, 8, 15))
    with netcdf_file(output, "r", mmap=False) as dataset:
        annual = dataset.variables["annual_max_24h_mm"][:].copy()
        years = dataset.variables["n_years"][:].copy()
        periods = dataset.variables["return_period_years"][:].copy()
    np.testing.assert_allclose(annual[:, 0, 0], 80 + np.arange(20) * 5)
    assert np.all(annual[:, 0, 1] == -9999)
    np.testing.assert_array_equal(years, [[20, 0]])
    assert periods[0, 0] > 0
    assert periods[0, 1] == -9999
