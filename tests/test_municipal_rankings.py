from datetime import datetime, timedelta, timezone

import numpy as np
from shapely.geometry import box

from chiba_heavy_rain.municipal_rankings import rank_municipalities
from chiba_heavy_rain.nusdas_rain import FILL

JST = timezone(timedelta(hours=9))


def test_separate_municipal_maxima_and_paired_values():
    lon = np.array([140.0, 140.01, 140.02, 140.03])
    lat = np.array([35.01, 35.0])
    rain = np.array([[50.0, 100.0, 80.0, 30.0],
                     [60.0, 90.0, 70.0, 40.0]])
    periods = np.array([[1000.0, 10.0, 300.0, 2.0],
                        [500.0, 20.0, 200.0, 3.0]])
    hours = np.full_like(rain, 48, dtype=np.int16)
    municipalities = [
        ("12001", "甲", box(139.995, 34.995, 140.015, 35.015)),
        ("12002", "乙", box(140.015, 34.995, 140.035, 35.015)),
    ]
    by_rain, by_period = rank_municipalities(
        lon, lat, rain, periods, hours, municipalities, datetime(2026, 8, 13, tzinfo=JST),
    )
    assert [(row["name"], row["rainfall_mm"], row["return_period_years"])
            for row in by_rain] == [("甲", 100.0, 10.0), ("乙", 80.0, 300.0)]
    assert [(row["name"], row["rainfall_mm"], row["return_period_years"])
            for row in by_period] == [("甲", 50.0, 1000.0), ("乙", 80.0, 300.0)]
    assert by_rain[0]["end_time_jst"] == "2026-08-15 00:00"
    assert by_rain[0]["grid_cells"] == 4


def test_missing_period_does_not_discard_rainfall_maximum():
    lon = np.array([140.0, 140.01])
    lat = np.array([35.0])
    rain = np.array([[200.0, 100.0]])
    periods = np.array([[FILL, 40.0]])
    hours = np.array([[24, 25]])
    municipality = [("12001", "甲", box(139.99, 34.99, 140.02, 35.01))]
    by_rain, by_period = rank_municipalities(
        lon, lat, rain, periods, hours, municipality, datetime(2026, 8, 13, tzinfo=JST),
    )
    assert by_rain[0]["rainfall_mm"] == 200.0
    assert by_rain[0]["return_period_years"] is None
    assert by_period[0]["rainfall_mm"] == 100.0
    assert by_period[0]["return_period_years"] == 40.0
