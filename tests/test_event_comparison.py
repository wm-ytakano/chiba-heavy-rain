from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chiba_heavy_rain.event_comparison import (
    EVENTS,
    _window_status,
    nearest_cell,
    station_coordinates,
    write_outputs,
)
from chiba_heavy_rain.stations import Station


def test_station_coordinates_parse_jma_degrees_and_minutes(tmp_path: Path) -> None:
    path = tmp_path / "stations.html"
    area = ("<area href='../index.php?prec_no=45&block_no=0381&year=' "
            "onmouseover=\"javascript:viewPoint('a','0381','茂原','モバラ',"
            "'35','26.2','140','17.6','11');\">")
    path.write_text(f"<html>{area}{area}</html>", encoding="utf-8")
    coords = station_coordinates(path, (Station("0381", "茂原", "a"),))
    assert coords["0381"] == pytest.approx((35 + 26.2 / 60, 140 + 17.6 / 60))


def test_nearest_cell_uses_geographic_distance() -> None:
    y, x, distance = nearest_cell(np.array([35.0, 35.1]), np.array([140.0, 140.1]),
                                  35.08, 140.02)
    assert (y, x) == (1, 0)
    assert 2 < distance < 3


def test_event_window_boundary_and_unknown_end_time() -> None:
    start, end = date(2026, 8, 13), date(2026, 8, 15)
    assert _window_status(13, "10:00", start, end) == "crosses_event_boundary"
    assert _window_status(14, "00:00", start, end) == "within_event"
    assert _window_status(15, "24:00", start, end) == "within_event"
    assert _window_status(14, "///", start, end) == "end_time_unknown"


def test_markdown_shows_two_top_ten_rankings_and_csv_keeps_all(tmp_path: Path) -> None:
    rows = []
    for tag, _, _ in EVENTS:
        for rank in range(12):
            rows.append({
                "event": tag, "station": f"地点{rank}", "block_no": f"{rank:04d}",
                "amedas_24h_mm": 100.0,
                "amedas_return_period_1976_2014_years": 3.0,
                "amedas_return_period_2006_2025_years": 4.0,
                "grid_rainfall_mm": 110.0, "grid_return_period_years": 12.0 - rank,
                "amedas_quality": "numeric", "amedas_window_status": "within_event",
                "grid_status": "ok",
            })
    write_outputs(pd.DataFrame(rows), tmp_path)
    markdown = (tmp_path / "event_comparison_2026.md").read_text(encoding="utf-8")
    csv_table = pd.read_csv(tmp_path / "event_comparison_2026.csv")
    assert markdown.count("| 1 | 地点0 |") == 2
    assert markdown.count("| 10 | 地点9 |") == 2
    assert "地点10" not in markdown
    assert markdown.count("## 2026年") == 2
    assert len(csv_table) == 24
