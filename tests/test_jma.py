from pathlib import Path

from chiba_heavy_rain.jma import (
    latest_contiguous_complete_run,
    parse_annual_24h,
    parse_event_24h,
)


def _table_row(first: str, value_24h: str, date: str, cell_class: str = "data_0_0") -> str:
    cells = [first]
    for pair in range(1, 8):
        cells.extend([value_24h if pair == 5 else "1.0", date if pair == 5 else "1/1"])
    return "<tr>" + "".join(
        f'<td class="{cell_class}">{value}</td>' for value in cells
    ) + "</tr>"


def test_parse_annual_and_contiguous_run(tmp_path: Path) -> None:
    path = tmp_path / "annual.html"
    rows = [_table_row(str(year), "100.5 )" if year != 1981 else "///", "8/13") for year in range(1979, 2026)]
    path.write_text(
        "<html><table><tr><th>最大24時間降水量</th></tr>" + "".join(rows) + "</table></html>",
        encoding="utf-8",
    )
    frame = parse_annual_24h(path)
    assert frame.loc[frame["year"] == 1980, "max_24h_mm"].item() == 100.5
    run = latest_contiguous_complete_run(frame)
    assert run["year"].min() == 1979
    assert 1981 not in set(run["year"])
    assert len(run) == 46


def test_statistical_break_restarts_series_and_insufficient_year_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "annual_break.html"
    rows = []
    for year in range(1976, 2026):
        if year == 1999:
            rows.append(_table_row(str(year), "116 ]", "8/14", "data_2t_0"))
        else:
            rows.append(_table_row(str(year), "100", "8/13"))
    path.write_text(
        "<html><table><tr><th>最大24時間降水量</th></tr>" + "".join(rows) + "</table></html>",
        encoding="utf-8",
    )
    run = latest_contiguous_complete_run(parse_annual_24h(path))
    assert run["year"].min() == 2000
    assert len(run) == 26


def test_parse_event_selects_largest_event_day(tmp_path: Path) -> None:
    path = tmp_path / "event.html"
    rows = [_table_row(str(day), str(value), f"{day}:00") for day, value in [(12, 80), (13, 300), (14, 350), (15, 40)]]
    path.write_text(
        "<html><table><tr><th>最大24時間降水量</th></tr>" + "".join(rows) + "</table></html>",
        encoding="utf-8",
    )
    event = parse_event_24h(path)
    assert event["day"] == 14
    assert event["event_24h_mm"] == 350.0
