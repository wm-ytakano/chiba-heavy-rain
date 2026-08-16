from pathlib import Path

import numpy as np

import chiba_heavy_rain.daily_vs_24h as comparison
from chiba_heavy_rain.daily_vs_24h import (
    _empirical_percentile,
    paired_bootstrap_metrics,
    return_period_metrics,
)
from chiba_heavy_rain.extremes import fit_gev, return_period
from chiba_heavy_rain.jma import _merge_manifest, parse_annual_daily_max, parse_event_daily


def _daily_table(rows: list[tuple[str, str]], amedas: bool) -> str:
    unit = " (mm)" if amedas else ""
    header = f"""
    <tr><th rowspan="3">年</th><th colspan="4">降水量{' ' if amedas else '(mm)'}</th></tr>
    <tr><th rowspan="2">合計{unit}</th><th colspan="3">最大</th></tr>
    <tr><th>日{unit}</th><th>1時間{unit}</th><th>10分間{unit}</th></tr>
    """
    body = "".join(
        f'<tr><td>{label}</td><td>100</td><td class="data_0_0">{value}</td>'
        "<td>10</td><td>5</td></tr>"
        for label, value in rows
    )
    return f"<html><table>{header}{body}</table></html>"


def _event_table(rows: list[tuple[str, str]], amedas: bool) -> str:
    unit = " (mm)" if amedas else ""
    header = f"""
    <tr><th rowspan="2">日</th><th colspan="3">降水量{' ' if amedas else '(mm)'}</th></tr>
    <tr><th>合計{unit}</th><th>最大1時間{unit}</th><th>最大10分間{unit}</th></tr>
    """
    body = "".join(
        f'<tr><td>{label}</td><td class="data_0_0">{value}</td><td>10</td><td>5</td></tr>'
        for label, value in rows
    )
    return f"<html><table>{header}{body}</table></html>"


def _n_hour_table(rows: list[tuple[str, str]]) -> str:
    body = []
    for label, value in rows:
        cells = [label]
        for pair in range(1, 8):
            cells.extend([value if pair == 5 else "1", "8/13 12:00"])
        body.append("<tr>" + "".join(f'<td class="data_0_0">{cell}</td>' for cell in cells) + "</tr>")
    return (
        "<html><table><tr><th>最大24時間降水量</th></tr>"
        + "".join(body)
        + "</table></html>"
    )


def test_parse_annual_daily_max_for_surface_and_amedas_headers(tmp_path: Path) -> None:
    for amedas in (False, True):
        path = tmp_path / f"annual_{amedas}.html"
        path.write_text(
            _daily_table([("1976", "120.5"), ("1977", "130 ]")], amedas),
            encoding="utf-8",
        )
        frame = parse_annual_daily_max(path)
        assert frame["max_daily_mm"].tolist() == [120.5, 130.0]
        assert frame["daily_usable"].tolist() == [True, False]


def test_parse_event_daily_selects_largest_of_event_days(tmp_path: Path) -> None:
    path = tmp_path / "event.html"
    path.write_text(
        _event_table([("12", "80"), ("13", "250"), ("14", "200"), ("15", "30")], True),
        encoding="utf-8",
    )
    event = parse_event_daily(path)
    assert event["day"] == 13
    assert event["event_daily_mm"] == 250.0


def test_manifest_merge_preserves_other_table_types(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '[{"block_no":"47682","table":"annual","sha256":"old"}]\n',
        encoding="utf-8",
    )
    _merge_manifest(
        tmp_path,
        [{"block_no": "47682", "table": "annual_daily", "sha256": "new"}],
    )
    text = manifest.read_text(encoding="utf-8")
    assert '"table": "annual"' in text
    assert '"table": "annual_daily"' in text


def test_three_return_period_scenarios_match_direct_calculation() -> None:
    daily = np.linspace(70.0, 170.0, 39)
    hourly = daily * np.linspace(1.02, 1.18, 39)
    metrics = return_period_metrics(daily, hourly, 185.0, 210.0)
    daily_fit = fit_gev(daily)
    hourly_fit = fit_gev(hourly)
    expected_24h = return_period(210.0, hourly_fit)
    expected_mismatch = return_period(210.0, daily_fit)
    expected_daily = return_period(185.0, daily_fit)
    assert np.isclose(metrics["rp_24h_matched"], expected_24h)
    assert np.isclose(metrics["rp_daily_mismatched"], expected_mismatch)
    assert np.isclose(metrics["rp_daily_matched"], expected_daily)
    assert np.isclose(metrics["mismatch_inflation"], expected_mismatch / expected_24h)


def test_paired_bootstrap_is_reproducible_and_preserves_infinite_upper() -> None:
    rng = np.random.default_rng(9)
    daily = rng.gamma(4.0, 20.0, 39)
    hourly = daily * rng.uniform(1.0, 1.2, 39)
    first = paired_bootstrap_metrics(daily, hourly, 180.0, 200.0, samples=40, seed=4)
    second = paired_bootstrap_metrics(daily, hourly, 180.0, 200.0, samples=40, seed=4)
    assert first == second
    assert first["bootstrap_valid"] == 40
    assert _empirical_percentile([1.0] * 97 + [float("inf")] * 3, 0.975) == float("inf")


def test_run_writes_only_new_artifacts_and_preserves_report(
    tmp_path: Path, monkeypatch: object
) -> None:
    raw_dir = tmp_path / "raw"
    results_dir = tmp_path / "results"
    raw_dir.mkdir()
    results_dir.mkdir()
    report = results_dir / "report.md"
    report.write_text("keep this blog unchanged\n", encoding="utf-8")
    years = list(range(1976, 2015))
    annual_daily_rows = [
        (str(year), f"{80 + index * 2:.1f}") for index, year in enumerate(years)
    ]
    annual_24h_rows = [
        (str(year), f"{90 + index * 2.2:.1f}") for index, year in enumerate(years)
    ]
    for station in comparison.ARTICLE_STATIONS:
        (raw_dir / f"{station.block_no}_annual_daily.html").write_text(
            _daily_table(annual_daily_rows, station.kind == "a"), encoding="utf-8"
        )
        (raw_dir / f"{station.block_no}_annual.html").write_text(
            _n_hour_table(annual_24h_rows), encoding="utf-8"
        )
        (raw_dir / f"{station.block_no}_event_daily.html").write_text(
            _event_table([("13", "170"), ("14", "20"), ("15", "0")], station.kind == "a"),
            encoding="utf-8",
        )
        (raw_dir / f"{station.block_no}_event.html").write_text(
            _n_hour_table([("13", "190"), ("14", "30"), ("15", "0")]),
            encoding="utf-8",
        )
    monkeypatch.setattr(comparison, "download_pages", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        comparison, "download_daily_comparison_pages", lambda *args, **kwargs: []
    )
    output = comparison.run(raw_dir, results_dir, bootstrap_samples=5)
    assert len(output) == 8
    assert report.read_text(encoding="utf-8") == "keep this blog unchanged\n"
    for filename in (
        "annual_max_daily_vs_24h.csv",
        "daily_vs_24h_return_periods.csv",
        "daily_vs_24h_gev.png",
        "daily_vs_24h_ratio.png",
    ):
        assert (results_dir / filename).exists()
