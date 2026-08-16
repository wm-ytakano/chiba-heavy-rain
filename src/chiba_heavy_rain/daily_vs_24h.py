"""Compare fixed-calendar-day and moving 24-hour rainfall return periods."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extremes import fit_gev, return_level, return_period
from .jma import (
    download_daily_comparison_pages,
    download_pages,
    parse_annual_24h,
    parse_annual_daily_max,
    parse_event_24h,
    parse_event_daily,
)
from .stations import STATIONS, Station

ARTICLE_STATION_NAMES = ("千葉", "佐倉", "茂原", "牛久")
ARTICLE_STATIONS: tuple[Station, ...] = tuple(
    station for name in ARTICLE_STATION_NAMES for station in STATIONS if station.name == name
)


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Hiragino Sans",
                "Yu Gothic",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )


def _period_ratio(numerator: float, denominator: float) -> float:
    if np.isinf(numerator) and np.isfinite(denominator):
        return float("inf")
    if np.isfinite(numerator) and np.isinf(denominator):
        return 0.0
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return numerator / denominator


def return_period_metrics(
    daily_values: np.ndarray,
    values_24h: np.ndarray,
    event_daily_mm: float,
    event_24h_mm: float,
) -> dict[str, float]:
    """Fit paired daily/24-hour GEVs and calculate the three planned scenarios."""
    daily = np.asarray(daily_values, dtype=float)
    hourly = np.asarray(values_24h, dtype=float)
    if len(daily) != len(hourly):
        raise ValueError("Daily and 24-hour annual maxima must be paired by year")
    daily_fit = fit_gev(daily)
    hourly_fit = fit_gev(hourly)
    rp_24h_matched = return_period(event_24h_mm, hourly_fit)
    rp_daily_mismatched = return_period(event_24h_mm, daily_fit)
    rp_daily_matched = return_period(event_daily_mm, daily_fit)
    return {
        "annual_ratio_median": float(np.median(hourly / daily)),
        "rp_24h_matched": rp_24h_matched,
        "rp_daily_mismatched": rp_daily_mismatched,
        "rp_daily_matched": rp_daily_matched,
        "mismatch_inflation": _period_ratio(rp_daily_mismatched, rp_24h_matched),
    }


def _empirical_percentile(values: list[float], probability: float) -> float:
    usable = np.sort(np.asarray([value for value in values if not np.isnan(value)], dtype=float))
    if not len(usable):
        return float("nan")
    index = max(0, math.ceil(probability * len(usable)) - 1)
    return float(usable[index])


def paired_bootstrap_metrics(
    daily_values: np.ndarray,
    values_24h: np.ndarray,
    event_daily_mm: float,
    event_24h_mm: float,
    samples: int = 2000,
    seed: int = 20260813,
) -> dict[str, float | int]:
    """Jointly resample years and return percentile intervals for all metrics."""
    daily = np.asarray(daily_values, dtype=float)
    hourly = np.asarray(values_24h, dtype=float)
    if len(daily) != len(hourly):
        raise ValueError("Daily and 24-hour annual maxima must be paired by year")
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {
        "annual_ratio_median": [],
        "rp_24h_matched": [],
        "rp_daily_mismatched": [],
        "rp_daily_matched": [],
        "mismatch_inflation": [],
    }
    failures = 0
    for _ in range(samples):
        indices = rng.integers(0, len(daily), size=len(daily))
        try:
            draw = return_period_metrics(
                daily[indices],
                hourly[indices],
                event_daily_mm,
                event_24h_mm,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
            continue
        for name, value in draw.items():
            estimates[name].append(value)
    if not estimates["rp_24h_matched"]:
        raise RuntimeError("All paired bootstrap fits failed")
    output: dict[str, float | int] = {
        "bootstrap_valid": len(estimates["rp_24h_matched"]),
        "bootstrap_failures": failures,
    }
    for name, values in estimates.items():
        output[f"{name}_ci_low"] = _empirical_percentile(values, 0.025)
        output[f"{name}_ci_high"] = _empirical_percentile(values, 0.975)
    return output


def _paired_annual_frame(station: Station, raw_dir: Path) -> pd.DataFrame:
    hourly = parse_annual_24h(raw_dir / f"{station.block_no}_annual.html")
    daily = parse_annual_daily_max(raw_dir / f"{station.block_no}_annual_daily.html")
    start_year = 1978 if station.name == "牛久" else 1976
    paired = daily.merge(hourly, on="year", how="inner", validate="one_to_one")
    paired = paired.loc[
        paired["year"].between(start_year, 2014)
        & paired["max_daily_mm"].notna()
        & paired["max_24h_mm"].notna()
    ].copy()
    paired.insert(0, "station", station.name)
    paired.insert(1, "block_no", station.block_no)
    paired["ratio_24h_to_daily"] = paired["max_24h_mm"] / paired["max_daily_mm"]
    paired["strict_quality"] = paired["daily_usable"] & paired["usable"]
    columns = [
        "station",
        "block_no",
        "year",
        "max_daily_mm",
        "raw_daily_value",
        "daily_statistical_break",
        "daily_usable",
        "daily_value_classes",
        "max_24h_mm",
        "date",
        "raw_value",
        "statistical_break",
        "usable",
        "value_classes",
        "ratio_24h_to_daily",
        "strict_quality",
    ]
    return paired[columns].reset_index(drop=True)


def _period_summary(
    station: Station,
    paired: pd.DataFrame,
    event_daily: dict[str, object],
    event_24h: dict[str, object],
    samples: int,
) -> list[dict[str, object]]:
    event_daily_mm = float(event_daily["event_daily_mm"])
    event_24h_mm = float(event_24h["event_24h_mm"])
    rows: list[dict[str, object]] = []
    for variant, strict in (("all_numeric", False), ("strict_quality", True)):
        frame = paired.loc[paired["strict_quality"]].copy() if strict else paired.copy()
        daily_values = frame["max_daily_mm"].to_numpy(float)
        values_24h = frame["max_24h_mm"].to_numpy(float)
        metrics = return_period_metrics(
            daily_values, values_24h, event_daily_mm, event_24h_mm
        )
        intervals = paired_bootstrap_metrics(
            daily_values,
            values_24h,
            event_daily_mm,
            event_24h_mm,
            samples=samples,
            seed=20260000 + int(station.block_no) + int(strict),
        )
        pre = frame.loc[frame["year"] <= 2002, "ratio_24h_to_daily"]
        post = frame.loc[frame["year"] >= 2003, "ratio_24h_to_daily"]
        rows.append(
            {
                "station": station.name,
                "block_no": station.block_no,
                "analysis_variant": variant,
                "start_year": int(frame["year"].min()),
                "end_year": int(frame["year"].max()),
                "n_years": len(frame),
                "event_daily_day": int(event_daily["day"]),
                "event_daily_mm": event_daily_mm,
                "event_24h_mm": event_24h_mm,
                "event_24h_to_daily_ratio": event_24h_mm / event_daily_mm,
                "annual_ratio_median_pre2003": float(pre.median()) if len(pre) else np.nan,
                "annual_ratio_median_post2003": float(post.median()) if len(post) else np.nan,
                **metrics,
                **intervals,
            }
        )
    return rows


def _plot_gev(
    paired_frames: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    output: Path,
) -> None:
    panel_station_names = ("千葉", "茂原", "牛久", "佐倉")
    periods = np.geomspace(1.01, 1000, 500)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), squeeze=False)
    primary = summary.loc[summary["analysis_variant"] == "all_numeric"].set_index("station")
    for panel_index, (ax, station_name) in enumerate(
        zip(axes.ravel(), panel_station_names, strict=True)
    ):
        frame = paired_frames[station_name]
        row = primary.loc[station_name]
        daily_values = frame["max_daily_mm"].to_numpy(float)
        hourly_values = frame["max_24h_mm"].to_numpy(float)
        daily_fit = fit_gev(daily_values)
        hourly_fit = fit_gev(hourly_values)
        ranks = np.arange(1, len(frame) + 1)
        empirical_periods = (len(frame) + 0.12) / (len(frame) - ranks + 0.44)
        ax.scatter(
            empirical_periods,
            np.sort(daily_values),
            s=13,
            alpha=0.75,
            color="#e7a33c",
            label="年最大日降水量",
        )
        ax.scatter(
            empirical_periods,
            np.sort(hourly_values),
            s=13,
            alpha=0.75,
            color="#72a6cf",
            label="年最大24時間降水量",
        )
        ax.plot(
            periods,
            return_level(periods, daily_fit),
            color="#d17c00",
            lw=1.8,
            label="日降水量GEV",
        )
        ax.plot(
            periods,
            return_level(periods, hourly_fit),
            color="#1261a0",
            lw=1.8,
            label="24時間降水量GEV",
        )
        marker_points = (
            (
                float(row["rp_24h_matched"]),
                float(row["event_24h_mm"]),
                "#1261a0",
                "24時間降水量GEV（整合）",
            ),
            (
                float(row["rp_daily_mismatched"]),
                float(row["event_24h_mm"]),
                "#c23b22",
                "日降水量GEV（不整合）",
            ),
        )
        marker_x_values = [
            min(period, 1000) if np.isfinite(period) else 1000
            for period, _, _, _ in marker_points
        ]
        ax.plot(
            marker_x_values,
            [float(row["event_24h_mm"])] * 2,
            color="#777777",
            lw=1.0,
            linestyle="--",
            zorder=4,
        )
        for period, rainfall, color, label in marker_points:
            x_value = min(period, 1000) if np.isfinite(period) else 1000
            ax.scatter(x_value, rainfall, marker="*", s=85, color=color, zorder=5, label=label)
        ax.set_xscale("log")
        ax.set_xlim(1, 1000)
        ax.set_ylim(0, 400)
        ax.set_xticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])
        ax.set_title(
            f"({chr(ord('a') + panel_index)}) {station_name} "
            f"n={len(frame)} ({int(frame['year'].min())}-{int(frame['year'].max())})",
            loc="left",
        )
        ax.set_xlabel("再現期間（年）")
        ax.set_ylabel("降水量（mm）")
        ax.grid(which="both", alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # Matplotlib fills multi-column legends down each column.
    legend_order = [1, 0, 3, 2, 4, 5]
    fig.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output)
    plt.close(fig)


def _plot_ratio(
    paired_frames: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), squeeze=False)
    primary = summary.loc[summary["analysis_variant"] == "all_numeric"].set_index("station")
    for ax, station_name in zip(axes.ravel(), ARTICLE_STATION_NAMES, strict=True):
        frame = paired_frames[station_name]
        row = primary.loc[station_name]
        maximum = max(
            float(frame[["max_daily_mm", "max_24h_mm"]].to_numpy().max()),
            float(row["event_24h_mm"]),
        )
        line = np.array([0.0, maximum * 1.05])
        median_ratio = float(row["annual_ratio_median"])
        ax.scatter(frame["max_daily_mm"], frame["max_24h_mm"], s=20, alpha=0.7)
        ax.plot(line, line, color="#555555", ls="--", label="1:1")
        ax.plot(line, line * median_ratio, color="#d17c00", label=f"中央値 {median_ratio:.3f}")
        ax.scatter(
            float(row["event_daily_mm"]),
            float(row["event_24h_mm"]),
            marker="*",
            s=110,
            color="#c23b22",
            label="2026年豪雨",
            zorder=5,
        )
        ax.set_xlim(0, maximum * 1.05)
        ax.set_ylim(0, maximum * 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(station_name)
        ax.set_xlabel("年最大日降水量（mm）")
        ax.set_ylabel("年最大24時間降水量（mm）")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("年最大日降水量と年最大24時間降水量の対応")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output)
    plt.close(fig)


def run(
    raw_dir: Path,
    results_dir: Path,
    refresh: bool = False,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    """Run the four-station comparison without modifying the blog report."""
    download_pages(ARTICLE_STATIONS, raw_dir, refresh=refresh)
    download_daily_comparison_pages(ARTICLE_STATIONS, raw_dir, refresh=refresh)
    results_dir.mkdir(parents=True, exist_ok=True)
    paired_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for station in ARTICLE_STATIONS:
        paired = _paired_annual_frame(station, raw_dir)
        event_daily = parse_event_daily(raw_dir / f"{station.block_no}_event_daily.html")
        event_24h = parse_event_24h(raw_dir / f"{station.block_no}_event.html")
        paired_frames[station.name] = paired
        summary_rows.extend(
            _period_summary(station, paired, event_daily, event_24h, bootstrap_samples)
        )
    annual = pd.concat(paired_frames.values(), ignore_index=True)
    annual.to_csv(results_dir / "annual_max_daily_vs_24h.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "daily_vs_24h_return_periods.csv", index=False)
    _configure_plotting()
    _plot_gev(paired_frames, summary, results_dir / "daily_vs_24h_gev.png")
    _plot_ratio(paired_frames, summary, results_dir / "daily_vs_24h_ratio.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--refresh", action="store_true", help="redownload JMA snapshots")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    results = run(args.raw_dir, args.results_dir, args.refresh, args.bootstrap)
    columns = [
        "station",
        "analysis_variant",
        "n_years",
        "event_daily_mm",
        "event_24h_mm",
        "rp_24h_matched",
        "rp_daily_mismatched",
        "rp_daily_matched",
        "mismatch_inflation",
    ]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()
