"""Compare GEV fits at the anomalous northern Ichihara cell and its neighbor."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

from .extremes import bootstrap_return_period, fit_gev, return_level, return_period

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/nusdas_2006_2025"
OUTPUT = ROOT / "results/ichihara_gev_fit_diagnostic.png"
TARGET_LON = 140.10625
TARGET_LAT = 35.5375
NEIGHBOR_LAT = TARGET_LAT - 1 / 120
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260813


def _read_cell(path: Path, target_lat: float) -> tuple[np.ndarray, float, dict[str, float]]:
    # Slice only one cell from the large NetCDF file.
    with netcdf_file(path, "r", mmap=False) as dataset:
        lat = np.asarray(dataset.variables["lat"][:])
        lon = np.asarray(dataset.variables["lon"][:])
        iy = int(np.argmin(abs(lat - target_lat)))
        ix = int(np.argmin(abs(lon - TARGET_LON)))
        if abs(lat[iy] - target_lat) > 0.001 or abs(lon[ix] - TARGET_LON) > 0.001:
            raise ValueError("target grid cell not found")
        annual = np.array(dataset.variables["annual_max_24h_mm"][:, iy, ix], dtype=float)
        event = float(dataset.variables["event_max_24h_mm"][iy, ix])
        fitted = {
            key: float(dataset.variables[key][iy, ix])
            for key in ("shape_xi", "location", "scale", "return_period_years")
        }
    return annual, event, fitted


def _bootstrap_level_band(annual: np.ndarray, periods: np.ndarray) -> np.ndarray:
    # Same seed and draw sequence as bootstrap_return_period, so both use one resample set.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    levels = []
    for _ in range(BOOTSTRAP_SAMPLES):
        draw = rng.choice(annual, size=len(annual), replace=True)
        try:
            levels.append(return_level(periods, fit_gev(draw)))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    return np.percentile(np.asarray(levels), [2.5, 97.5], axis=0)


def _format_period(value: float) -> str:
    if not np.isfinite(value):
        return "∞"
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def main() -> None:
    august = DATA_DIR / "nusdas_2006_2025_event_2026.nc"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"],
            "font.size": 13,
            "axes.unicode_minus": False,
        }
    )
    periods = np.geomspace(1.01, 1e6, 800)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180, constrained_layout=True)
    summaries = []
    rows = []
    for ax, target_lat, title in zip(
        axes,
        (TARGET_LAT, NEIGHBOR_LAT),
        ("(a) 市原市北部の対象格子", "(b) 南隣の格子"),
        strict=True,
    ):
        annual, august_rain, saved_august = _read_cell(august, target_lat)
        if not np.isfinite(annual).all():
            raise ValueError("the cell does not have 20 complete annual maxima")
        fit = fit_gev(annual)
        august_period = return_period(august_rain, fit)
        for saved, calculated in (
            (saved_august["shape_xi"], fit.shape_xi),
            (saved_august["return_period_years"], august_period),
        ):
            if not np.isclose(saved, calculated, rtol=1e-4, atol=1e-5):
                raise ValueError(f"saved fit differs from recalculation: {saved} vs {calculated}")

        ordered = np.sort(annual)
        n = len(ordered)
        ranks = np.arange(1, n + 1)
        empirical_periods = (n + 0.12) / (n - ranks + 0.44)
        ax.scatter(
            empirical_periods,
            ordered,
            s=48,
            color="#6699bd",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="年最大24時間降水量（20年）",
        )
        band = _bootstrap_level_band(annual, periods)
        ax.fill_between(
            periods,
            band[0],
            band[1],
            color="#6699bd",
            alpha=0.2,
            linewidth=0,
            zorder=1,
            label=f"GEVの95%区間（ブートストラップ{BOOTSTRAP_SAMPLES:,}回）",
        )
        ax.plot(
            periods,
            return_level(periods, fit),
            color="#145e96",
            lw=2.5,
            label="GEV（Lモーメント法）",
        )
        ax.set_xscale("log")
        ax.set_xlim(1, 1e6)
        ax.set_ylim(50, 390)
        ax.set_title(title, loc="left", fontsize=17)
        ax.set_xlabel("再現期間（年）", fontsize=15)
        ax.set_ylabel("24時間降水量（mm）", fontsize=15)
        ax.grid(which="major", color="0.78", linewidth=0.8)
        ax.grid(which="minor", axis="x", color="0.89", linewidth=0.5)
        ax.tick_params(
            axis="both", which="major", direction="out", length=6, width=1.2, labelsize=12
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.15")
            spine.set_linewidth(1.2)
        ax.set_xticks(
            [1, 10, 100, 1000, 10000, 100000, 1000000],
            labels=["1", "10", "100", "千", "1万", "10万", "100万"],
        )
        summaries.append((fit.shape_xi, august_period))
        # A negative shape gives a finite GEV upper endpoint.
        upper_bound = fit.location - fit.scale / fit.shape_xi if fit.shape_xi < 0 else np.inf
        x_max = ax.get_xlim()[1]
        for event, month, rain, period, color in (
            ("2026-08-13_2026-08-15", "8月", august_rain, august_period, "#c34636"),
        ):
            interval = bootstrap_return_period(annual, rain, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            low, high = interval["ci_low"], interval["ci_high"]
            # An infinite upper limit runs to the axis edge and ends in an arrowhead.
            ax.plot(
                [low, min(high, x_max)],
                [rain, rain],
                color=color,
                lw=2,
                marker="|",
                markevery=[0],
                markersize=10,
                markeredgewidth=2,
                zorder=5,
            )
            if not np.isfinite(high):
                ax.plot(x_max, rain, marker=">", color=color, markersize=8, clip_on=False, zorder=5)
            ax.scatter(
                period,
                rain,
                marker="*",
                s=220,
                color=color,
                edgecolor="0.2",
                linewidth=0.5,
                zorder=6,
                label=(
                    f"2026年{month} {_format_period(period)}年"
                    f"（95%区間 {_format_period(low)}～{_format_period(high)}年）"
                ),
            )
            rows.append(
                {
                    "cell": "target" if target_lat == TARGET_LAT else "south_neighbor",
                    "latitude": round(target_lat, 5),
                    "longitude": TARGET_LON,
                    "event": event,
                    "event_max_24h_mm": round(rain, 1),
                    "sample_max_24h_mm": round(float(annual.max()), 1),
                    "shape_xi": fit.shape_xi,
                    "location": fit.location,
                    "scale": fit.scale,
                    "upper_bound_mm": upper_bound,
                    "return_period_years": period,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    **interval,
                }
            )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    caption = (
        "# 市原市北部の対象格子と南隣格子におけるGEVフィットの比較\n\n"
        "(a) の格子中心は東経140.10625°、北緯35.5375°。"
        "(b) は南隣の東経140.10625°、北緯35.52917°で、隣接8格子のうち"
        "再現期間が最長（約485年）。青点は2006–2025年の年最大24時間降水量を"
        "昇順に並べ、report.md図1と同じGringortenプロット位置 "
        "`(n+0.12)/(n-rank+0.44)` に置いた値。青線は同じ20値にLモーメント法で当てはめた"
        "定常GEV分布の再現水準、薄青の帯は20値を復元抽出して2,000回再推定したブートストラップの"
        "95%区間。星は推定に使っていない2026年8月13～15日の事例値で、横線はその再現期間の95%区間"
        "（右端の矢印は上限が∞）。"
        "横軸は推定再現期間まで対数軸を延ばした。"
        "GEV形状母数 ξ は (a) −0.183、(b) −0.140。20年の標本から長い再現期間へ外挿した点推定であり、"
        "図だけで適合度検定の結論は出せない。\n"
    )
    OUTPUT.with_suffix(".md").write_text(caption, encoding="utf-8")
    with OUTPUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for title, (xi, august_period) in zip(("target", "south neighbor"), summaries, strict=True):
        print(f"{title}: xi={xi:.6f}, August={august_period:.1f}")
    for row in rows:
        print(
            f"{row['cell']} {row['event']}: upper={row['upper_bound_mm']:.1f} mm, "
            f"95% CI={row['ci_low']:.1f}-{row['ci_high']:.1f}, "
            f"infinite={row['bootstrap_infinite']}/{row['bootstrap_valid']}"
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
