"""Compare GEV fits at the anomalous northern Ichihara cell and its neighbor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

from .extremes import fit_gev, return_level, return_period

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/nusdas_2006_2025"
OUTPUT = ROOT / "results/ichihara_gev_fit_diagnostic.png"
TARGET_LON = 140.10625
TARGET_LAT = 35.5375
NEIGHBOR_LAT = TARGET_LAT - 1 / 120


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


def main() -> None:
    august = DATA_DIR / "nusdas_2006_2025_event_2026.nc"
    september = DATA_DIR / "nusdas_2006_2025_event_20260919_20260922.nc"
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
    for ax, target_lat, title in zip(
        axes,
        (TARGET_LAT, NEIGHBOR_LAT),
        ("(a) 市原市北部の対象格子", "(b) 南隣の格子"),
        strict=True,
    ):
        annual, august_rain, saved_august = _read_cell(august, target_lat)
        annual_september, september_rain, saved_september = _read_cell(september, target_lat)
        if not np.array_equal(annual, annual_september) or not np.isfinite(annual).all():
            raise ValueError("the two events do not share 20 complete annual maxima")
        fit = fit_gev(annual)
        august_period = return_period(august_rain, fit)
        september_period = return_period(september_rain, fit)
        for saved, calculated in (
            (saved_august["shape_xi"], fit.shape_xi),
            (saved_september["shape_xi"], fit.shape_xi),
            (saved_august["return_period_years"], august_period),
            (saved_september["return_period_years"], september_period),
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
        ax.scatter(
            september_period,
            september_rain,
            marker="*",
            s=220,
            color="#e69a23",
            edgecolor="0.2",
            linewidth=0.5,
            zorder=6,
            label=f"2026年9月 {september_period:,.0f}年",
        )
        ax.scatter(
            august_period,
            august_rain,
            marker="*",
            s=220,
            color="#c34636",
            edgecolor="0.2",
            linewidth=0.5,
            zorder=6,
            label=f"2026年8月 {august_period:,.0f}年",
        )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
        summaries.append((fit.shape_xi, august_period, september_period))
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    caption = (
        "# 市原市北部の対象格子と南隣格子におけるGEVフィットの比較\n\n"
        "(a) の格子中心は東経140.10625°、北緯35.5375°。"
        "(b) は南隣の東経140.10625°、北緯35.52917°で、隣接8格子のうち両事例とも"
        "再現期間が最長（8月約485年、9月約996年）。青点は2006–2025年の年最大24時間降水量を"
        "昇順に並べ、report.md図1と同じGringortenプロット位置 "
        "`(n+0.12)/(n-rank+0.44)` に置いた値。青線は同じ20値にLモーメント法で当てはめた"
        "定常GEV分布の再現水準。星は推定に使っていない2026年の事例値。"
        "横軸は推定再現期間まで対数軸を延ばした。"
        "GEV形状母数 ξ は (a) −0.183、(b) −0.140。20年の標本から長い再現期間へ外挿した点推定であり、"
        "図だけで適合度検定の結論は出せない。\n"
    )
    OUTPUT.with_suffix(".md").write_text(caption, encoding="utf-8")
    for title, (xi, august_period, september_period) in zip(
        ("target", "south neighbor"), summaries, strict=True
    ):
        print(f"{title}: xi={xi:.6f}, August={august_period:.1f}, September={september_period:.1f}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
