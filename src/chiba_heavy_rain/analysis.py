"""Command-line workflow for the Chiba August 2026 rainfall analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme

from .extremes import (
    bootstrap_return_period,
    fit_gev,
    return_level,
    return_period,
)
from .jma import (
    download_pages,
    latest_contiguous_complete_run,
    parse_annual_24h,
    parse_event_24h,
)
from .stations import STATIONS


def _format_period(value: float) -> str:
    if not np.isfinite(value):
        return "上限なし"
    if value >= 1000:
        return f"{value:,.0f}年"
    if value >= 100:
        return f"{value:.0f}年"
    if value >= 10:
        return f"{value:.1f}年"
    return f"{value:.2f}年"


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


def _plot_summary(results: pd.DataFrame, output: Path) -> None:
    eligible = results.loc[results["eligible"]].sort_values("return_period_years")
    y = np.arange(len(eligible))
    point = eligible["return_period_years"].to_numpy(float)
    low = eligible["ci_low"].to_numpy(float)
    high = eligible["ci_high"].to_numpy(float)
    finite_high = np.where(np.isfinite(high), high, np.maximum(point * 3, 1e5))
    xerr = np.vstack((np.maximum(point - low, 0), np.maximum(finite_high - point, 0)))

    fig, ax = plt.subplots(figsize=(9, max(5, 0.48 * len(eligible))))
    ax.errorbar(point, y, xerr=xerr, fmt="o", color="#1261a0", ecolor="#72a6cf", capsize=3)
    ax.set_yticks(y, eligible["station"])
    ax.set_xscale("log")
    ax.set_xlabel("GEV return period (years; log scale)")
    ax.set_title("August 2026 event: observed maximum 24-hour rainfall")
    ax.grid(axis="x", which="both", alpha=0.25)
    for yi, (_, row) in zip(y, eligible.iterrows(), strict=True):
        label = _format_period(float(row["return_period_years"]))
        if not np.isfinite(float(row["ci_high"])):
            label += " (CI upper unbounded)"
        ax.annotate(label, (float(row["return_period_years"]), yi), xytext=(6, 3), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_return_levels(series: dict[str, pd.DataFrame], results: pd.DataFrame, output: Path) -> None:
    eligible = results.loc[results["eligible"]].sort_values("station")
    ncols = 3
    nrows = math.ceil(len(eligible) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), squeeze=False)
    periods = np.geomspace(1.05, 1e5, 500)
    for ax, (_, row) in zip(axes.ravel(), eligible.iterrows(), strict=False):
        station = str(row["station"])
        values = series[station]["max_24h_mm"].to_numpy(float)
        fit = fit_gev(values)
        ordered = np.sort(values)
        ranks = np.arange(1, len(ordered) + 1)
        empirical_period = (len(ordered) + 0.12) / (len(ordered) - ranks + 0.44)
        ax.scatter(empirical_period, ordered, s=12, alpha=0.7, color="#555555", label="annual maxima")
        ax.plot(periods, return_level(periods, fit), color="#1261a0", lw=1.8, label="GEV MLE")
        ax.scatter(
            [float(row["return_period_years"])],
            [float(row["event_24h_mm"])],
            marker="*",
            s=95,
            color="#c23b22",
            zorder=5,
            label="Aug 2026 event",
        )
        ax.set_xscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_title(f"{station}  n={int(row['n_years'])} ({int(row['start_year'])}–2025)")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("24-hour rainfall (mm)")
        ax.grid(which="both", alpha=0.2)
    unused_axes = axes.ravel()[len(eligible) :]
    for ax in unused_axes:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if len(unused_axes):
        unused_axes[0].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Observed annual maxima and stationary GEV fits", y=0.998, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output)
    plt.close(fig)


def run(raw_dir: Path, results_dir: Path, refresh: bool, bootstrap_samples: int) -> pd.DataFrame:
    download_pages(STATIONS, raw_dir, refresh=refresh)
    results_dir.mkdir(parents=True, exist_ok=True)
    station_results: list[dict[str, object]] = []
    historical_series: dict[str, pd.DataFrame] = {}
    all_annual: list[pd.DataFrame] = []

    for index, station in enumerate(STATIONS):
        annual = parse_annual_24h(raw_dir / f"{station.block_no}_annual.html")
        annual.insert(0, "station", station.name)
        annual.insert(1, "block_no", station.block_no)
        all_annual.append(annual)
        history = latest_contiguous_complete_run(annual, end_year=2025)
        event = parse_event_24h(raw_dir / f"{station.block_no}_event.html")
        span_years = (
            int(history["year"].max() - history["year"].min() + 1) if len(history) else 0
        )
        eligible = len(history) >= 40 and span_years >= 40
        base: dict[str, object] = {
            "station": station.name,
            "block_no": station.block_no,
            "kind": station.kind,
            "start_year": int(history["year"].min()) if len(history) else pd.NA,
            "end_year": int(history["year"].max()) if len(history) else pd.NA,
            "n_years": len(history),
            "span_years": span_years,
            "eligible": eligible,
            **event,
        }
        if eligible:
            values = history["max_24h_mm"].to_numpy(float)
            fit = fit_gev(values)
            period = return_period(float(event["event_24h_mm"]), fit)
            full_fit = fit_gev(np.append(values, float(event["event_24h_mm"])))
            uncertainty = bootstrap_return_period(
                values,
                float(event["event_24h_mm"]),
                samples=bootstrap_samples,
                seed=20260813 + index,
            )
            log_likelihood = float(
                np.sum(genextreme.logpdf(values, fit.scipy_shape, loc=fit.location, scale=fit.scale))
            )
            base.update(
                {
                    "historical_max_mm": float(np.max(values)),
                    "historical_exceedances": int(np.sum(values >= float(event["event_24h_mm"]))),
                    "shape_xi": fit.shape_xi,
                    "location": fit.location,
                    "scale": fit.scale,
                    "aic": 6 - 2 * log_likelihood,
                    "return_period_years": period,
                    "return_period_including_2026": return_period(
                        float(event["event_24h_mm"]), full_fit
                    ),
                    **uncertainty,
                }
            )
            historical_series[station.name] = history
        station_results.append(base)

    results = pd.DataFrame(station_results)
    results.to_csv(results_dir / "station_return_periods.csv", index=False)
    pd.concat(all_annual, ignore_index=True).to_csv(
        results_dir / "annual_max_24h.csv", index=False
    )
    _configure_plotting()
    _plot_summary(results, results_dir / "return_period_summary.png")
    _plot_return_levels(historical_series, results, results_dir / "return_level_diagnostics.png")
    _write_report(results, results_dir / "report.md", bootstrap_samples)
    return results


def render_existing(results_dir: Path, bootstrap_samples: int) -> pd.DataFrame:
    """Regenerate report and figures without repeating the bootstrap calculation."""
    results = pd.read_csv(results_dir / "station_return_periods.csv")
    annual = pd.read_csv(results_dir / "annual_max_24h.csv")
    historical_series: dict[str, pd.DataFrame] = {}
    for row in results.loc[results["eligible"]].itertuples():
        historical_series[row.station] = annual.loc[
            (annual["station"] == row.station)
            & annual["year"].between(int(row.start_year), 2025)
            & annual["usable"]
        ].copy()
    _configure_plotting()
    _plot_summary(results, results_dir / "return_period_summary.png")
    _plot_return_levels(
        historical_series, results, results_dir / "return_level_diagnostics.png"
    )
    _write_report(results, results_dir / "report.md", bootstrap_samples)
    return results


def _write_report(results: pd.DataFrame, output: Path, bootstrap_samples: int) -> None:
    eligible = results.loc[results["eligible"]].copy()
    excluded = results.loc[~results["eligible"]].copy()
    eligible = eligible.sort_values("return_period_years", ascending=False)
    rows = []
    for _, row in eligible.iterrows():
        ci = f"{_format_period(float(row['ci_low']))} – {_format_period(float(row['ci_high']))}"
        rows.append(
            "| {station} | {years} | {rain:.1f} | {hist:.1f} | {period} | {ci} | {included} |".format(
                station=row["station"],
                years=f"{int(row['start_year'])}–2025 ({int(row['n_years'])}年)",
                rain=float(row["event_24h_mm"]),
                hist=float(row["historical_max_mm"]),
                period=_format_period(float(row["return_period_years"])),
                ci=ci,
                included=_format_period(float(row["return_period_including_2026"])),
            )
        )
    excluded_text = "、".join(
        f"{row.station}（連続{int(row.n_years)}年）" for row in excluded.itertuples()
    ) or "なし"
    extreme_count = int((eligible["return_period_years"] > 10000).sum())
    max_row = eligible.iloc[0]
    text = f"""# 令和8年8月千葉豪雨：地上観測に基づく24時間雨量の確率年

## 結論

気象庁の千葉県内地上観測のうち、JMAの統計切断表示後の同一系列が40年以上あり、利用可能な年最大24時間降水量を40年分以上持つ地点を対象に、定常GEV分布を最尤推定した。2026年8月13～15日の観測最大24時間雨量を、事象から独立な2025年までの分布に照らした主解析では、最大の点推定は **{max_row['station']}の{_format_period(float(max_row['return_period_years']))}** だった。1万年を超えた地点は **{extreme_count}地点** である。

ただし、記事の574.2 mmは千葉市緑区～市原市東部の「解析雨量」（格子値）であり、地上観測所の値ではない。今回の地上観測解析は、その574.2 mm自体の確率年を直接検証するものではなく、同じ豪雨を既存観測所で捉えた場合の局地ごとの頻度を示す。両者の空間代表性の違いを無視した一対一比較はできない。

## 地点別結果

主解析の「確率年」は2025年まででGEVを推定した値。95%区間は年最大値を地点ごとに再標本化した非パラメトリック・ブートストラップ（{bootstrap_samples:,}回）のパーセンタイル区間。「2026含む」は今回値を標本に追加して再推定した感度分析である。

| 地点 | 学習期間 | 今回24h (mm) | 既往最大 (mm) | 確率年 | 95%ブートストラップ区間 | 2026含む |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

対象外：{excluded_text}

![地点別確率年](return_period_summary.png)

![GEV診断](return_level_diagnostics.png)

## 方法と再現性

- 出典は気象庁「過去の気象データ検索」の年ごとの値（詳細・N時間降水量）と2026年8月の日ごとの値。取得URLとSHA-256は `data/raw/manifest.json` に保存した。
- 気象庁HTMLの `data_2t_*` 表示を統計切断として扱い、その後の最新系列が40年以上かつ利用可能な年最大値40個以上の地点を適格とした。資料不足値 `]` は除外したが、それ自体は統計切断とはみなしていない。依頼に従い測器・観測方法の変更補正は行っていない。
- GEVの形状母数は通常の記法 ξ（SciPyの `genextreme` の形状母数とは符号が逆）。最尤推定を用いた。
- 主解析は今回事象を分布推定から除外し、推定対象と評価対象を分離した。感度分析では今回値を1年追加した。
- 再現期間は `T = 1 / (1 - F(x))`。これは「T年ごとに規則的に起きる」の意味ではなく、定常性を仮定した年超過確率の逆数である。

再実行：

```bash
uv sync
uv run chiba-rain --bootstrap {bootstrap_samples}
```

## 解釈上の限界

40～数十年の標本から数百～数千年以上へ外挿すると、GEV形状母数のわずかな違いで確率年が桁違いになる。区間上限が「上限なし」の地点は、ブートストラップ標本の2.5%以上で生存確率が数値上ゼロまたは分布上端を超えたことを表す。したがって極端に長い点推定は精密な暦年数ではなく、「観測期間よりはるかに稀」という定性的証拠として扱うべきである。

また、気候の非定常性を無視した定常GEVであり、温暖化の寄与率は評価していない。観測とモデルは目的が異なるため、観測GEVだけで気候シミュレーションを否定することも、39年間のモデル時系列だけで極端な再現期間を確定することもできない。客観的な比較には、解析雨量を観測所位置に抽出した値との整合確認、モデルの観測再現性評価、アンサンブル数・バイアス補正・極値推定の不確実性開示が必要である。

## 参照資料

- [ウェザーニュース：千葉県で500mm超の記録的大雨](https://weathernews.jp/news/202608/140251/)
- [気象庁：過去の気象データ検索（千葉県の地点選択）](https://www.data.jma.go.jp/stats/etrn/select/prefecture.php?prec_no=45)
- [銚子地方気象台：千葉県における気温・降水量の経年変化](https://www.data.jma.go.jp/choshi/shosai/bousai/history_rain.html)
- [気象庁：気象観測データの品質と均質性](https://www.jma.go.jp/jma/kishou/know/stats/dounyu_3.html)
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--refresh", action="store_true", help="redownload JMA snapshots")
    parser.add_argument("--bootstrap", type=int, default=500, help="bootstrap replicates per station")
    parser.add_argument(
        "--render-only", action="store_true", help="reuse existing CSV results to render outputs"
    )
    args = parser.parse_args()
    if args.render_only:
        results = render_existing(args.results_dir, args.bootstrap)
    else:
        results = run(args.raw_dir, args.results_dir, args.refresh, args.bootstrap)
    columns = [
        "station",
        "n_years",
        "event_24h_mm",
        "return_period_years",
        "ci_low",
        "ci_high",
    ]
    print(results.loc[results["eligible"], columns].to_string(index=False))


if __name__ == "__main__":
    main()
