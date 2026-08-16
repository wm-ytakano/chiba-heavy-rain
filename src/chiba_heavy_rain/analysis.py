"""Command-line workflow for the Chiba August 2026 rainfall analysis."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

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

ARTICLE_STATION_LABELS = {
    "千葉": "記事地域：千葉市中央区～若葉区付近",
    "茂原": "記事地域：茂原市付近",
    "牛久": "記事地域：市原市中部",
    "佐倉": "記事地域：八千代・佐倉付近",
}


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


def _year_month_label(row: pd.Series) -> str:
    match = re.search(r"(\d{1,2})/", str(row["date"]))
    if match is None:
        return f"{int(row['year'])}年"
    return f"{int(row['year'])}年{int(match.group(1))}月"


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
        ax.annotate(
            label,
            (float(row["return_period_years"]), yi),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_return_levels(
    series: dict[str, pd.DataFrame],
    results: pd.DataFrame,
    output: Path,
    title: str = "Observed annual maxima and stationary GEV L-moment fits",
    annotate_maximum: bool = False,
) -> None:
    eligible = results.loc[results["eligible"]].sort_values("station")
    ncols = 2 if len(eligible) <= 4 else 3
    nrows = math.ceil(len(eligible) / ncols)
    figure_width = 11 if ncols == 2 else 13
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, 4.2 * nrows), squeeze=False)
    periods = np.geomspace(1.05, 1000, 500)
    for ax, (_, row) in zip(axes.ravel(), eligible.iterrows(), strict=False):
        station = str(row["station"])
        values = series[station]["max_24h_mm"].to_numpy(float)
        fit = fit_gev(values)
        ordered = np.sort(values)
        ranks = np.arange(1, len(ordered) + 1)
        empirical_period = (len(ordered) + 0.12) / (len(ordered) - ranks + 0.44)
        ax.scatter(
            empirical_period, ordered, s=12, alpha=0.7, color="#555555", label="annual maxima"
        )
        if annotate_maximum:
            maximum_row = series[station].loc[series[station]["max_24h_mm"].idxmax()]
            ax.annotate(
                _year_month_label(maximum_row),
                xy=(float(empirical_period[-1]), float(ordered[-1])),
                xytext=(7, -18),
                textcoords="offset points",
                fontsize=9,
                ha="left",
                arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
            )
        ax.plot(
            periods,
            return_level(periods, fit),
            color="#1261a0",
            lw=1.8,
            label="GEV L-moments",
        )
        ax.scatter(
            [float(row["return_period_years"])],
            [float(row["event_24h_mm"])],
            marker="*",
            s=95,
            color="#c23b22",
            zorder=5,
            label="JMA gauge, Aug 2026",
        )
        ax.set_xscale("log")
        ax.set_xlim(1, 1000)
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
        # Keep short-record shape estimates from making the observed range unreadable.
        # The fitted curve may leave the panel at long return periods; that behavior is
        # itself a warning about extrapolation instability.
        observed_cap = max(float(np.max(ordered)), float(row["event_24h_mm"]))
        ax.set_ylim(0, observed_cap * 1.35)
        station_label = str(row.get("display_label", station))
        ax.set_title(
            f"{station_label}\nJMA {station}  n={int(row['n_years'])} "
            f"({int(row['start_year'])}–{int(row['end_year'])})"
        )
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("24-hour rainfall (mm)")
        ax.grid(which="both", alpha=0.2)
    unused_axes = axes.ravel()[len(eligible) :]
    for ax in unused_axes:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, y=0.998, fontsize=15)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(output)
    plt.close(fig)


def _historical_period_results(
    results: pd.DataFrame,
    annual: pd.DataFrame,
    start_year: int = 1976,
    end_year: int = 2014,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit the same stations over the 39 years centered on 1995.

    This deliberately applies no usable/quality filter to numeric annual maxima,
    as requested. It also does not re-screen stations by record length.
    """
    rows: list[dict[str, object]] = []
    series: dict[str, pd.DataFrame] = {}
    for station_row in results.loc[results["eligible"]].itertuples():
        frame = annual.loc[
            (annual["station"] == station_row.station)
            & annual["year"].between(start_year, end_year)
            & annual["max_24h_mm"].notna()
        ].sort_values("year")
        values = frame["max_24h_mm"].to_numpy(float)
        fit = fit_gev(values)
        event_value = float(station_row.event_24h_mm)
        rows.append(
            {
                "station": station_row.station,
                "block_no": station_row.block_no,
                "start_year": int(frame["year"].min()),
                "end_year": int(frame["year"].max()),
                "n_years": len(frame),
                "eligible": True,
                "window_basis": "inferred_from_article_not_confirmed_by_source_description",
                "event_24h_mm": event_value,
                "historical_max_mm": float(values.max()),
                "shape_xi": fit.shape_xi,
                "location": fit.location,
                "scale": fit.scale,
                "return_period_years": return_period(event_value, fit),
            }
        )
        series[str(station_row.station)] = frame.copy()
    return pd.DataFrame(rows), series


def _article_location_subset(
    results: pd.DataFrame, series: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Select eligible gauges directly corresponding to locations named in the article."""
    selected = results.loc[
        results["eligible"] & results["station"].isin(ARTICLE_STATION_LABELS)
    ].copy()
    selected["display_label"] = selected["station"].map(ARTICLE_STATION_LABELS)
    selected_series = {station: series[station] for station in selected["station"]}
    return selected, selected_series


def _plot_period_comparison(current: pd.DataFrame, historical: pd.DataFrame, output: Path) -> None:
    merged = current.loc[current["eligible"], ["station", "return_period_years"]].merge(
        historical[["station", "return_period_years"]],
        on="station",
        suffixes=("_through_2025", "_1976_2014"),
    )
    merged = merged.sort_values("return_period_years_1976_2014")
    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.48 * len(merged))))
    ax.scatter(
        merged["return_period_years_through_2025"],
        y - 0.12,
        label="Fit through 2025",
        color="#777777",
        s=35,
    )
    ax.scatter(
        merged["return_period_years_1976_2014"],
        y + 0.12,
        label="Inferred 1976–2014 fit (exact WNI window undisclosed)",
        color="#c23b22",
        marker="D",
        s=38,
    )
    ax.set_yticks(y, merged["station"])
    ax.set_xscale("log")
    ax.set_xlabel("Return period of the August 2026 observed value (years; log scale)")
    ax.set_title("Effect of an article-implied historical window")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
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
        span_years = int(history["year"].max() - history["year"].min() + 1) if len(history) else 0
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
            base.update(
                {
                    "historical_max_mm": float(np.max(values)),
                    "historical_exceedances": int(np.sum(values >= float(event["event_24h_mm"]))),
                    "shape_xi": fit.shape_xi,
                    "location": fit.location,
                    "scale": fit.scale,
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
    annual_frame = pd.concat(all_annual, ignore_index=True)
    annual_frame.to_csv(results_dir / "annual_max_24h.csv", index=False)
    period_results, period_series = _historical_period_results(results, annual_frame)
    article_current_results, article_current_series = _article_location_subset(
        results, historical_series
    )
    article_results, article_series = _article_location_subset(period_results, period_series)
    period_results.to_csv(results_dir / "historical_1976_2014_return_periods.csv", index=False)
    _configure_plotting()
    _plot_summary(results, results_dir / "return_period_summary.png")
    _plot_return_levels(
        article_current_series,
        article_current_results,
        results_dir / "return_level_diagnostics.png",
        title="Article-mentioned locations: JMA annual maxima through 2025",
    )
    _plot_return_levels(
        article_series,
        article_results,
        results_dir / "historical_1976_2014_return_levels.png",
        title="Article-mentioned locations: JMA annual maxima, 1976–2014",
        annotate_maximum=True,
    )
    _plot_period_comparison(
        results, period_results, results_dir / "historical_period_comparison.png"
    )
    _write_report(results, results_dir / "report.md", bootstrap_samples, period_results)
    return results


def render_existing(results_dir: Path, bootstrap_samples: int) -> pd.DataFrame:
    """Regenerate report and figures without repeating the bootstrap calculation."""
    results = pd.read_csv(results_dir / "station_return_periods.csv", dtype={"block_no": str})
    annual = pd.read_csv(results_dir / "annual_max_24h.csv")
    historical_series: dict[str, pd.DataFrame] = {}
    for row in results.loc[results["eligible"]].itertuples():
        historical_series[row.station] = annual.loc[
            (annual["station"] == row.station)
            & annual["year"].between(int(row.start_year), 2025)
            & annual["usable"]
        ].copy()
    period_results, period_series = _historical_period_results(results, annual)
    article_current_results, article_current_series = _article_location_subset(
        results, historical_series
    )
    article_results, article_series = _article_location_subset(period_results, period_series)
    period_results.to_csv(results_dir / "historical_1976_2014_return_periods.csv", index=False)
    _configure_plotting()
    _plot_summary(results, results_dir / "return_period_summary.png")
    _plot_return_levels(
        article_current_series,
        article_current_results,
        results_dir / "return_level_diagnostics.png",
        title="Article-mentioned locations: JMA annual maxima through 2025",
    )
    _plot_return_levels(
        article_series,
        article_results,
        results_dir / "historical_1976_2014_return_levels.png",
        title="Article-mentioned locations: JMA annual maxima, 1976–2014",
        annotate_maximum=True,
    )
    _plot_period_comparison(
        results, period_results, results_dir / "historical_period_comparison.png"
    )
    _write_report(results, results_dir / "report.md", bootstrap_samples, period_results)
    return results


def _write_report(
    results: pd.DataFrame,
    output: Path,
    bootstrap_samples: int,
    period_results: pd.DataFrame | None = None,
) -> None:
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
    excluded_text = (
        "、".join(f"{row.station}（連続{int(row.n_years)}年）" for row in excluded.itertuples())
        or "なし"
    )
    max_row = eligible.iloc[0]
    period_rows: list[str] = []
    period_max_station = ""
    period_max_value = float("nan")
    if period_results is not None:
        ordered_period = period_results.loc[
            period_results["station"].isin(ARTICLE_STATION_LABELS)
        ].sort_values("return_period_years", ascending=False)
        for _, row in ordered_period.iterrows():
            period_rows.append(
                "| {area} | {station} | {years} | {n} | {rain:.1f} | {hist:.1f} | {period} |".format(
                    area=ARTICLE_STATION_LABELS[str(row["station"])].removeprefix("記事地域："),
                    station=row["station"],
                    years=f"{int(row['start_year'])}–{int(row['end_year'])}",
                    n=int(row["n_years"]),
                    rain=float(row["event_24h_mm"]),
                    hist=float(row["historical_max_mm"]),
                    period=_format_period(float(row["return_period_years"])),
                )
            )
        period_max = ordered_period.iloc[0]
        period_max_station = str(period_max["station"])
        period_max_value = float(period_max["return_period_years"])
    text = f"""# 令和8年8月千葉豪雨：地上観測に基づく24時間雨量の確率年

## 結論

記事文言から推定した **1976～2014年** を主解析期間とし、元記事の地域に直接対応づけられるJMA観測所だけを表示した。標本Lモーメント（PWM）による定常GEVの最大点推定は **{period_max_station}の{_format_period(period_max_value)}** だった。記事対応4地点に1万年を超える推定はない。

ただし、記事の574.2 mmは千葉市緑区～市原市東部の「解析雨量」（格子値）であり、地上観測所の値ではない。今回の地上観測解析は、その574.2 mm自体の確率年を直接検証するものではなく、同じ豪雨を既存観測所で捉えた場合の局地ごとの頻度を示す。両者の空間代表性の違いを無視した一対一比較はできない。

## 主解析：記事言及地域、1976～2014年

**注意：1976～2014年は元データdescriptionに明記された期間ではない。** 記事の「1995年を中心とする約39年間」を中央年の前後19年と読み、かつCMIP6 historical runが2014年で終わることから逆算した条件付きの推定である。ウェザーニューズが極値統計に実際に切り出した開始・終了年は、記事にも元データdescriptionにも記載されていない。

元データdescriptionを確認すると、NASA NEX-GDDP-CMIP6はhistoricalが1950～2014年、SSPが2015～2100年である。NIES2020で「39年」と明記されるのは極値統計の標本期間ではなく、CDFDMバイアス補正の基準期間 **1980～2018年** で、その内訳はhistorical runの1980～2014年とSSP585 runの2015～2018年である。この1980～2018年を記事の「1995年中心の過去気候期間」と読み替える根拠はないため、本図には採用していない。

期間を固定し、数値が掲載されている年最大24時間雨量を品質記号 `]` も含めて使用した。牛久は観測掲載開始の関係で37値、ほか3地点は39値である。図の赤星は記事の解析雨量ではなく、対応するJMA地上観測所の2026年8月の観測値である。

| 記事の地域 | JMA観測所 | 使用期間 | 年最大値数 | 今回観測値 (mm) | 期間内最大 (mm) | 確率年 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(period_rows)}

![記事言及地域の1976～2014年GEV診断](historical_1976_2014_return_levels.png)

記事言及地域のうち、千葉市緑区～市原市東部、東金・大網白里、長南・長柄、野田・流山には、今回の選定条件を満たし同一地域と直接みなせるJMA観測所がない。近隣観測所を恣意的に代理せず、フィッティング曲線から除外した。全13適格地点の数値は `historical_1976_2014_return_periods.csv` に残している。

このGEV入力はブロック最大法としての年最大値である。「フィルタなし」は、年最大値の品質記号による除外と地点の再選別を行わない、という意味で実装した。日々の24時間雨量をすべてGEVへ投入するPOT解析ではない。

## 感度分析：各地点の最新連続系列～2025年

記事地域に限定しない全13適格地点について、2025年までの最新連続系列で再推定した。最大点推定は **{max_row["station"]}の{_format_period(float(max_row["return_period_years"]))}** だった。95%区間は年最大値を地点ごとに再標本化した非パラメトリック・ブートストラップ（{bootstrap_samples:,}回）のパーセンタイル区間。「2026含む」は今回値を標本に追加した感度分析である。

| 地点 | 学習期間 | 今回24h (mm) | 既往最大 (mm) | 確率年 | 95%ブートストラップ区間 | 2026含む |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

対象外：{excluded_text}

![地点別確率年](return_period_summary.png)

![記事言及地域の最新系列GEV診断](return_level_diagnostics.png)

![期間による確率年の比較](historical_period_comparison.png)

## 方法と再現性

- 出典は気象庁「過去の気象データ検索」の年ごとの値（詳細・N時間降水量）と2026年8月の日ごとの値。取得URLとSHA-256は `data/raw/manifest.json` に保存した。
- 気象庁HTMLの `data_2t_*` 表示を統計切断として扱い、その後の最新系列が40年以上かつ利用可能な年最大値40個以上の地点を適格とした。資料不足値 `]` は除外したが、それ自体は統計切断とはみなしていない。依頼に従い測器・観測方法の変更補正は行っていない。
- GEV母数は標本の確率重み付きモーメントからL1、L2、L-skewnessを計算し、理論L-skewnessを数値的に逆算して推定した。形状母数は通常の記法 ξ（SciPyの `genextreme` の形状母数とは符号が逆）。
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
- [NASA NCCS：NEX-GDDP-CMIP6 dataset description](https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/)
- [国立環境研究所：NIES2020 dataset description](https://www.nies.go.jp/doi/10.17595/20210501.001.html)
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
    parser.add_argument(
        "--bootstrap", type=int, default=500, help="bootstrap replicates per station"
    )
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
