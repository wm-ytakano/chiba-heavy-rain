"""Compare JMA gauge and nearest RR60 grid-cell return periods for two 2026 events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.io import netcdf_file

from .extremes import GEVFit, fit_gev, return_period
from .jma import USER_AGENT, _url, parse_event_24h
from .nusdas_rain import FILL
from .stations import STATIONS, Station

ROOT = Path(__file__).resolve().parents[2]
COORD_URL = "https://www.data.jma.go.jp/stats/etrn/select/prefecture.php?prec_no=45"
EVENTS = (
    ("2026-08-13_2026-08-15", date(2026, 8, 13), date(2026, 8, 15)),
    ("2026-09-19_2026-09-22", date(2026, 9, 19), date(2026, 9, 22)),
)
GRID_FILES = {
    EVENTS[0][0]: "nusdas_2006_2025_event_2026.nc",
    EVENTS[1][0]: "nusdas_2006_2025_event_20260919_20260922.nc",
}


def station_coordinates(path: Path, stations: tuple[Station, ...]) -> dict[str, tuple[float, float]]:
    """Parse JMA's degree/minute coordinates, keyed by the ETRN block number."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    expected = {station.block_no: station for station in stations}
    found: dict[str, tuple[float, float]] = {}
    for area in soup.find_all("area"):
        block = parse_qs(urlparse(area.get("href", "")).query).get("block_no", [None])[0]
        if block not in expected:
            continue
        match = re.search(r"viewPoint\((.*)\);?", area.get("onmouseover", ""))
        if match is None:
            raise ValueError(f"JMA coordinates missing for {block}")
        fields = next(csv.reader(io.StringIO(match.group(1)), quotechar="'"))
        station = expected[block]
        if fields[0:3] != [station.kind, block, station.name]:
            raise ValueError(f"JMA station identity mismatch: {block}")
        lat = float(fields[4]) + float(fields[5]) / 60
        lon = float(fields[6]) + float(fields[7]) / 60
        if block in found:
            if found[block] != (lat, lon):
                raise ValueError(f"conflicting JMA station coordinates: {block}")
            continue
        found[block] = (lat, lon)
    if set(found) != set(expected):
        raise ValueError(f"JMA station coordinates incomplete: {set(expected) - set(found)}")
    return found


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def nearest_cell(lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> tuple[int, int, float]:
    """Find the nearest center among the four surrounding rectilinear grid centers."""
    yi = np.argsort(np.abs(lats - lat))[:2]
    xi = np.argsort(np.abs(lons - lon))[:2]
    candidates = [
        (_distance_km(lat, lon, float(lats[y]), float(lons[x])), int(y), int(x))
        for y in yi for x in xi
    ]
    distance, y, x = min(candidates)
    return y, x, distance


def _grid_value(value: object) -> float | None:
    number = float(value)
    return None if np.isnan(number) or number == FILL else number


def grid_samples(path: Path, coords: dict[str, tuple[float, float]]) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    with netcdf_file(path, "r", mmap=True) as dataset:
        lats = dataset.variables["lat"][:].copy()
        lons = dataset.variables["lon"][:].copy()
        for block, (lat, lon) in coords.items():
            y, x, distance = nearest_cell(lats, lons, lat, lon)
            rain = _grid_value(dataset.variables["event_max_24h_mm"][y, x])
            period = _grid_value(dataset.variables["return_period_years"][y, x])
            result[block] = {
                "grid_latitude": float(lats[y]), "grid_longitude": float(lons[x]),
                "grid_distance_km": distance,
                "grid_rainfall_mm": rain, "grid_return_period_years": period,
                "grid_status": "ok" if rain is not None and period is not None else "missing",
            }
    return result


def ensure_inputs(raw_dir: Path, stations: tuple[Station, ...], refresh: bool = False) -> None:
    """Cache JMA source pages and merge their URLs and hashes into the raw manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages = [("chiba", "station_coordinates", COORD_URL,
              raw_dir / "stations_chiba_202609.html", None)]
    pages.extend((station.block_no, "event_202609",
                  _url(station, station.daily_page, year=2026, month=9, day="", view="a5"),
                  raw_dir / f"{station.block_no}_event_202609.html", station)
                 for station in stations)
    records = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                      timeout=30.0) as client:
        for block, label, url, path, station in pages:
            if refresh or not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
                time.sleep(0.2)
            payload = path.read_bytes()
            try:
                manifest_path = str(path.relative_to(ROOT))
            except ValueError:
                manifest_path = str(path)
            records.append({
                "block_no": block, "name": station.name if station else "千葉県地点一覧",
                "kind": station.kind if station else "metadata", "table": label,
                "url": url, "path": manifest_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    manifest_path = raw_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    keys = {(record["block_no"], record["table"]) for record in records}
    merged = [record for record in existing
              if (record.get("block_no"), record.get("table")) not in keys] + records
    merged.sort(key=lambda record: (str(record.get("block_no")), str(record.get("table"))))
    manifest_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")


def _window_status(day: int, end_text: str, start: date, end: date) -> str:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", end_text)
    if match is None:
        return "end_time_unknown"
    hour, minute = map(int, match.groups())
    if hour > 24 or minute > 59 or (hour == 24 and minute != 0):
        return "end_time_unknown"
    jst = timezone(timedelta(hours=9))
    ending = datetime(start.year, start.month, day, tzinfo=jst) + timedelta(
        hours=hour, minutes=minute)
    return "within_event" if (
        ending - timedelta(hours=24) >= datetime.combine(start, datetime.min.time(), tzinfo=jst)
        and ending <= datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=jst)
    ) else "crosses_event_boundary"


def build_table(raw_dir: Path, results_dir: Path) -> pd.DataFrame:
    historical = pd.read_csv(results_dir / "historical_1976_2014_return_periods.csv",
                             dtype={"block_no": str})
    august = pd.read_csv(results_dir / "station_return_periods.csv", dtype={"block_no": str})
    annual = pd.read_csv(results_dir / "annual_max_24h.csv", dtype={"block_no": str})
    stations = {station.block_no: station for station in STATIONS}
    eligible = historical.loc[historical["eligible"]]
    if len(eligible) != 13 or eligible.block_no.duplicated().any():
        raise ValueError("expected 13 unique eligible stations")
    coords = station_coordinates(raw_dir / "stations_chiba_202609.html",
                                 tuple(stations[block] for block in eligible.block_no))
    fits = {}
    for block in eligible.block_no:
        sample = annual.loc[(annual.block_no == block) & annual.year.between(2006, 2025)]
        if len(sample) != 20 or set(sample.year) != set(range(2006, 2026)):
            raise ValueError(f"2006-2025 annual sample incomplete: {block}")
        fits[block] = fit_gev(sample.max_24h_mm.to_numpy(float))
    old = historical.set_index("block_no")
    aug = august.set_index("block_no")
    rows = []
    for tag, start, end in EVENTS:
        grid = grid_samples(results_dir / "nusdas_2006_2025" / GRID_FILES[tag], coords)
        for block in eligible.block_no:
            station = stations[block]
            if start.month == 8:
                source = aug.loc[block]
                value = float(source.event_24h_mm)
                day, ending, raw = int(source.day), str(source.ending_time), str(source.raw_value)
                old_period = float(old.loc[block, "return_period_years"])
                if not np.isclose(value, float(old.loc[block, "event_24h_mm"])):
                    raise ValueError(f"August source mismatch: {block}")
            else:
                source = parse_event_24h(raw_dir / f"{block}_event_202609.html",
                                         range(start.day, end.day + 1))
                value = float(source["event_24h_mm"])
                day, ending, raw = source["day"], source["ending_time"], source["raw_value"]
                baseline = old.loc[block]
                old_fit = GEVFit(float(baseline.shape_xi), float(baseline.location),
                                 float(baseline.scale))
                old_period = return_period(value, old_fit)
            lat, lon = coords[block]
            rows.append({
                "event": tag, "station": station.name, "block_no": block,
                "station_latitude": lat, "station_longitude": lon,
                "amedas_24h_mm": value, "amedas_day": day,
                "amedas_end_time_jst": ending, "amedas_raw_value": raw,
                "amedas_quality": "insufficient_data_mark" if "]" in raw else "numeric",
                "amedas_window_status": _window_status(day, ending, start, end),
                "amedas_return_period_1976_2014_years": old_period,
                "amedas_n_years_1976_2014": int(old.loc[block, "n_years"]),
                "amedas_return_period_2006_2025_years": return_period(value, fits[block]),
                "amedas_n_years_2006_2025": 20,
                **grid[block],
            })
    return pd.DataFrame(rows)


def _fmt(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    if math.isinf(float(value)):
        return "∞"
    return f"{float(value):,.{digits}f}"


RANK_COLUMNS = {
    "grid": ("grid_return_period_years", "解析雨量2006–2025年の確率年"),
    "amedas_2006_2025": ("amedas_return_period_2006_2025_years",
                          "アメダス2006–2025年の確率年"),
    "amedas_1976_2014": ("amedas_return_period_1976_2014_years",
                          "アメダス1976–2014年の確率年"),
}


def write_outputs(table: pd.DataFrame, results_dir: Path,
                  rank_by: str = "grid", top_n: int = 10) -> None:
    if rank_by not in RANK_COLUMNS:
        raise ValueError(f"unknown ranking basis: {rank_by}")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    csv_path = results_dir / "event_comparison_2026.csv"
    md_path = results_dir / "event_comparison_2026.md"
    table.to_csv(csv_path, index=False)
    rank_column, rank_label = RANK_COLUMNS[rank_by]
    lines = [
        "# 2026年2事例：アメダスと解析雨量の比較", "",
        f"順位は{rank_label}の高い順。各事例の上位{top_n}地点を表示し、全地点はCSVに収録。", "",
    ]
    for tag, start, end in EVENTS:
        subset = table.loc[table.event == tag].copy()
        ranked = subset.loc[subset[rank_column].notna()].sort_values(
            [rank_column, "block_no"], ascending=[False, True]).head(top_n)
        lines.extend([
            f"## {start.year}年{start.month}月{start.day}日～{end.month}月{end.day}日", "",
            "| 順位 | 観測所 | アメダス (mm) | アメダス1976–14 (年) | アメダス2006–25 (年) | 解析雨量 (mm) | 解析雨量2006–25 (年) | 注記 |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ])
        for rank, row in enumerate(ranked.itertuples(), 1):
            notes = []
            if row.amedas_quality != "numeric":
                notes.append("雨量]")
            if row.amedas_window_status == "end_time_unknown":
                notes.append("終了時刻不明")
            elif row.amedas_window_status == "crosses_event_boundary":
                notes.append("24h窓が期間外に及ぶ")
            if row.grid_status != "ok":
                notes.append("格子欠測")
            lines.append("| " + " | ".join([
                str(rank), row.station,
                _fmt(row.amedas_24h_mm),
                _fmt(row.amedas_return_period_1976_2014_years, 2),
                _fmt(row.amedas_return_period_2006_2025_years, 2),
                _fmt(row.grid_rainfall_mm), _fmt(row.grid_return_period_years, 2),
                "、".join(notes) or "—",
            ]) + " |")
        lines.append("")
    lines.extend([
        "アメダスは各観測所の年最大24時間降水量に定常GEVを適用した点推定。1976–2014年列は既存解析の数値・標本を使用し、牛久・坂畑は37年、その他は39年。2006–2025年列は各地点20年の数値を使用。2026年の事例値は推定標本に含めない。", "",
        "解析雨量はRR60の1時間雨量による24時間最大値を最近傍格子で抽出し、同格子の2006–2025年の年最大値にGEVを適用した点推定。格子値と地上観測は空間代表性が異なる。アメダスの日別最大24時間窓と、解析雨量の事例期間内に完全に収まる窓は必ずしも同一ではない。", "",
        "「雨量]」は気象庁表の資料不足値で、既存解析と同じく掲載された数値を使用。「終了時刻不明」は24時間窓の期間内判定ができない地点。∞はGEV分布の上限を事例雨量が超えた推定値。", "",
        f"出典: [気象庁の観測所座標]({COORD_URL})、気象庁『過去の気象データ検索』の日別・年別表（URLとSHA-256は `data/raw/manifest.json`）、`results/nusdas_2006_2025/` の事例別NetCDF。詳細値・格子中心座標・終了時刻・品質状態は [`{csv_path.name}`]({csv_path.name})。", "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--refresh-inputs", action="store_true")
    parser.add_argument("--rank-by", choices=RANK_COLUMNS, default="grid")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    historical = pd.read_csv(args.results_dir / "historical_1976_2014_return_periods.csv",
                             dtype={"block_no": str})
    blocks = set(historical.loc[historical.eligible, "block_no"])
    ensure_inputs(args.raw_dir, tuple(station for station in STATIONS
                                      if station.block_no in blocks), args.refresh_inputs)
    write_outputs(build_table(args.raw_dir, args.results_dir), args.results_dir,
                  rank_by=args.rank_by, top_n=args.top_n)


if __name__ == "__main__":
    main()
