"""Rank Chiba municipalities by analyzed-rainfall maxima and return periods."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file
from shapely import contains_xy
from shapely.geometry import shape

from .nusdas_rain import DEFAULT_CITY_SHP, EVENT_START, FILL, _event_names

JST = timezone(timedelta(hours=9))


def load_municipalities(city_shp: Path) -> list[tuple[str, str, object]]:
    """Read the 54 Chiba municipal polygons from the supplied GIS data."""
    completed = subprocess.run(
        ["ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(city_shp),
         "-where", "regioncode LIKE '12%'"],
        check=True, capture_output=True, text=True,
    )
    municipalities = []
    for feature in json.loads(completed.stdout)["features"]:
        properties = feature["properties"]
        code = str(properties.get("regioncode") or "")
        name = properties.get("name")
        if code.startswith("12") and name and feature["geometry"] is not None:
            municipalities.append((code, name, shape(feature["geometry"])))
    if len({code for code, _, _ in municipalities}) != len(municipalities):
        raise ValueError("duplicate Chiba municipality codes")
    return sorted(municipalities)


def _selected_cell(values: np.ndarray, inside: np.ndarray) -> tuple[int, int] | None:
    eligible = inside & np.isfinite(values) & (values != FILL)
    if not eligible.any():
        return None
    candidates = np.where(eligible, values, -np.inf)
    return tuple(map(int, np.unravel_index(np.argmax(candidates), values.shape)))


def rank_municipalities(
    lon: np.ndarray, lat: np.ndarray, rainfall: np.ndarray, periods: np.ndarray,
    end_hours: np.ndarray, municipalities: list[tuple[str, str, object]],
    event_start: datetime,
) -> tuple[list[dict], list[dict]]:
    """Choose separate rainfall and period maxima at grid centers in each polygon."""
    if rainfall.shape != periods.shape or rainfall.shape != end_hours.shape:
        raise ValueError("rainfall, period, and end-hour shapes differ")
    if rainfall.shape != (len(lat), len(lon)):
        raise ValueError("field shape differs from coordinate dimensions")
    rain_rows, period_rows = [], []
    for code, name, polygon in municipalities:
        west, south, east, north = polygon.bounds
        xi = np.flatnonzero((lon >= west) & (lon <= east))
        yi = np.flatnonzero((lat >= south) & (lat <= north))
        if not len(xi) or not len(yi):
            raise ValueError(f"no grid centers in {name} ({code})")
        xx, yy = np.meshgrid(lon[xi], lat[yi])
        inside = contains_xy(polygon, xx, yy)
        if not inside.any():
            raise ValueError(f"no grid centers in {name} ({code})")
        rain = rainfall[np.ix_(yi, xi)]
        period = periods[np.ix_(yi, xi)]
        hours = end_hours[np.ix_(yi, xi)]
        for primary, rows in ((rain, rain_rows), (period, period_rows)):
            chosen = _selected_cell(primary, inside)
            if chosen is None:
                rows.append({"code": code, "name": name, "grid_cells": int(inside.sum())})
                continue
            y, x = chosen
            rain_value = float(rain[y, x])
            period_value = float(period[y, x])
            hour = int(hours[y, x])
            rows.append({
                "code": code, "name": name, "grid_cells": int(inside.sum()),
                "rainfall_mm": rain_value if np.isfinite(rain_value) and rain_value != FILL else None,
                "return_period_years": (
                    period_value if np.isfinite(period_value) and period_value != FILL else None
                ),
                "longitude": float(lon[xi[x]]), "latitude": float(lat[yi[y]]),
                "end_time_jst": (
                    (event_start + timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M")
                    if hour > 0 else None
                ),
            })
    for rows, key in ((rain_rows, "rainfall_mm"), (period_rows, "return_period_years")):
        rows.sort(key=lambda row: (
            row.get(key) is None,
            -(row.get(key) if row.get(key) is not None else 0),
            row["code"],
        ))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank if row.get(key) is not None else None
    return rain_rows, period_rows


FIELDS = ("rank", "code", "name", "grid_cells", "rainfall_mm",
          "return_period_years", "longitude", "latitude", "end_time_jst")


def _fmt(value: object, precision: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{precision}f}"
    return str(value)


def write_rankings(output_dir: Path, rain_rows: list[dict], period_rows: list[dict],
                   event_start: datetime, event_end: datetime) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, suffix = _event_names(event_start.date(), event_end.date())
    for filename, rows in ((f"municipal_max_rainfall{suffix}.csv", rain_rows),
                           (f"municipal_max_return_period{suffix}.csv", period_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                formatted = dict(row)
                for field in ("rainfall_mm", "return_period_years"):
                    if formatted.get(field) is not None:
                        formatted[field] = (
                            f"{formatted[field]:.3f}" if field == "return_period_years"
                            else f"{formatted[field]:.1f}"
                        )
                for field in ("longitude", "latitude"):
                    if formatted.get(field) is not None:
                        formatted[field] = f"{formatted[field]:.6f}"
                writer.writerow(formatted)
    lines = [
        "# 千葉県内の市区町村別ランキング", "",
        (f"対象期間: {event_start:%Y-%m-%d}～{event_end:%Y-%m-%d}。"
         "解析雨量RR60による1時間刻みの最大24時間降水量と、"
         "2006–2025年の年最大値から求めた格子別GEV再現期間。"), "",
        ("市区町村の境界内に中心がある約1 km格子を対象とし、"
         "千葉市は1市として集計した。2表はそれぞれ別の格子で最大値を選ぶため、"
         "対になる値は市区町村内の最大値とは限らない。"
         "確率年は格子別の点推定値であり、市区町村全体の再現期間を表すものではない。"
         "欠測は順位から除外した。"), "",
    ]
    for title, rows in (("最大24時間降水量順", rain_rows),
                        ("最大確率年順", period_rows)):
        lines.extend([
            f"## {title}", "",
            "| 順位 | 市区町村 | 雨量 (mm) | 確率年 (年) | 格子中心 (東経, 北緯) | 24時間窓の終了 (JST) |",
            "|---:|---|---:|---:|---|---|",
        ])
        for row in rows:
            location = (f"{row['longitude']:.4f}, {row['latitude']:.4f}"
                        if "longitude" in row else "—")
            lines.append(
                f"| {_fmt(row['rank'])} | {row['name']} | {_fmt(row.get('rainfall_mm'))} | "
                f"{_fmt(row.get('return_period_years'), 2)} | {location} | "
                f"{_fmt(row.get('end_time_jst'))} |"
            )
        lines.append("")
    (output_dir / f"municipal_rankings{suffix}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="final NetCDF; defaults to the selected event in --output-dir")
    parser.add_argument("--output-dir", type=Path, default=Path("results/nusdas_2006_2025"))
    parser.add_argument("--city-shp", type=Path, default=DEFAULT_CITY_SHP)
    parser.add_argument("--event-start", type=date.fromisoformat, default=EVENT_START)
    parser.add_argument("--event-end", type=date.fromisoformat, default=date(2026, 8, 15))
    args = parser.parse_args()
    if args.event_start > args.event_end:
        parser.error("event start must not exceed end")
    _, default_name, _ = _event_names(args.event_start, args.event_end)
    source = args.input or args.output_dir / default_name
    municipalities = load_municipalities(args.city_shp)
    with netcdf_file(source, "r", mmap=True) as dataset:
        lon = dataset.variables["lon"][:].copy()
        lat = dataset.variables["lat"][:].copy()
        west = min(polygon.bounds[0] for _, _, polygon in municipalities)
        south = min(polygon.bounds[1] for _, _, polygon in municipalities)
        east = max(polygon.bounds[2] for _, _, polygon in municipalities)
        north = max(polygon.bounds[3] for _, _, polygon in municipalities)
        xi = np.flatnonzero((lon >= west) & (lon <= east))
        yi = np.flatnonzero((lat >= south) & (lat <= north))
        if not len(xi) or not len(yi):
            raise ValueError("no grid cells overlap Chiba municipalities")
        section = np.s_[yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1]
        rainfall = dataset.variables["event_max_24h_mm"][section].copy()
        periods = dataset.variables["return_period_years"][section].copy()
        end_hours = dataset.variables["event_end_hour"][section].copy()
    start_time = datetime.combine(args.event_start, time.min, tzinfo=JST)
    end_time = datetime.combine(args.event_end, time.min, tzinfo=JST)
    rain_rows, period_rows = rank_municipalities(
        lon[xi], lat[yi], rainfall, periods, end_hours, municipalities, start_time,
    )
    write_rankings(args.output_dir, rain_rows, period_rows, start_time, end_time)
    print(f"wrote rankings for {len(municipalities)} municipalities to {args.output_dir}")


if __name__ == "__main__":
    main()
