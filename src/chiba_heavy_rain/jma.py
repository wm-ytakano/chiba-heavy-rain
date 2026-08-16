"""Download and parse JMA historical-observation tables."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from .stations import Station

BASE_URL = "https://www.data.jma.go.jp/stats/etrn/view"
PREC_NO = "45"
USER_AGENT = "chiba-heavy-rain-study/0.1 (academic reproducibility; low request rate)"


def _url(station: Station, page: str, **params: object) -> str:
    query = {"prec_no": PREC_NO, "block_no": station.block_no, **params}
    return str(httpx.URL(f"{BASE_URL}/{page}", params=query))


def annual_url(station: Station) -> str:
    return _url(station, station.annual_page, year="", month="", day="", view="a5")


def annual_daily_url(station: Station) -> str:
    """JMA annual main-elements table containing maximum daily rainfall."""
    return _url(station, station.annual_page, year="", month="", day="", view="p1")


def event_url(station: Station) -> str:
    return _url(
        station,
        station.daily_page,
        year=2026,
        month=8,
        day="",
        view="a5",
    )


def event_daily_url(station: Station) -> str:
    """JMA August 2026 daily main-elements table containing calendar-day totals."""
    return _url(
        station,
        station.daily_page,
        year=2026,
        month=8,
        day="",
        view="p1",
    )


def _merge_manifest(raw_dir: Path, records: list[dict[str, object]]) -> None:
    """Merge newly fetched table records without discarding other cached tables."""
    manifest_path = raw_dir / "manifest.json"
    existing: list[dict[str, object]] = []
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            existing = [record for record in loaded if isinstance(record, dict)]
    replaced = {(str(record["block_no"]), str(record["table"])) for record in records}
    preserved = [
        record
        for record in existing
        if (str(record.get("block_no")), str(record.get("table"))) not in replaced
    ]
    merged = preserved + records
    merged.sort(key=lambda record: (str(record.get("block_no")), str(record.get("table"))))
    manifest_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def download_pages(
    stations: tuple[Station, ...], raw_dir: Path, refresh: bool = False
) -> list[dict[str, object]]:
    """Fetch one annual and one August-2026 table per station, with a local snapshot."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30.0
    ) as client:
        for station in stations:
            for label, url in (("annual", annual_url(station)), ("event", event_url(station))):
                path = raw_dir / f"{station.block_no}_{label}.html"
                if refresh or not path.exists():
                    response = client.get(url)
                    response.raise_for_status()
                    path.write_bytes(response.content)
                    time.sleep(0.2)
                payload = path.read_bytes()
                manifest.append(
                    {
                        **asdict(station),
                        "table": label,
                        "url": url,
                        "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    _merge_manifest(raw_dir, manifest)
    return manifest


def download_daily_comparison_pages(
    stations: tuple[Station, ...], raw_dir: Path, refresh: bool = False
) -> list[dict[str, object]]:
    """Fetch annual daily maxima and August-2026 calendar-day rainfall tables."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30.0
    ) as client:
        for station in stations:
            pages = (
                (
                    "annual_daily",
                    annual_daily_url(station),
                    raw_dir / f"{station.block_no}_annual_daily.html",
                ),
                (
                    "event_daily",
                    event_daily_url(station),
                    raw_dir / f"{station.block_no}_event_daily.html",
                ),
            )
            for label, url, path in pages:
                if refresh or not path.exists():
                    response = client.get(url)
                    response.raise_for_status()
                    path.write_bytes(response.content)
                    time.sleep(0.2)
                payload = path.read_bytes()
                manifest.append(
                    {
                        **asdict(station),
                        "table": label,
                        "url": url,
                        "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    _merge_manifest(raw_dir, manifest)
    return manifest


def _numeric(text: str) -> float | None:
    cleaned = re.sub(r"[^0-9.+-]", "", text.replace("−", "-"))
    if cleaned in {"", ".", "+", "-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _target_table(path: Path, title_fragment: str) -> BeautifulSoup:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for table in soup.find_all("table"):
        if title_fragment in table.get_text(" ", strip=True):
            return table
    raise ValueError(f"Could not find {title_fragment!r} table in {path}")


def _header_paths(table: BeautifulSoup) -> list[tuple[str, ...]]:
    """Expand rowspan/colspan headers into one semantic path per data column."""
    header_rows: list[list[BeautifulSoup]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        if any(cell.name == "td" for cell in cells):
            break
        header_rows.append(cells)
    grid: dict[tuple[int, int], str] = {}
    max_column = 0
    for row_index, cells in enumerate(header_rows):
        column = 0
        for cell in cells:
            while (row_index, column) in grid:
                column += 1
            text = cell.get_text(" ", strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    grid[(row_index + row_offset, column + column_offset)] = text
            column += colspan
            max_column = max(max_column, column)
    paths: list[tuple[str, ...]] = []
    for column in range(max_column):
        parts: list[str] = []
        for row_index in range(len(header_rows)):
            text = grid.get((row_index, column), "")
            if text and (not parts or text != parts[-1]):
                parts.append(text)
        paths.append(tuple(parts))
    return paths


def _find_semantic_table_column(
    path: Path, predicate: Callable[[tuple[str, ...]], bool]
) -> tuple[BeautifulSoup, int]:
    """Find a table column by its expanded header path."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for table in soup.find_all("table"):
        for index, parts in enumerate(_header_paths(table)):
            if predicate(parts):
                return table, index
    raise ValueError(f"Could not find requested semantic column in {path}")


def parse_annual_24h(path: Path) -> pd.DataFrame:
    """Parse JMA's annual detailed N-hour table into annual 24-hour maxima."""
    table = _target_table(path, "最大24時間降水量")
    rows: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        year_match = re.fullmatch(r"(18|19|20)\d{2}", texts[0])
        if not year_match:
            continue
        # Annual N-hour tables have year + seven value/date pairs. 24 h is pair 5.
        if len(texts) < 11:
            continue
        value_text = texts[9]
        value_classes = " ".join(cells[9].get("class", []))
        rows.append(
            {
                "year": int(texts[0]),
                "max_24h_mm": _numeric(value_text),
                "date": texts[10],
                "raw_value": value_text,
                # JMA uses data_2t_* at a discontinuity in the statistical series.
                # data_1t_* marks the switch to finer observation intervals, which
                # this study intentionally does not treat as a break.
                "statistical_break": "data_2t" in value_classes,
                # ']' is JMA's insufficient-data value and is not used in rankings.
                # Parenthesized quasi-normal values remain statistically usable.
                "usable": "]" not in value_text and _numeric(value_text) is not None,
                "value_classes": value_classes,
                "row_classes": " ".join(tr.get("class", [])),
                "row_html": str(tr),
            }
        )
    if not rows:
        raise ValueError(f"No annual rows parsed from {path}")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def parse_annual_daily_max(path: Path) -> pd.DataFrame:
    """Parse annual maximum calendar-day rainfall from JMA main-elements tables."""

    def is_daily_max(parts: tuple[str, ...]) -> bool:
        joined = " ".join(parts)
        return (
            "降水量" in joined
            and "最大" in parts
            and any(part == "日" or part.startswith("日 ") for part in parts)
        )

    table, value_index = _find_semantic_table_column(path, is_daily_max)
    rows: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        texts = [cell.get_text(" ", strip=True) for cell in cells]
        if not texts or not re.fullmatch(r"(18|19|20)\d{2}", texts[0]):
            continue
        if value_index >= len(cells):
            continue
        value_text = texts[value_index]
        value_classes = " ".join(cells[value_index].get("class", []))
        rows.append(
            {
                "year": int(texts[0]),
                "max_daily_mm": _numeric(value_text),
                "raw_daily_value": value_text,
                "daily_statistical_break": "data_2t" in value_classes,
                "daily_usable": "]" not in value_text and _numeric(value_text) is not None,
                "daily_value_classes": value_classes,
            }
        )
    if not rows:
        raise ValueError(f"No annual maximum daily rainfall rows parsed from {path}")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def parse_event_24h(path: Path, days: range = range(13, 16)) -> dict[str, object]:
    """Return the largest rolling 24-hour value reported on event days."""
    table = _target_table(path, "最大24時間降水量")
    candidates: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        texts = [c.get_text(" ", strip=True) for c in cells]
        if not texts or not re.fullmatch(r"\d{1,2}", texts[0]):
            continue
        day = int(texts[0])
        if day not in days or len(texts) < 11:
            continue
        candidates.append(
            {
                "day": day,
                "event_24h_mm": _numeric(texts[9]),
                "ending_time": texts[10],
                "raw_value": texts[9],
            }
        )
    valid = [r for r in candidates if r["event_24h_mm"] is not None]
    if not valid:
        raise ValueError(f"No event 24-hour value parsed from {path}")
    return max(valid, key=lambda r: float(r["event_24h_mm"]))


def parse_event_daily(path: Path, days: range = range(13, 16)) -> dict[str, object]:
    """Return the largest fixed-calendar-day rainfall total on the event days."""

    def is_daily_total(parts: tuple[str, ...]) -> bool:
        joined = " ".join(parts)
        return "降水量" in joined and any(part.startswith("合計") for part in parts)

    table, value_index = _find_semantic_table_column(path, is_daily_total)
    candidates: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        texts = [cell.get_text(" ", strip=True) for cell in cells]
        if not texts or not re.fullmatch(r"\d{1,2}", texts[0]):
            continue
        day = int(texts[0])
        if day not in days or value_index >= len(cells):
            continue
        value_text = texts[value_index]
        value_classes = " ".join(cells[value_index].get("class", []))
        candidates.append(
            {
                "day": day,
                "event_daily_mm": _numeric(value_text),
                "event_daily_raw_value": value_text,
                "event_daily_usable": "]" not in value_text and _numeric(value_text) is not None,
                "event_daily_value_classes": value_classes,
            }
        )
    valid = [row for row in candidates if row["event_daily_mm"] is not None]
    if not valid:
        raise ValueError(f"No event daily rainfall value parsed from {path}")
    return max(valid, key=lambda row: float(row["event_daily_mm"]))


def latest_contiguous_complete_run(
    frame: pd.DataFrame, end_year: int = 2025
) -> pd.DataFrame:
    """Select usable values in JMA's latest statistically unbroken segment.

    JMA's bracketed/parenthesized quality annotations remain auditable in raw_value;
    insufficient-data annual values are omitted but do not themselves define a statistical
    discontinuity. Observation-interval changes are ignored as explicitly requested.
    """
    values = frame.loc[frame["year"] <= end_year].copy()
    breaks = values.loc[values["statistical_break"]]
    start = int(values["year"].min())
    if len(breaks):
        start = int(breaks["year"].max())
    selected = values.loc[(values["year"] >= start) & values["usable"]].copy()
    return selected.reset_index(drop=True)
