"""Download and parse JMA historical-observation tables."""

from __future__ import annotations

import hashlib
import json
import re
import time
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


def event_url(station: Station) -> str:
    return _url(
        station,
        station.daily_page,
        year=2026,
        month=8,
        day="",
        view="a5",
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
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
