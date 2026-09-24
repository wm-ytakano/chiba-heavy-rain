# Chiba heavy-rain return periods

This repository estimates station-wise return periods for the 13–15 August 2026
Chiba heavy-rain event from JMA ground observations. It uses annual maximum rolling
24-hour rainfall, a stationary GEV distribution fitted by sample L-moments, and
station-wise nonparametric bootstrap intervals.

The primary fit uses the inferred 1976–2014 historical window and displays the four
eligible JMA gauges directly corresponding to locations named in the Weathernews
article: Chiba, Mobara, Ushiku (central Ichihara), and Sakura. Fits through 2025 and
fits including the 2026 event are retained as sensitivity analyses.

## Run

```bash
uv sync
uv run pytest
uv run chiba-rain --bootstrap 500
```

Use `--refresh` to replace the cached JMA HTML snapshots. The source URL and SHA-256
of every snapshot are recorded in `data/raw/manifest.json`.

To compare fixed-calendar-day rainfall with moving 24-hour rainfall at the four
article-matched stations without rewriting the blog report, run:

```bash
uv run chiba-rain-daily-vs-24h --bootstrap 2000
```

This writes paired annual-maxima data, three return-period scenarios, and two
diagnostic figures to `results/`. The comparison command never writes
`results/report.md`.

Outputs are written to `results/`, including the Japanese report
`results/report.md`, tidy CSV files, and diagnostic plots.

## Nationwide NuSDaS analyzed rainfall

The independent `chiba-rain-nusdas` command computes grid-cell annual maximum
moving 24-hour rainfall from hourly RR60 for 2006–2025, fits the same stationary
L-moment GEV model, and evaluates the 13–15 August 2026 event. The event is not
included in the fit. The NuSDaS database defaults to `/mnt/yt02/db_an.nus`.

The local `pynusdas` package and its native libraries are required. On this host,
the source is `/home/ytakano/NuSDaS/Python_Nusdas`; make it importable before
running, for example:

```bash
PYTHONPATH=/home/ytakano/NuSDaS/Python_Nusdas \
  .venv/bin/python -m chiba_heavy_rain.nusdas_rain
```

After installing the project with `uv sync`, `chiba-rain-nusdas` is also available
as a console command. If the source has been copied to SSD, pass
`--db-root /mnt/ssd/chiba-heavy-rain/db_an.nus`.

The command writes separate `annual_YYYY.nc` checkpoints and resumes from them.
It processes up to four years concurrently by default; use `--workers 1` to
reduce memory use.
For a limited initial run, use `--first-year 2006 --last-year 2006 --annual-only`;
run the default command afterwards to fill the remaining years and produce the
combined NetCDF, Chiba two-panel map, caption, and metadata in
`results/nusdas_2006_2025/`. Use `--db-root` and `--output-dir` to change paths.
For another post-2025 event, pass `--event-start YYYY-MM-DD --event-end YYYY-MM-DD`.
The command reuses all existing `annual_YYYY.nc` files and gives the new event
outputs date-specific names; it does not recalculate the historical years.
This national computation reads roughly 48 GB of compressed source files and
processes about 8.6 million grid cells, so it can take a long time.

An absent or unreadable daily file contributes 24 missing hours; a failed hourly
record contributes one missing hour. Only 24 consecutive valid hourly values form
a rainfall total. Each year's maximum is retained if any valid 24-hour window
exists, including the incomplete 2006 source year. Daily file outcomes and
missing timestamps are saved in `input_status_*.csv` and `missing_times_*.csv`.
The final GEV fit requires all 20 annual maxima at a grid cell. The event maximum
uses windows wholly within 13–15 August 2026; a cell with no valid event window
has no return period. RR60 yields hourly-step maxima, which may differ from the
article's 10-minute-step analyzed rainfall.

The Chiba map has (a) the event's maximum moving 24-hour rainfall from
`event_max_24h_mm` and (b) its GEV return period from `return_period_years`.
Panel (a) uses the eight rainfall colors in [JMA's 2020 color guide, Table 2-1](https://www.jma.go.jp/jma/kishou/info/colorguide/HPColorGuide_202007.pdf),
reassigned to the requested **24-hour** thresholds of 1, 10, 20, 50, 100,
200, 300, and 400 mm. JMA's original rainfall thresholds in that table are
for **hourly** rainfall. Values below 1 mm and missing values are white;
the right colorbar extension marks values above 400 mm. Panel (b) uses
`inferno_r` with boundaries at 10, 20, 50, 100, 200, 500, 1,000, 2,000,
5,000, and 10,000 years. Values below 10 years and missing values are white;
the right extension marks values above 10,000 years.

Both panels use the municipal and related boundaries in `data/city/city.shp` (WGS84),
with thicker prefecture borders formed by dissolving municipalities according
to the first two characters of `regioncode`. Fine municipal borders are drawn
only for Chiba (`regioncode` prefix `12`); surrounding prefectures show only
their thicker borders. Both panels label the 54 Chiba municipalities selected
by `regioncode` prefix `12` from its UTF-8 `name` field. It uses a Mercator
projection over 139.6–140.9°E and 34.8–36.2°N, with geographic ticks and no
coordinate grid lines. This requires GDAL's `ogr2ogr` command.

The prefecture and Chiba municipal boundaries are black, and each colorbar is
aligned below its panel.

To update the figure from the existing final NetCDF, without reading NuSDaS
or recomputing the GEV, run:

```bash
MPLCONFIGDIR=/tmp/chiba-mpl .venv/bin/python -m chiba_heavy_rain.nusdas_rain --render-existing
```

Use `--city-shp` to specify another city shapefile with the same coordinate system.

The 1976–2014 window is inferred from the article's “39 years centered on 1995”
wording and the 2014 endpoint of the CMIP6 historical experiment; it is not stated
in the NEX-GDDP-CMIP6 or NIES2020 source metadata. NIES2020's documented 39-year
period is instead the 1980–2018 bias-correction reference period. Numerical results
for all 13 eligible gauges remain available in CSV, but fitting-curve figures show
only the four article-matched gauges.

## Interpretation

Weathernews reported 574.2 mm from gridded analyzed precipitation over Chiba Midori
Ward to eastern Ichihara. This project analyzes point rain-gauge observations, so it
does not assign a return period to that gridded 574.2 mm value. It provides an
observation-based, reproducible comparison for gauges that sampled the same event.
