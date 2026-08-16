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

Outputs are written to `results/`, including the Japanese report
`results/report.md`, tidy CSV files, and diagnostic plots.

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
