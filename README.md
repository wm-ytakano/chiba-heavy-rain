# Chiba heavy-rain return periods

This repository estimates station-wise return periods for the 13–15 August 2026
Chiba heavy-rain event from JMA ground observations. It uses annual maximum rolling
24-hour rainfall, a stationary GEV distribution fitted by sample L-moments, and
station-wise nonparametric bootstrap intervals.

The primary fit uses observations only through 2025, so the event being evaluated is
not also used to estimate its reference distribution. A fit including the 2026 event
is reported as a sensitivity analysis.

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

The workflow also produces a conditional 1976–2014 fit for the same 13 stations.
That window is inferred from the article's “39 years centered on 1995” wording and
the 2014 endpoint of the CMIP6 historical experiment; it is not stated in the
NEX-GDDP-CMIP6 or NIES2020 source metadata. NIES2020's documented 39-year period
is instead the 1980–2018 bias-correction reference period.

## Interpretation

Weathernews reported 574.2 mm from gridded analyzed precipitation over Chiba Midori
Ward to eastern Ichihara. This project analyzes point rain-gauge observations, so it
does not assign a return period to that gridded 574.2 mm value. It provides an
observation-based, reproducible comparison for gauges that sampled the same event.
