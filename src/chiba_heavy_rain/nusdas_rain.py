"""Nationwide RR60 annual 24-hour maxima and August 2026 return periods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.io import netcdf_file
from scipy.special import gamma
from scipy.stats import genextreme

GRID_Y = 3360
GRID_X = 2560
YEARS = tuple(range(2006, 2026))
EVENT_START = date(2026, 8, 13)
EVENT_END = date(2026, 8, 15)
FILL = np.float32(-9999.0)
DEFAULT_CITY_SHP = Path(__file__).resolve().parents[2] / "data/city/city.shp"
CHIBA_EXTENT = (139.6, 140.9, 34.8, 36.2)
RETURN_PERIOD_TICKS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
RAINFALL_TICKS = (1, 10, 20, 50, 100, 200, 300, 400)
# JMA Color Guide (2020), Table 2-1 rainfall colors in ascending severity.
JMA_RAINFALL_RGB = (
    (242, 242, 255), (160, 210, 255), (33, 140, 255), (0, 65, 255),
    (250, 245, 0), (255, 153, 0), (255, 40, 0), (180, 0, 104),
)
COLORBAR_EXTEND_FRACTION = 0.05


def days_inclusive(first: date, last: date):
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)


@contextmanager
def nusdas_workdir(db_root: Path):
    """The native NuSDaS library resolves NUSDAS60 relative to the process cwd."""
    root = db_root.resolve(strict=True)
    if not (root / "nusdas_def").is_dir():
        raise ValueError(f"NuSDaS definition directory missing: {root}")
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="chiba-nusdas-") as directory:
        (Path(directory) / "NUSDAS60").symlink_to(root, target_is_directory=True)
        try:
            os.chdir(directory)
            yield
        finally:
            os.chdir(previous)


class RR60Reader:
    """Read one hourly end time using the same PR10/RR60LV contract as the reference."""

    def __init__(self) -> None:
        try:
            import pynusdas.nusdas_const as nusc
            import pynusdas.nusdas_util as nusu
            from pynusdas import srf
        except ImportError as exc:
            raise RuntimeError("pynusdas is required; see README.md") from exc
        self.nusc = nusc
        self.nusu = nusu
        self.srf = srf
        nusu.iocntl(nusc.N_IO_R_FCLOSE, nusc.N_OFF)
        self.reader = nusu.NSDutil(t1="_SRFLLLY", t2="AASV", t3="PR10", nrd=60)
        self.reader.set_ndef()
        self.reader.mem = self.reader.ndef.memlist[0]
        self.reader.pl = "SURF"
        self.reader.el = "RR60LV"
        self.reader.fmt = nusc.N_I4
        self.reader.size = self.reader.ndef.grid.size
        grid = self.reader.ndef.grid
        nx, ny = map(int, grid.size)
        bx, by, blat, blon = grid.base
        dx, dy = grid.dist
        self.shape = (ny, nx)
        self.lon = (blon + (np.arange(nx) + bx) * dx).astype(np.float64)
        self.lat = (blat - (np.arange(ny) + by) * dy).astype(np.float64)

    def read(self, day: date, hour: int) -> np.ndarray:
        if not 1 <= hour <= 24:
            raise ValueError("hour must be 1..24")
        self.reader.bt = np.datetime64(day.isoformat(), "m")
        self.reader.vt = self.reader.bt + np.timedelta64(hour, "h")
        self.reader.set_cntl()
        self.reader.set_subc()
        raw = self.reader.read()
        decoded, code = self.srf.lv_trans(
            idat=raw.astype(np.int32, copy=False), ispec=self.reader.subc.ispc
        )
        if int(code) != raw.size:
            raise RuntimeError(f"RR60LV conversion returned {code}, expected {raw.size}")
        if decoded.shape != self.shape:
            raise RuntimeError(f"RR60LV shape {decoded.shape}, expected {self.shape}")
        result = np.asarray(decoded, dtype=np.float32)
        result[result < 0] = np.nan
        return result

    def close_files(self) -> None:
        self.nusu.allfile_close(self.nusc.N_FOPEN_READ, silent=True)


class RollingMaximum:
    """24 consecutive valid hours, with an independent missing mask per grid cell."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.ring = np.zeros((24, *shape), dtype=np.float32)
        self.ring_valid = np.zeros((24, *shape), dtype=bool)
        self.total = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.uint8)
        self.maximum = np.full(shape, np.nan, dtype=np.float32)
        self.argmax_hour = np.zeros(shape, dtype=np.int16)
        self.valid_windows = np.zeros(shape, dtype=np.uint32)
        self.hour_count = 0

    def add(self, field: np.ndarray | None) -> None:
        if field is not None and field.shape != self.shape:
            raise ValueError(f"field shape {field.shape} != {self.shape}")
        position = self.hour_count % 24
        old = self.ring[position]
        valid_slot = self.ring_valid[position]
        self.total -= old
        self.count -= valid_slot
        old.fill(0)
        if field is None:
            valid_slot.fill(False)
        else:
            np.isfinite(field, out=valid_slot)
            np.copyto(old, field, where=valid_slot)
        self.total += old
        self.count += valid_slot
        self.hour_count += 1
        if self.hour_count < 24:
            return
        valid = self.count == 24
        self.valid_windows += valid
        better = valid & (np.isnan(self.maximum) | (self.total > self.maximum))
        self.maximum[better] = self.total[better]
        self.argmax_hour[better] = self.hour_count


def _source_file(db_root: Path, day: date) -> Path | None:
    base = db_root / "_SRF/LL/AA/PR10" / day.strftime("%Y%m%d0000")
    if base.is_file():
        return base
    compressed = base.with_name(base.name + ".gz")
    return compressed if compressed.is_file() else None


def aggregate_period(
    reader: RR60Reader,
    db_root: Path,
    first: date,
    last: date,
    status_path: Path,
    missing_path: Path,
) -> RollingMaximum:
    aggregate = RollingMaximum(reader.shape)
    with status_path.open("w", newline="") as status_file, missing_path.open(
        "w", newline=""
    ) as missing_file:
        status = csv.writer(status_file)
        missing = csv.writer(missing_file)
        status.writerow(["date", "source", "read_hours", "missing_hours", "failed_hours"])
        missing.writerow(["valid_time", "source", "reason"])
        for day in days_inclusive(first, last):
            source = _source_file(db_root, day)
            read_hours = failed_hours = 0
            for hour in range(1, 25):
                valid_time = np.datetime64(day.isoformat(), "h") + np.timedelta64(hour, "h")
                if source is None:
                    field = None
                    reason = "file_missing"
                else:
                    try:
                        field = reader.read(day, hour)
                        reason = ""
                        read_hours += 1
                    except Exception as exc:  # noqa: BLE001 - record native reader failures
                        field = None
                        reason = f"read_failed:{type(exc).__name__}:{str(exc)[:200]}"
                        failed_hours += 1
                if reason:
                    missing.writerow([str(valid_time), str(source or ""), reason])
                aggregate.add(field)
            status.writerow(
                [day.isoformat(), str(source or ""), read_hours, 24 - read_hours, failed_hours]
            )
            if hasattr(reader, "close_files"):
                reader.close_files()
            status_file.flush()
            missing_file.flush()
    return aggregate


def _write_field(
    path: Path, field: np.ndarray, valid_windows: np.ndarray,
    lon: np.ndarray, lat: np.ndarray,
) -> None:
    with netcdf_file(path, "w", version=2) as dataset:
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("lat", "d", ("lat",))[:] = lat
        dataset.createVariable("lon", "d", ("lon",))[:] = lon
        var = dataset.createVariable("max_24h_mm", "f", ("lat", "lon"))
        var[:] = np.where(np.isfinite(field), field, FILL)
        var.missing_value = FILL
        var.units = "mm"
        count = dataset.createVariable("valid_window_count", "i", ("lat", "lon"))
        count[:] = valid_windows.astype(np.int32)
        count.units = "count"


def _read_field(path: Path, y0: int, y1: int) -> np.ndarray:
    with netcdf_file(path, "r", mmap=False) as dataset:
        field = dataset.variables["max_24h_mm"][y0:y1, :].copy()
    field[field == FILL] = np.nan
    return field


def fit_gev_grid(annual: np.ndarray, event: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized equivalent of extremes.fit_gev, evaluated in caller-sized tiles."""
    if annual.ndim != 3 or annual.shape[0] != 20 or annual.shape[1:] != event.shape:
        raise ValueError("expected 20 annual fields and one same-shape event field")
    shape = event.shape
    result = {
        key: np.full(shape, np.nan, dtype=np.float32)
        for key in ("shape_xi", "location", "scale", "return_period_years")
    }
    complete = np.isfinite(annual).all(axis=0)
    if not complete.any():
        return result
    x = np.sort(annual[:, complete].astype(np.float64), axis=0)
    index = np.arange(20, dtype=np.float64)[:, None]
    b0 = np.mean(x, axis=0)
    b1 = np.mean(index / 19 * x, axis=0)
    b2 = np.mean(index * (index - 1) / (19 * 18) * x, axis=0)
    l2 = 2 * b1 - b0
    tau3 = (6 * b2 - 6 * b1 + b0) / np.where(l2 > 0, l2, np.nan)
    eligible = (l2 > 0) & (tau3 > -1) & (tau3 < 1)
    if not eligible.any():
        return result
    tau = tau3[eligible]
    lo = np.full(tau.shape, -10.0)
    hi = np.full(tau.shape, 0.999999)
    log2 = np.log(2.0)
    log3 = np.log(3.0)
    for _ in range(46):
        mid = (lo + hi) / 2
        theory = 2 * np.expm1(mid * log3) / np.expm1(mid * log2) - 3
        theory[np.abs(mid) < 1e-8] = 2 * log3 / log2 - 3
        lower = theory < tau
        lo[lower] = mid[lower]
        hi[~lower] = mid[~lower]
    xi = (lo + hi) / 2
    xi[np.abs(xi) < 1e-7] = 0
    selected_l2 = l2[eligible]
    selected_l1 = b0[eligible]
    scale = np.empty_like(xi)
    location = np.empty_like(xi)
    zero = xi == 0
    scale[zero] = selected_l2[zero] / log2
    location[zero] = selected_l1[zero] - np.euler_gamma * scale[zero]
    nonzero = ~zero
    g = gamma(1 - xi[nonzero])
    scale[nonzero] = (
        selected_l2[nonzero] * xi[nonzero] / (g * np.expm1(xi[nonzero] * log2))
    )
    location[nonzero] = selected_l1[nonzero] - scale[nonzero] * (g - 1) / xi[nonzero]
    valid_fit = np.isfinite(scale) & (scale > 0) & np.isfinite(location)
    selected_indices = np.flatnonzero(complete)[eligible][valid_fit]
    if not len(selected_indices):
        return result
    shape_values = xi[valid_fit]
    location_values = location[valid_fit]
    scale_values = scale[valid_fit]
    event_values = event.ravel()[selected_indices]
    log_survival = genextreme.logsf(
        event_values, -shape_values, loc=location_values, scale=scale_values
    )
    with np.errstate(over="ignore"):
        period = np.exp(-log_survival)
    for key, values in (
        ("shape_xi", shape_values),
        ("location", location_values),
        ("scale", scale_values),
        ("return_period_years", period),
    ):
        result[key].ravel()[selected_indices] = values
    return result


def _write_final(
    output: Path, annual_paths: list[Path], event: RollingMaximum, reader: RR60Reader,
    event_start: date, event_end: date,
) -> None:
    ny, nx = reader.shape
    with netcdf_file(output, "w", version=2) as dataset:
        dataset.createDimension("year", 20)
        dataset.createDimension("lat", ny)
        dataset.createDimension("lon", nx)
        dataset.createVariable("year", "i", ("year",))[:] = YEARS
        dataset.createVariable("lat", "d", ("lat",))[:] = reader.lat
        dataset.createVariable("lon", "d", ("lon",))[:] = reader.lon
        annual_var = dataset.createVariable("annual_max_24h_mm", "f", ("year", "lat", "lon"))
        annual_var.missing_value = FILL
        annual_var.units = "mm"
        variables = {}
        for name, kind, units in (
            ("event_max_24h_mm", "f", "mm"),
            ("event_end_hour", "h", f"hours since {event_start} 00:00 JST"),
            ("event_valid_window_count", "i", "count"),
            ("n_years", "b", "count"),
            ("shape_xi", "f", "1"),
            ("location", "f", "mm"),
            ("scale", "f", "mm"),
            ("return_period_years", "f", "years"),
        ):
            var = dataset.createVariable(name, kind, ("lat", "lon"))
            var.units = units
            if kind == "f":
                var.missing_value = FILL
            variables[name] = var
        variables["event_max_24h_mm"][:] = np.where(
            np.isfinite(event.maximum), event.maximum, FILL
        )
        variables["event_end_hour"][:] = event.argmax_hour
        variables["event_valid_window_count"][:] = event.valid_windows.astype(np.int32)
        for y0 in range(0, ny, 64):
            y1 = min(y0 + 64, ny)
            annual = np.stack(
                [_read_field(path, y0, y1) for path in annual_paths], axis=0
            )
            annual_var[:, y0:y1, :] = np.where(np.isfinite(annual), annual, FILL)
            variables["n_years"][y0:y1, :] = np.isfinite(annual).sum(axis=0).astype(np.int8)
            fitted = fit_gev_grid(annual, event.maximum[y0:y1])
            for key, field in fitted.items():
                variables[key][y0:y1, :] = np.where(np.isnan(field), FILL, field)
        dataset.history = (
            f"RR60 2006-2025 annual maxima; {event_start}/{event_end} event evaluated separately"
        )
        dataset.annual_window_rule = "24 consecutive valid hourly values wholly within calendar year"
        dataset.event_window_rule = (
            f"24 consecutive valid hourly values wholly within {event_start}/{event_end}"
        )


def _geometry_segments(geometry: dict) -> list[np.ndarray]:
    """Include exterior and interior boundaries of polygon and multipolygon features."""
    kind = geometry["type"]
    if kind == "Polygon":
        polygons = [geometry["coordinates"]]
    elif kind == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"unexpected city geometry: {kind}")
    return [
        np.asarray(ring, dtype=np.float64)[:, :2]
        for polygon in polygons
        for ring in polygon
        if len(ring) >= 2
    ]


def _load_city_boundaries(
    city_shp: Path, extent: tuple[float, float, float, float]
) -> list[np.ndarray]:
    """Read only Chiba municipal boundaries within the plotted extent."""
    if not city_shp.is_file():
        raise FileNotFoundError(f"city boundary shapefile missing: {city_shp}")
    if shutil.which("ogr2ogr") is None:
        raise RuntimeError("ogr2ogr (GDAL) is required to draw city boundaries")
    west, east, south, north = extent
    completed = subprocess.run(
        [
            "ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(city_shp),
            "-where", "regioncode LIKE '12%'",
            "-spat", str(west), str(south), str(east), str(north),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    features = json.loads(completed.stdout)["features"]
    return [
        segment
        for feature in features
        if feature["geometry"] is not None
        for segment in _geometry_segments(feature["geometry"])
    ]


def _load_chiba_labels(city_shp: Path) -> list[tuple[str, float, float]]:
    """Use the UTF-8 name field for Chiba municipalities (regioncode prefix 12)."""
    from shapely.geometry import shape

    completed = subprocess.run(
        [
            "ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(city_shp),
            "-where", "regioncode LIKE '12%'",
        ],
        check=True, capture_output=True, text=True,
    )
    labels = []
    for feature in json.loads(completed.stdout)["features"]:
        properties = feature["properties"]
        code = str(properties["regioncode"])
        name = properties["name"]
        if not code.startswith("12") or not name or feature["geometry"] is None:
            continue
        geometry = shape(feature["geometry"])
        if geometry.geom_type == "MultiPolygon":
            geometry = max(geometry.geoms, key=lambda part: part.area)
        point = geometry.representative_point()
        labels.append((name, point.x, point.y))
    return labels


def _dissolved_prefecture_segments(
    features: list[dict], extent: tuple[float, float, float, float],
) -> tuple[list[np.ndarray], int]:
    """Dissolve all municipalities by prefecture code before clipping borders."""
    from collections import defaultdict

    from shapely import make_valid, union_all
    from shapely.geometry import box, shape

    groups = defaultdict(list)
    west, east, south, north = extent
    map_box = box(west, south, east, north)
    visible_codes = set()
    for feature in features:
        geometry = feature["geometry"]
        code = str(feature["properties"].get("regioncode") or "")
        if geometry is not None and len(code) >= 2 and code[:2].isdigit():
            polygon = shape(geometry)
            groups[code[:2]].append(polygon)
            if polygon.intersects(map_box):
                visible_codes.add(code[:2])
    if not groups:
        raise ValueError("no prefecture codes found in city shapefile")
    segments = []
    for code in sorted(visible_codes):
        polygons = [make_valid(polygon) if not polygon.is_valid else polygon
                    for polygon in groups[code]]
        # Clip the dissolved boundary, not the polygon, to avoid false map-edge lines.
        boundary = union_all(polygons).boundary.intersection(map_box)
        lines = boundary.geoms if hasattr(boundary, "geoms") else (boundary,)
        for line in lines:
            if line.geom_type == "LineString" and not line.is_empty:
                segments.append(np.asarray(line.coords, dtype=np.float64))
    return segments, len(groups)


def _load_dissolved_prefecture_boundaries(
    city_shp: Path, extent: tuple[float, float, float, float],
) -> tuple[list[np.ndarray], int]:
    completed = subprocess.run(
        ["ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(city_shp)],
        check=True, capture_output=True, text=True,
    )
    return _dissolved_prefecture_segments(json.loads(completed.stdout)["features"], extent)


def _return_period_color_scale():
    base_cmap = plt.colormaps["inferno_r"]
    cmap = base_cmap.with_extremes(bad="white", under="white", over=base_cmap(1.0))
    # The uppermost ordinary bin includes exactly 10,000 years.
    bounds = np.array((*RETURN_PERIOD_TICKS[:-1],
                       np.nextafter(float(RETURN_PERIOD_TICKS[-1]), np.inf)))
    return cmap, BoundaryNorm(bounds, cmap.N, clip=False), bounds


def _rainfall_color_scale():
    rgb = np.asarray(JMA_RAINFALL_RGB, dtype=np.float64) / 255.0
    cmap = ListedColormap(rgb[:-1]).with_extremes(
        bad="white", under="white", over=rgb[-1],
    )
    bounds = np.array((*RAINFALL_TICKS[:-1],
                       np.nextafter(float(RAINFALL_TICKS[-1]), np.inf)))
    return cmap, BoundaryNorm(bounds, cmap.N, clip=False), bounds


def _draw_chiba_panel(ax, lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
                      cmap, norm, title: str, city_segments: list[np.ndarray],
                      prefecture_segments: list[np.ndarray],
                      city_labels: list[tuple[str, float, float]] | None = None):
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    data_crs = ccrs.PlateCarree()
    image = ax.pcolormesh(
        lon, lat, np.ma.masked_invalid(values), shading="auto", cmap=cmap,
        norm=norm, transform=data_crs,
    )
    ax.set_extent(CHIBA_EXTENT, crs=data_crs)
    ax.set_xticks(np.arange(139.6, 140.91, 0.2), crs=data_crs)
    ax.set_yticks(np.arange(34.8, 36.21, 0.2), crs=data_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".1f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".1f"))
    ax.add_collection(LineCollection(
        city_segments, colors="black", linewidths=0.65, alpha=0.85,
        zorder=3, transform=data_crs,
    ))
    ax.add_collection(LineCollection(
        prefecture_segments, colors="white", linewidths=3.2,
        zorder=4, transform=data_crs,
    ))
    ax.add_collection(LineCollection(
        prefecture_segments, colors="black", linewidths=2.0,
        zorder=4.1, transform=data_crs,
    ))
    ax.set_title(title, loc="left", fontsize=15)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.15")
        spine.set_linewidth(1.2)
    ax.tick_params(direction="out", bottom=True, left=True, length=6, width=1.2,
                   labelsize=11.5)
    if city_labels is not None:
        from adjustText import adjust_text

        texts = [
            ax.text(x, y, name, fontsize=10.0, color="white", ha="center", va="center",
                    transform=data_crs, zorder=5,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
            for name, x, y in city_labels
        ]
        adjust_text(
            texts, target_x=[x for _, x, _ in city_labels],
            target_y=[y for _, _, y in city_labels], ax=ax,
            avoid_self=False, prevent_crossings=False, force_explode=0,
            ensure_inside_axes=True, expand=(1.05, 1.18), force_text=(0.12, 0.18),
            force_pull=(0.08, 0.08), min_arrow_len=4,
            arrowprops={"arrowstyle": "-", "color": "#ffe680", "lw": 0.9},
        )
    return image


def _plot_chiba_panels(output: Path, lon: np.ndarray, lat: np.ndarray,
                       rainfall: np.ndarray, periods: np.ndarray,
                       city_segments: list[np.ndarray],
                       prefecture_segments: list[np.ndarray],
                       city_labels: list[tuple[str, float, float]]) -> None:
    import cartopy.crs as ccrs

    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})
    west, east, south, north = CHIBA_EXTENT
    x = np.flatnonzero((lon >= west - 0.02) & (lon <= east + 0.02))
    y = np.flatnonzero((lat >= south - 0.02) & (lat <= north + 0.02))
    lon = lon[x]
    lat = lat[y]
    rainfall = rainfall[np.ix_(y, x)]
    periods = periods[np.ix_(y, x)]
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 10.5), dpi=220, constrained_layout=True,
        subplot_kw={"projection": ccrs.Mercator(central_longitude=140.0)},
    )
    bars = []
    for ax, values, scale, title, label, labels in (
        (axes[0], rainfall, _rainfall_color_scale(), "(a) 最大24時間降水量",
         "最大24時間降水量（mm）", city_labels),
        (axes[1], periods, _return_period_color_scale(), "(b) 再現期間",
         "再現期間（年）", city_labels),
    ):
        cmap, norm, bounds = scale
        image = _draw_chiba_panel(
            ax, lon, lat, values, cmap, norm, title,
            city_segments, prefecture_segments, labels,
        )
        bar = fig.colorbar(
            image, ax=ax, orientation="horizontal", extend="max", spacing="uniform",
            extendfrac=COLORBAR_EXTEND_FRACTION,
            ticks=bounds, pad=0.075, fraction=0.055,
        )
        bar.set_label(label, fontsize=12.5)
        bar.set_ticklabels([f"{tick:,}" for tick in (
            RAINFALL_TICKS if ax is axes[0] else RETURN_PERIOD_TICKS
        )])
        bar.ax.tick_params(labelsize=10.5)
        bars.append((ax, bar))
    fig.canvas.draw()
    # Include the extension triangle when matching each colorbar to its map width.
    fig.set_layout_engine(None)
    for ax, bar in bars:
        map_position = ax.get_position()
        bar_position = bar.ax.get_position()
        bar.ax.set_in_layout(False)
        bar.ax.set_axes_locator(None)
        bar.ax.set_box_aspect(None)
        bar.ax.set_aspect("auto")
        bar.ax.set_position([
            map_position.x0, bar_position.y0,
            map_position.width / (1 + COLORBAR_EXTEND_FRACTION), bar_position.height,
        ])
    fig.savefig(output)
    plt.close(fig)


def _event_names(event_start: date, event_end: date) -> tuple[str, str, str]:
    default_event = event_start == EVENT_START and event_end == EVENT_END
    tag = f"{event_start:%Y%m%d}_{event_end:%Y%m%d}"
    event_suffix = "event" if default_event else f"event_{tag}"
    final_name = (
        "nusdas_2006_2025_event_2026.nc" if default_event
        else f"nusdas_2006_2025_event_{tag}.nc"
    )
    map_suffix = "" if default_event else f"_{tag}"
    return event_suffix, final_name, map_suffix


def render_chiba_map(
    final_path: Path, output_dir: Path, event_start: date, event_end: date,
    city_shp: Path,
) -> None:
    """Rebuild the two-panel Chiba map from cached statistics."""
    if not final_path.is_file():
        raise FileNotFoundError(f"computed return-period file missing: {final_path}")
    city_shp = city_shp.resolve(strict=True)
    with netcdf_file(final_path, "r", mmap=False) as dataset:
        lon = dataset.variables["lon"][:].copy()
        lat = dataset.variables["lat"][:].copy()
        rainfall = dataset.variables["event_max_24h_mm"][:].copy()
        periods = dataset.variables["return_period_years"][:].copy()
    rainfall[rainfall == FILL] = np.nan
    periods[periods == FILL] = np.nan
    _, _, map_suffix = _event_names(event_start, event_end)
    city_labels = _load_chiba_labels(city_shp)
    city_segments = _load_city_boundaries(city_shp, CHIBA_EXTENT)
    prefecture_segments, prefecture_count = _load_dissolved_prefecture_boundaries(
        city_shp, CHIBA_EXTENT,
    )
    _plot_chiba_panels(
        output_dir / f"return_period_chiba{map_suffix}.png",
        lon, lat, rainfall, periods, city_segments, prefecture_segments, city_labels,
    )
    (output_dir / f"return_period_chiba{map_suffix}.md").write_text(
        f"{event_start}～{event_end}の解析雨量RR60による格子別最大24時間降水量"
        "（1時間刻み）と、2006–2025年の年最大値に定常GEVを当てはめた再現期間。"
        "(a)は事例最大値（mm）。区分境界は1、10、20、50、100、200、300、400 mmで、"
        "400 mm超は右端の延長、1 mm未満と欠測は白。"
        "気象庁の2020年配色指針・表2-1の解析雨量8色を、この24時間雨量の区分に順に割り当てた。"
        "元の気象庁の区分は1時間雨量用であり、この図の閾値とは異なる。"
        "(b)は格子別再現期間（年）。inferno_rの区分境界は10、20、50、100、200、500、"
        "1,000、2,000、5,000、10,000年。10,000年超は右端の延長、10年未満・推定不能は白。"
        "両図ともメルカトル図法、東経139.6–140.9°・北緯34.8–36.2°。"
        "細い黒線は千葉県内の市区町村境界のみ。太い黒線（白縁）はcity.shpを"
        "regioncode先頭2桁で結合した都道府県境界。"
        f"両パネルの名称は千葉県内{len(city_labels)}件のname属性。"
        "気象庁配色指針: https://www.jma.go.jp/jma/kishou/info/colorguide/"
        "HPColorGuide_202007.pdf\n"
    )
    metadata_name = "metadata.json" if not map_suffix else f"metadata{map_suffix}.json"
    metadata_path = output_dir / metadata_name
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["map_panels"] = ["event_max_24h_mm", "return_period_years"]
        metadata["rainfall_palette_source"] = (
            "https://www.jma.go.jp/jma/kishou/info/colorguide/HPColorGuide_202007.pdf"
        )
        metadata["rainfall_palette_rgb_low_to_high"] = JMA_RAINFALL_RGB
        metadata["rainfall_ticks_mm"] = RAINFALL_TICKS
        metadata["return_period_palette"] = "inferno_r"
        metadata["return_period_ticks_years"] = RETURN_PERIOD_TICKS
        metadata["chiba_boundary_source"] = str(city_shp)
        metadata["chiba_boundary_shp_sha256"] = hashlib.sha256(
            city_shp.read_bytes()
        ).hexdigest()
        metadata["chiba_municipal_boundary_filter"] = "regioncode LIKE '12%'"
        metadata["chiba_label_count"] = len(city_labels)
        metadata["chiba_label_filter"] = "regioncode LIKE '12%'"
        metadata["chiba_label_field"] = "name"
        metadata["chiba_projection"] = "Mercator (central_longitude=140.0)"
        metadata["chiba_extent_lon_lat"] = list(CHIBA_EXTENT)
        metadata["prefecture_dissolve_field"] = "regioncode[:2]"
        metadata["prefecture_dissolve_group_count"] = prefecture_count
        metadata.pop("map_colormap", None)
        metadata.pop("map_return_period_ticks_years", None)
        metadata.pop("map_under_10_years", None)
        metadata.pop("map_over_10000_years", None)
        metadata.pop("boundary_source", None)
        metadata.pop("boundary_shp_sha256", None)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def _process_year(db_root: Path, output_dir: Path, year: int) -> tuple[int, int]:
    path = output_dir / f"annual_{year}.nc"
    with nusdas_workdir(db_root):
        reader = RR60Reader()
        if reader.shape != (GRID_Y, GRID_X):
            raise ValueError(f"unexpected NuSDaS grid shape: {reader.shape}")
        aggregate = aggregate_period(
            reader, db_root, date(year, 1, 1), date(year, 12, 31),
            output_dir / f"input_status_{year}.csv",
            output_dir / f"missing_times_{year}.csv",
        )
        temporary = path.with_suffix(".nc.tmp")
        _write_field(
            temporary, aggregate.maximum, aggregate.valid_windows,
            reader.lon, reader.lat,
        )
        temporary.replace(path)
        return year, int(np.isfinite(aggregate.maximum).sum())


def run(db_root: Path, output_dir: Path, *, first_year: int = 2006,
        last_year: int = 2025, final: bool = True, workers: int = 4,
        event_start: date = EVENT_START, event_end: date = EVENT_END,
        city_shp: Path = DEFAULT_CITY_SHP) -> None:
    db_root = db_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for year in range(first_year, last_year + 1):
        path = output_dir / f"annual_{year}.nc"
        if path.exists():
            print(f"reuse {path}", flush=True)
        else:
            pending.append(year)
    if workers == 1:
        for year in pending:
            print(f"processing {year}", flush=True)
            finished, count = _process_year(db_root, output_dir, year)
            print(f"completed {finished}; valid cells={count}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_year, db_root, output_dir, year): year
                for year in pending
            }
            for future in as_completed(futures):
                finished, count = future.result()
                print(f"completed {finished}; valid cells={count}", flush=True)
    if not final:
        return
    event_suffix, final_name, map_suffix = _event_names(event_start, event_end)
    with nusdas_workdir(db_root):
        reader = RR60Reader()
        if reader.shape != (GRID_Y, GRID_X):
            raise ValueError(f"unexpected NuSDaS grid shape: {reader.shape}")
        annual_paths = [output_dir / f"annual_{year}.nc" for year in YEARS]
        absent = [str(path) for path in annual_paths if not path.exists()]
        if absent:
            raise RuntimeError(f"annual files missing: {absent}")
        event = aggregate_period(
            reader, db_root, event_start, event_end,
            output_dir / f"input_status_{event_suffix}.csv",
            output_dir / f"missing_times_{event_suffix}.csv",
        )
        final_path = output_dir / final_name
        temporary = final_path.with_suffix(".nc.tmp")
        _write_final(temporary, annual_paths, event, reader, event_start, event_end)
        temporary.replace(final_path)
        with netcdf_file(final_path, "r", mmap=False) as dataset:
            periods = dataset.variables["return_period_years"][:].copy()
        periods[periods == FILL] = np.nan
        metadata = {
            "source": str(db_root), "product": "_SRFLLLY/AASV/PR10 RR60LV nrd=60",
            "years": [2006, 2025], "event_dates": [str(event_start), str(event_end)],
            "annual_missing_policy": "missing hours invalidate crossing windows; available annual maxima retained",
            "fit": "stationary GEV, sample L-moments, 20 complete annual maxima per cell",
            "event_in_fit": False,
            "valid_event_cells": int(np.isfinite(event.maximum).sum()),
            "valid_return_period_cells": int((~np.isnan(periods)).sum()),
            "infinite_return_period_cells": int(np.isinf(periods).sum()),
        }
        metadata_name = "metadata.json" if not map_suffix else f"metadata{map_suffix}.json"
        (output_dir / metadata_name).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        render_chiba_map(final_path, output_dir, event_start, event_end, city_shp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=Path("/mnt/yt02/db_an.nus"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/nusdas_2006_2025")
    )
    parser.add_argument("--first-year", type=int, default=2006)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--annual-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--event-start", type=date.fromisoformat, default=EVENT_START)
    parser.add_argument("--event-end", type=date.fromisoformat, default=EVENT_END)
    parser.add_argument("--city-shp", type=Path, default=DEFAULT_CITY_SHP)
    parser.add_argument(
        "--render-existing", action="store_true",
        help="redraw the Chiba two-panel map from the existing final NetCDF",
    )
    args = parser.parse_args()
    if not 2006 <= args.first_year <= args.last_year <= 2025:
        parser.error("year range must be within 2006..2025")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.event_start > args.event_end:
        parser.error("--event-start must be on or before --event-end")
    if args.event_start.year <= 2025:
        parser.error("event must start after the 2006–2025 fit period")
    if args.render_existing:
        _, final_name, _ = _event_names(args.event_start, args.event_end)
        output_dir = args.output_dir.resolve()
        render_chiba_map(
            output_dir / final_name, output_dir,
            args.event_start, args.event_end, args.city_shp,
        )
        return
    run(args.db_root, args.output_dir, first_year=args.first_year,
        last_year=args.last_year, final=not args.annual_only, workers=args.workers,
        event_start=args.event_start, event_end=args.event_end,
        city_shp=args.city_shp)


if __name__ == "__main__":
    main()
