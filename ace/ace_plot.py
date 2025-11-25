import sys, os, re
from datetime import datetime
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import splev, splrep

sys.path.append("..")
sys.path.append("../eqwaves/")

from atcf_2025_loader2 import load_atcf_dataset_for_2025
from timeseries import segment_tracks

HURDAT_DIR = "hurdat_data_clim"
ATCF_DIR = "../atcf_btk_2025"

CLIM_START = 1991
CLIM_END = 2020
TARGET_YEAR = 2025

def open_files(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        data.append(open(os.path.join(data_dir, filename)).read())

    return data

def wind_to_col_label(max_wind):
    if max_wind < 34:
        return 'grey', 'TD'
    elif max_wind < 64:
        return 'orange', 'TS'
    elif max_wind < 96:
        return 'red', 'H'
    else:
        return 'magenta', 'MH'

def parse_hurdat2_text(txt: str) -> pd.DataFrame:
    """
    Parse Atlantic HURDAT2 text.

    Returns DataFrame with columns:
        datetime, year, wind, status, ace_contrib, day_of_year
    """
    records = []
    current_storm = None

    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Header line: storm ID, name, etc.
        if line.upper().startswith("AL"):
            parts = [p.strip() for p in line.split(",")]
            current_storm = {
                "id": parts[0] if len(parts) > 0 else "",
                "name": parts[1] if len(parts) > 1 else "",
            }
            continue

        # Data line
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue

        date_str = parts[0]   # YYYYMMDD
        time_str = parts[1]   # HHMM
        # parts[2] = record identifier (e.g. "0", "1", etc.)
        status = parts[3]     # TD, TS, HU, EX, SD, SS, LO, etc.
        wind_str = parts[6]   # max sustained wind (kt)

        try:
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        except Exception:
            continue

        try:
            wind = int(wind_str)
        except Exception:
            continue

        records.append(
            {
                "datetime": dt,
                "year": dt.year,
                "wind": wind,
                "status": status,
                "storm_id": current_storm["id"] if current_storm else "",
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid HURDAT2 data parsed.")

    # Only tropical + subtropical stages are used for ACE
    tropical_statuses = {"TD", "TS", "HU", "SD", "SS"}
    df = df[df["status"].isin(tropical_statuses)].copy()

    # ACE contribution: only for winds >= 34 kt
    df["ace_contrib"] = df["wind"].apply(
        lambda v: (v * v * 1e-4) if v >= 34 else 0.0
    )
    df["day_of_year"] = df["datetime"].dt.dayofyear

    return df


def build_climatology(df_hurdat: pd.DataFrame,
                      start_year: int,
                      end_year: int) -> pd.Series:
    """
    Build daily climatological mean cumulative ACE between start_year and end_year.
    """
    # Filter to climatology window
    clim_df = df_hurdat[
        (df_hurdat["year"] >= start_year) &
        (df_hurdat["year"] <= end_year)
    ].copy()

    if clim_df.empty:
        raise RuntimeError("No HURDAT2 data in climatology window.")

    # Daily ACE per year (raw)
    daily = clim_df.groupby(["year", "day_of_year"])["ace_contrib"].sum()

    # Create a full DOY index (1–366)
    full_doy = pd.Index(range(1, 367), name="day_of_year")

    # Build full daily series for each year
    daily_full_list = []
    for yr in range(start_year, end_year + 1):
        s = daily.loc[yr] if yr in daily.index.levels[0] else pd.Series(dtype=float)
        s = s.reindex(full_doy, fill_value=0)  # <-- THIS is the key fix
        s_cum = s.cumsum()
        s_cum.name = yr
        daily_full_list.append(s_cum)

    # Combine into a dataframe: index=DOY, columns=years
    clim_df_full = pd.concat(daily_full_list, axis=1)

    # Mean cumulative ACE across years (smooth curve)
    clim_mean = clim_df_full.mean(axis=1)

    return clim_mean


# -------------------- ATCF BTK --------------------

def parse_atcf_btk_text(txt: str) -> pd.DataFrame:
    """
    Parse ATCF best-track (.dat) file for Atlantic.

    Typical line (comma-separated):
      AL, 01, 2025062118,  , BEST, 0, 298N, 574W, 20, 1017, ...
    """
    records = []

    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue

        basin = parts[0].upper()
        if basin != "AL":
            continue

        dt_token = re.sub(r"\D", "", parts[2])
        if not dt_token:
            continue

        # Normalize datetime: YYYYMMDDHH or YYYYMMDDHHMM
        if len(dt_token) == 10:  # YYYYMMDDHH
            dt_token = dt_token + "00"  # add minutes
        elif len(dt_token) > 12:
            dt_token = dt_token[:12]

        try:
            dt = datetime.strptime(dt_token, "%Y%m%d%H%M")
        except Exception:
            # last fallback: try without minutes
            try:
                dt = datetime.strptime(dt_token[:10], "%Y%m%d%H")
            except Exception:
                continue

        # Wind (kt) usually at index 8
        wind_str = re.sub(r"\D", "", parts[8])
        if not wind_str:
            continue

        try:
            wind = int(wind_str)
        except Exception:
            continue

        # Filter obviously wrong winds
        if wind < 0 or wind > 250:
            continue

        records.append(
            {
                "datetime": dt,
                "year": dt.year,
                "wind": wind,
            }
        )

    if not records:
        return pd.DataFrame(columns=["datetime", "year", "wind", "ace_contrib", "day_of_year"])

    df = pd.DataFrame(records)

    # Compute ACE contrib here or in the daily builder; either way is fine
    df["ace_contrib"] = df["wind"].apply(
        lambda v: (v * v * 1e-4) if v >= 34 else 0.0
    )
    df["day_of_year"] = df["datetime"].dt.dayofyear

    return df


def build_current_year_daily_from_atcf(
    dfs_atcf: list[pd.DataFrame],
    target_year: int | None = None,
) -> pd.Series:
    """
    Concatenate ATCF DataFrames, sanitize, and return DAILY ACE for target_year.

    Steps:
      - Concatenate all parsed ATCF records
      - Drop missing datetimes/winds
      - Filter by target_year (or mode year if None)
      - For duplicate datetimes, keep maximum wind
      - Compute ACE contrib and sum per day_of_year

    Returns a Series indexed by day_of_year with DAILY ACE (not cumulative).
    """
    if not dfs_atcf:
        return pd.Series(dtype=float)

    df_all = pd.concat(dfs_atcf, ignore_index=True)
    df_all = df_all.dropna(subset=["datetime", "wind"]).copy()

    # Force datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df_all["datetime"]):
        df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
        df_all = df_all.dropna(subset=["datetime"])

    # Choose target year if not provided
    if target_year is None:
        target_year = int(df_all["datetime"].dt.year.mode().iloc[0])

    df_all = df_all[df_all["datetime"].dt.year == int(target_year)].copy()
    if df_all.empty:
        return pd.Series(dtype=float)

    # In case we lost ace_contrib/day_of_year in earlier parsing
    df_all["ace_contrib"] = df_all["wind"].apply(
        lambda v: (v * v * 1e-4) if v >= 34 else 0.0
    )
    df_all["day_of_year"] = df_all["datetime"].dt.dayofyear

    # Deduplicate timestamps: keep max wind per datetime
    df_all = df_all.sort_values("datetime")
    grouped = (
        df_all.groupby("datetime", as_index=False)
        .agg({"wind": "max"})
        .sort_values("datetime")
    )

    grouped["ace_contrib"] = grouped["wind"].apply(
        lambda v: (v * v * 1e-4) if v >= 34 else 0.0
    )
    grouped["day_of_year"] = grouped["datetime"].dt.dayofyear

    # DAILY ACE
    daily = grouped.groupby("day_of_year")["ace_contrib"].sum().sort_index()

    return daily


# -------------------- MAIN --------------------

# 1. Open & parse HURDAT2
hurdat_txt = open_files(HURDAT_DIR)[0]
hurdat_df = parse_hurdat2_text(hurdat_txt)

# 2. Build 1991–2020 climatology
clim_mean = build_climatology(hurdat_df, CLIM_START, CLIM_END)

# 3. Load tracks
tracks, _ = load_atcf_dataset_for_2025()
tracks["datetime"] = pd.to_datetime(tracks["datetime"])
tracks["lon_plot"] = np.where(
    tracks["lon"] > 180,
    tracks["lon"] - 360,
    tracks["lon"]
)

# 4. Open & parse each ATCF file
dfs_atcf: list[pd.DataFrame] = []
atcf_data = open_files(ATCF_DIR)
print(atcf_data)
for chunk in atcf_data:
    try:
        df = parse_atcf_btk_text(chunk)
        if not df.empty:
            dfs_atcf.append(df)
    except Exception as e:
        print(f"Warning: could not parse {url}: {e}", file=sys.stderr)

# 5. Build DAILY ACE for target year from ATCF
current_daily = build_current_year_daily_from_atcf(dfs_atcf, TARGET_YEAR)

sigma = current_daily.std()
mu = current_daily.mean()
burstiness = (sigma - mu)/(sigma + mu)
print(burstiness, current_daily.kurt())
input()

# 6. Align and build cumulative series
max_day = max(
    int(current_daily.index.max()),
    int(clim_mean.index.max()),
)
idx = range(1, max_day + 1)

# Climatology: reindex, forward fill, then fill remaining NaNs with 0
clim_vals = clim_mean.reindex(idx).ffill().fillna(0).values

# Current year: reindex DAILY ACE (missing days = 0), then cumsum
curr_daily_reindexed = current_daily.reindex(idx, fill_value=0)
curr_vals = curr_daily_reindexed.cumsum().values

# 7. Plot
plt.figure(figsize=(8, 3), constrained_layout=True)
plt.plot(
    list(idx),
    clim_vals,
    label=f"{CLIM_START}–{CLIM_END} climatology (HURDAT2)",
    color='k'
)
plt.plot(
    list(idx)[:304],
    curr_vals[:304],
    label=f"{TARGET_YEAR} (ATCF realtime)",
    color='b'
)

plt.ylim(0, 200)
plt.xlabel("Day of year")
plt.ylabel(r"Cumulative ACE ($10^{-4}$ kt$^2$)")

ax = plt.gca()

end_date = "2025-12-01"
tick_dates = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01"]
dates = pd.date_range("2025-01-01", "2025-12-01")
xticks = [dates.get_loc(date_str)+1 for date_str in tick_dates]

ax.set_xticks(xticks)
ax.set_xticklabels(tick_dates)
ax.set_xlabel("Date")

bar_ax = inset_axes(ax, width="100%", height="26%",
                    loc='upper left', bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=ax.transAxes, borderpad=0)

plot_dates = pd.date_range("2025-05-01", "2025-12-01")
bar_ax.set_xlim(plot_dates[0], plot_dates[-1])
ax.set_xlim(dates.get_loc(plot_dates[0])+1, dates.get_loc(plot_dates[-1])+1)

bar_ax.set_ylim(0, 1)
bar_ax.set_xticklabels([])
bar_ax.set_yticks([])

segments = segment_tracks(tracks)
y0 = 0.2
y = y0
dy = 0.3

for i, seg in enumerate(segments):
    seg = seg.sort_values("datetime")
    time_pts = seg["datetime"].values
    wind = seg["wind_kts"].values

    offset = pd.Timedelta(days=2)

    name = seg["name"].unique()[-1]
    if name in ['ERIN', 'GABRIELLE', 'HUMBERTO', 'IMELDA']:
        bar_ax.text(seg["datetime"].values[0]-offset, y, name, va='center_baseline', ha='right',
                fontsize=8, fontweight='bold')
    elif name in ['MELISSA']:
        bar_ax.text(seg["datetime"].values[-1]+offset, y, name, va='center_baseline', ha='left',
                fontsize=8, fontweight='bold')

    #for j in range(len(seg)-1):
        #col, lab = wind_to_col_label(wind[j])
        #bar_ax.plot([time_pts[j], time_pts[j+1]],
                    #[y, y], "s", color=col, label=lab, zorder=-10, ms=5)
        #bar_ax.plot([time_pts[j], time_pts[j+1]],
                    #[y, y],
                    #color=col, lw=8)

    xs = mdates.date2num(time_pts)
    ys = [y for i in range(len(xs))]
    cols = [wind_to_col_label(w)[0] for w in wind]
    lwidths=1+0.1*wind
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    eps = 5e-2
    lsegments = np.array([
        [[xs[i] - eps, ys[i]], [xs[i+1] + eps, ys[i+1]]]
        for i in range(len(xs)-1)
    ])

    #lsegments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(lsegments, linewidths=lwidths, colors=cols, capstyle='butt', joinstyle='miter')
    bar_ax.add_collection(lc)

    y += dy
    if y > 0.9:
        y = y0

# Legend

#legend_handles = [Patch(facecolor=wind_to_col_label(w)[0], edgecolor=None, label=wind_to_col_label(w)[1]) for w in [10, 50, 80, 100] ]
#bar_ax.legend(handles=legend_handles, loc="center left", ncol=2,
        #labelspacing=0.25, columnspacing=1.0)

leg_ax = inset_axes(bar_ax, width="20%", height="70%",
                    loc='upper left', bbox_to_anchor=(0.03, 0.70, 0.7, 0.27),
                    bbox_transform=ax.transAxes, borderpad=0)

wind = np.array([20, 50, 70, 100])
xs = np.array([0.1, 0.3, 0.5, 0.75, 0.9])
ys = [0.4 for i in range(len(xs))]
cols = [wind_to_col_label(w)[0] for w in wind]
lwidths=1+0.1*wind
points = np.array([xs, ys]).T.reshape(-1, 1, 2)
lsegments = np.array([
    [[xs[i] - eps, ys[i]], [xs[i+1] + eps, ys[i+1]]]
    for i in range(len(xs)-1)
])
lc = LineCollection(lsegments, linewidths=lwidths, colors=cols, capstyle='butt', joinstyle='miter')
leg_ax.add_collection(lc)

labels = ['TD', 'TS', 'H', 'MH']
x_pos = [0.07, 0.29, 0.53, 0.73]
for x, l in zip(x_pos, labels):
    leg_ax.text(x, 0.7, l, fontsize=8, fontweight='bold')

leg_ax.set_axis_off()
#leg_ax.set_xticks([])
#leg_ax.set_yticks([])


for x in xticks:
    ax.axvline(x, color='grey', alpha=0.5, zorder=-10)

yticks = [25, 50, 75, 100, 125]
ax.set_yticks(yticks)
for y in yticks:
    ax.axhline(y, color='grey', alpha=0.5, zorder=-10)

leg = ax.legend(loc="center left", frameon=True, bbox_to_anchor=(0.01, 0.6))
leg.get_frame().set_alpha(1.0)

outname = f"ace_atlantic_{TARGET_YEAR}_vs_{CLIM_START}_{CLIM_END}.png"
plt.savefig(outname, dpi=300)
plt.show()
