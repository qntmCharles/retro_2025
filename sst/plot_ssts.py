import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from datetime import datetime, timedelta
import datetime as dt

year = 2025
base_date = dt.date(year, 1, 1)

def doy_to_datestr(doy_value):
    """Convert day-of-year (float) to a date string for 2025."""
    # Day-of-year may be float (because of tick locations), round safely
    d = int(round(doy_value))
    if d < 1 or d > 365:
        return ""
    date = base_date + dt.timedelta(days=d-1)
    return date.strftime("%d %b")

def date_formatter(x, pos):
    return doy_to_datestr(x)

df = pd.read_csv("oisst_mdr_daily.csv", parse_dates=["time"])

fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

years = sorted(df["year"].unique())
recent_years = years[-3:]
latest_year = years[-1]


# Highlight last 3 years
highlight_colors = ['dodgerblue', 'b', 'r']
for y, col in zip(recent_years, highlight_colors):
    d = df[df["year"] == y]
    ax.plot(d["doy"], d["sst"], color=col, linewidth=2, label=str(y), zorder=10)

# Historical years
for y in years:
    if y not in recent_years:
        d = df[df["year"] == y]
        ax.plot(d["doy"], d["sst"], color="gray", linewidth=1, alpha=0.5)

ax.plot(d["doy"], d["sst"], color="gray", linewidth=1, alpha=0.5, label="1991-2020")

# Climatology 1991–2020
clim = (
    df[df["time"].between("1991-01-01", "2020-12-31")]
    .groupby("doy")["sst"]
    .mean()
)
ax.plot(clim.index, clim.values, "k--", linewidth=2, label="1991–2020 mean")

ax.set_xlim(0, 365)
ax.set_xlabel("Day of year")
ax.set_ylabel(r"MDR daily average SST ($^\circ$C)")

ax.legend(ncol=2)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.spines["bottom"].set_position(("outward", 40))
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")

xticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
ax2.set_xticks(xticks)
ax2.set_xticklabels(["Jan 1", "Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1", "Jul 1", "Aug 1", "Sep 1", "Oct 1", "Nov 1", "Dec 1"])  # optional
ax2.set_xlabel(f"Date in {year}")

for x in xticks:
    ax.axvline(x, color='grey', alpha=0.5)

yticks = [25, 26, 27, 28, 29]
for y in yticks:
    ax.axhline(y, color='grey', alpha=0.5)

plt.savefig("daily_sst.png", dpi=300)
plt.show()
