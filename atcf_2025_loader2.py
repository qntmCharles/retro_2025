#!/usr/bin/env python3
"""
ATCF loader fixed to match the working logic from ace_plot.py.

This script:
  ✔ Scrapes ATCF BTK index directory
  ✔ Selects files for Atlantic + 2025
  ✔ Downloads each .dat file
  ✔ Parses BTK format exactly as in ace_plot.py
  ✔ Returns:
        tracks_df  – 6-hourly records
        tc_daily   – daily n_storms + daily_ACE
"""

import os
import re
from datetime import datetime
from urllib.parse import urljoin

import requests
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Constants (identical to ace_plot.py)
# -------------------------------------------------------------------
ATCF_BTK_INDEX = "https://ftp.nhc.noaa.gov/atcf/btk/"
session = requests.Session()
session.headers.update({"User-Agent": "atcf-loader/1.0"})


# -------------------------------------------------------------------
# 1. Scrape BTK directory
# -------------------------------------------------------------------
def list_atcf_btk_files():
    """Return list of absolute URLs for .dat files in BTK directory."""
    r = session.get(ATCF_BTK_INDEX, timeout=60)
    r.raise_for_status()
    html = r.text

    files = re.findall(r'href="([^"]+\.dat)"', html)
    urls = [urljoin(ATCF_BTK_INDEX, f) for f in files]
    return urls


def select_atcf_files_for_year(files, year=2025):
    """Select Atlantic (“al”, “bal”) files for a specific year."""
    selected = []
    year_str = str(year)

    for url in files:
        name = url.split("/")[-1].lower()
        if not name.endswith(".dat"):
            continue
        if year_str not in name:
            continue
        if name.startswith("al") or name.startswith("bal"):
            selected.append(url)

    return sorted(set(selected))


# -------------------------------------------------------------------
# 2. Parse BTK text content into DataFrame
# -------------------------------------------------------------------
def parse_atcf_btk_text(txt):
    """Parse raw BTK file text into DataFrame with datetime, wind, lat, lon."""
    records = []

    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue

        basin = parts[0].upper()
        if basin != "AL":
            continue

        # datetime field (might be YYYYMMDDHH or YYYYMMDDHHMM)
        dt_token = re.sub(r"\D", "", parts[2])
        if not dt_token:
            continue

        # Normalize length
        if len(dt_token) == 10:     # YYYYMMDDHH
            dt_token += "00"        # → add MM
        elif len(dt_token) > 12:
            dt_token = dt_token[:12]

        # Parse
        try:
            dt = datetime.strptime(dt_token, "%Y%m%d%H%M")
        except:
            continue

        # Wind (field 8)
        wind_str = re.sub(r"\D", "", parts[8])
        if not wind_str:
            continue

        try:
            wind = int(wind_str)
        except:
            continue

        if wind < 0 or wind > 250:
            continue

        # Name (last field)
        name = parts[27].strip().upper()

        # Lat/lon fields: 298N, 574W
        def coord(s):
            if not s:
                return np.nan
            v = float(s[:-1]) / 10.0
            hemi = s[-1].upper()
            if hemi in ("S", "W"):
                return -v
            return v

        try:
            lat = coord(parts[6])
            lon = coord(parts[7])
        except:
            lat = lon = np.nan

        records.append({
            "datetime": dt,
            "lat": lat,
            "lon": lon,
            "wind_kts": wind,
            "name": name
        })

    if not records:
        return pd.DataFrame(columns=["datetime", "lat", "lon", "wind_kts", "name"])

    df = pd.DataFrame(records)
    return df


# -------------------------------------------------------------------
# 3. Download + parse all selected ATCF BTK files
# -------------------------------------------------------------------
def load_atcf_tracks_2025(dest_dir="atcf_btk_2025"):
    os.makedirs(dest_dir, exist_ok=True)

    # Scrape directory
    #all_files = list_atcf_btk_files()
    #selected = select_atcf_files_for_year(all_files, 2025)

    #if not selected:
        #raise RuntimeError("No ATCF files found for 2025.")

    dfs = []

    #for url in selected:
        #fname = url.split("/")[-1]
        #local = os.path.join(dest_dir, fname)

        #if not os.path.exists(local):
            #r = session.get(url, timeout=60)
            #r.raise_for_status()
            #with open(local, "w") as f:
                #f.write(r.text)

    for fname in os.listdir(dest_dir):
        local = os.path.join(dest_dir, fname)
        df = parse_atcf_btk_text(open(local).read())
        if not df.empty:
            # Use the full base filename as a unique storm identifier
            # e.g. "al012025.dat"  -> "AL012025"
            #      "bal012025.dat" -> "BAL012025"
            storm_id = os.path.splitext(fname)[0].upper()
            df["storm_id"] = storm_id
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid ATCF data parsed for 2025.")

    tracks = pd.concat(dfs).sort_values("datetime").reset_index(drop=True)
    return tracks


# -------------------------------------------------------------------
# 4. Daily TC statistics (n_storms and daily ACE)
# -------------------------------------------------------------------
def daily_tc_stats(tracks):
    df = tracks.copy()
    df["date"] = df["datetime"].dt.date

    df["ace_interval"] = np.where(
        df["wind_kts"] >= 34,
        (df["wind_kts"] / 10.0)**2 / 4.0,
        0.0
    )

    daily = df.groupby("date").agg(
        n_storms=("datetime", "count"),  # BTK does not have storm IDs, so count records
        daily_ACE=("ace_interval", "sum")
    )

    return daily


# -------------------------------------------------------------------
# 5. Public function: load everything
# -------------------------------------------------------------------
def load_atcf_dataset_for_2025(dest_dir="/home/cwp29/Documents/papers/2025/retro_2025/atcf_btk_2025"):
    tracks = load_atcf_tracks_2025(dest_dir=dest_dir)
    daily = daily_tc_stats(tracks)
    daily.index = pd.to_datetime(daily.index)
    return tracks, daily


if __name__ == "__main__":
    tracks, daily = load_atcf_dataset_for_2025()
    print(tracks.head())
    print(daily.head())

