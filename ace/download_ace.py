import sys, os, re
from datetime import datetime
from urllib.parse import urljoin

import requests
import pandas as pd
import matplotlib.pyplot as plt

HURDAT2_URL = (
    "https://www.nhc.noaa.gov/data/hurdat/"
    "hurdat2-1851-2024-040425.txt"
)
HURDAT_DIR = "hurdat_data_clim"

ATCF_BTK_INDEX = "https://ftp.nhc.noaa.gov/atcf/btk/"
ATCF_DIR = "../atcf_data_2025"

# Climatology window
CLIM_START = 1991
CLIM_END = 2020

# Target year to plot from ATCF (change this if you want another year)
TARGET_YEAR = 2025

# ------------------------------------------------

session = requests.Session()
session.headers.update({"User-Agent": "atlantic-ace-script/1.0"})


# -------------------- UTILITIES --------------------

def download_text(url: str) -> str:
    """Download a text file and return as string."""
    print(f"Downloading: {url}")
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.text


# -------------------- HURDAT2 --------------------

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


# -------------------- ATCF BTK --------------------

def list_atcf_btk_files() -> list[str]:
    """Return list of absolute URLs for .dat files in ATCF BTK directory."""
    print(f"Listing ATCF BTK directory: {ATCF_BTK_INDEX}")
    r = session.get(ATCF_BTK_INDEX, timeout=60)
    r.raise_for_status()
    html = r.text

    # Find all href="*.dat"
    files = re.findall(r'href="([^"]+\.dat)"', html)
    urls = [urljoin(ATCF_BTK_INDEX, f) for f in files]
    return urls


def select_atcf_files_for_year(files: list[str], year: int) -> list[str]:
    """
    Select Atlantic ATCF BTK files for a given year.

    Filenames often look like 'bal012025.dat' or 'al022025.dat'.
    We filter by:
        - contains the year (e.g. '2025')
        - starts with 'al' or 'bal' (Atlantic)
    """
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

    # Fallback: any .dat with 'al' and year_str
    if not selected:
        for url in files:
            name = url.split("/")[-1].lower()
            if year_str in name and "al" in name and name.endswith(".dat"):
                selected.append(url)

    selected = sorted(set(selected))
    return selected



# -------------------- MAIN --------------------

# 1. Download & parse HURDAT2
hurdat_txt = download_text(HURDAT2_URL)

print("Saving HURDAT2...")
with open(os.path.join(HURDAT_DIR, "hurdat.txt"), 'w') as f:
    f.write(hurdat_txt)
    f.close()

# 3. List ATCF BTK files and select those for target year
all_btk_files = list_atcf_btk_files()
atcf_files = select_atcf_files_for_year(all_btk_files, TARGET_YEAR)

print(f"Found {len(atcf_files)} ATCF files for {TARGET_YEAR}.")

# 4. Download & parse each ATCF file
for url in atcf_files:
    txt = download_text(url)

    with open(os.path.join(ATCF_DIR, url.split('/')[-1]), 'w') as f:
        f.write(txt)
        f.close()

print("Done.")
