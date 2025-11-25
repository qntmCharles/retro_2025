#!/usr/bin/env python3
"""
ERA5 JJASO and SO (2025) anomalies: MSLP, 850‑200hPa lapse rate, relative SST, and 500hPa RH.
Compared to 1990‑2020 climatology.
"""

import cdsapi
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
OUTDIR = "./output"
os.makedirs(OUTDIR, exist_ok=True)

MSL_FILE_CLIM = os.path.join(OUTDIR, "era5_msl_sst_climatology.nc")
MSL_FILE_2025 = os.path.join(OUTDIR, "era5_msl_sst_2025.nc")
PL_FILE_CLIM  = os.path.join(OUTDIR, "era5_pl_climatology.nc")
PL_FILE_2025  = os.path.join(OUTDIR, "era5_pl_2025.nc")

CLIM_YEARS     = list(range(1990, 2021))
TARGET_YEAR    = 2025
#MONTHS_JJASO     = [6,7,8,9,10]
MONTHS_JJASO     = [6,7,8]
PRESS_LEVELS   = ["850","200","500"]  # added 500 hPa for RH
ATL_LONGITUDE  = (-100, 20)            # for Atlantic tropical mean
TROP_LATITUDE  = (20, -20)

# --------------------------------------------------
# (Optional) DOWNLOAD FUNCTIONS
# --------------------------------------------------
# --------------------------------------------------
# DOWNLOAD FUNCTIONS
# --------------------------------------------------
def cds_download_single_monthly(years, months, target, variable):
    """Download ERA5 single-level monthly means (MSLP, SST)."""
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variable,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "time": "00:00",
            "area": [60, -100, 0, 20],
            "format": "netcdf",
        },
        target,
    )


def cds_download_pressure_monthly(years, months, target, levels=None):
    """Download ERA5 pressure-level monthly means including relative humidity."""
    import cdsapi
    c = cdsapi.Client()
    req = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ["temperature", "geopotential", "relative_humidity"],
        "pressure_level": levels if levels else ["850","200","500"],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "time": "00:00",
        "area": [60, -100, 0, 20],
        "format": "netcdf",
    }
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        req,
        target,
    )

# Download climatology (1990-2020)
#cds_download_single_monthly(CLIM_YEARS, MONTHS_JJASO, MSL_FILE_CLIM,
                            #variable=["mean_sea_level_pressure","sea_surface_temperature"])
#cds_download_pressure_monthly(CLIM_YEARS, MONTHS_JJA, PL_FILE_CLIM,
                              #levels=PRESS_LEVELS)

# Download 2025 data
#cds_download_single_monthly([TARGET_YEAR], MONTHS_JJA, MSL_FILE_2025,
                            #variable=["mean_sea_level_pressure","sea_surface_temperature"])
#cds_download_pressure_monthly([TARGET_YEAR], MONTHS_JJA, PL_FILE_2025,
                              #levels=PRESS_LEVELS)

# --------------------------------------------------
# OPEN DATASETS
# --------------------------------------------------
print("Opening datasets...")


ds_msl_clim = xr.open_dataset(MSL_FILE_CLIM)
ds_msl_2025 = xr.open_dataset(MSL_FILE_2025)
ds_pl_clim  = xr.open_dataset(PL_FILE_CLIM)
ds_pl_2025  = xr.open_dataset(PL_FILE_2025)

# Correct renaming without inplace
if "valid_time" in ds_msl_clim.coords: ds_msl_clim = ds_msl_clim.rename({"valid_time": "time"})
if "valid_time" in ds_msl_2025.coords: ds_msl_2025 = ds_msl_2025.rename({"valid_time": "time"})
if "valid_time" in ds_pl_clim.coords: ds_pl_clim = ds_pl_clim.rename({"valid_time": "time"})
if "valid_time" in ds_pl_2025.coords: ds_pl_2025 = ds_pl_2025.rename({"valid_time": "time"})

if "pressure_level" in ds_pl_clim.coords: ds_pl_clim = ds_pl_clim.rename({"pressure_level": "level"})
if "pressure_level" in ds_pl_2025.coords: ds_pl_2025 = ds_pl_2025.rename({"pressure_level": "level"})

def get_var(ds, candidates):
    for c in candidates:
        if c in ds: return ds[c]
    raise KeyError(f"None of {candidates} found in {list(ds.data_vars)}")

# Single level
msl_clim = get_var(ds_msl_clim, ["msl","mean_sea_level_pressure"]) / 100.0
msl_2025 = get_var(ds_msl_2025, ["msl","mean_sea_level_pressure"]) / 100.0
sst_clim = get_var(ds_msl_clim, ["sst","sea_surface_temperature"])
sst_2025 = get_var(ds_msl_2025, ["sst","sea_surface_temperature"])

# Pressure levels
t_clim  = get_var(ds_pl_clim, ["t","temperature"])
z_clim  = get_var(ds_pl_clim, ["z","geopotential"])
rh_clim = get_var(ds_pl_clim, ["r","relative_humidity"])
t_2025  = get_var(ds_pl_2025, ["t","temperature"])
z_2025  = get_var(ds_pl_2025, ["z","geopotential"])
rh_2025 = get_var(ds_pl_2025, ["r","relative_humidity"])

# Lapse rate K/km
def compute_lapse_rate(t, z):
    g0 = 9.80665
    z_m = z / g0
    t850 = t.sel(level=850, method="nearest")
    t200 = t.sel(level=200, method="nearest")
    z850 = z_m.sel(level=850, method="nearest")
    z200 = z_m.sel(level=200, method="nearest")
    lapse = (t850 - t200) / (z200 - z850) * 1000.0
    lapse.name = "lapse_rate"
    return lapse

lapse_clim_all = compute_lapse_rate(t_clim, z_clim)
lapse_2025_all = compute_lapse_rate(t_2025, z_2025)

# Seasonal helpers
def seasonal_mean_clim(da, months):
    return da.sel(time=da.time.dt.month.isin(months)).groupby("time.year").mean("time").mean("year")
def seasonal_mean_year(da, months):
    return da.sel(time=da.time.dt.month.isin(months)).mean("time")

def compute_fields(months):
    msl_c, msl_y       = seasonal_mean_clim(msl_clim, months), seasonal_mean_year(msl_2025, months)
    lapse_c, lapse_y   = seasonal_mean_clim(lapse_clim_all, months), seasonal_mean_year(lapse_2025_all, months)
    sst_c, sst_y       = seasonal_mean_clim(sst_clim, months), seasonal_mean_year(sst_2025, months)
    rh_c, rh_y         = seasonal_mean_clim(rh_clim.sel(level=500, method="nearest"), months), \
                         seasonal_mean_year(rh_2025.sel(level=500, method="nearest"), months)

    msl_anom   = msl_y - msl_c
    lapse_anom = lapse_y - lapse_c
    sst_anom   = sst_y - sst_c
    rh_anom    = rh_y - rh_c

    # Atlantic tropical relative SST
    lat_rad = np.deg2rad(sst_anom.latitude)
    weights = np.cos(lat_rad)
    weights.name = "weights"
    trop_mean = (sst_anom.sel(latitude=slice(TROP_LATITUDE[0], TROP_LATITUDE[1]),
                              longitude=slice(ATL_LONGITUDE[0], ATL_LONGITUDE[1]))
                 .weighted(weights.sel(latitude=slice(TROP_LATITUDE[0], TROP_LATITUDE[1])))
                 .mean(("latitude","longitude")))
    rel_sst_anom = sst_anom - trop_mean
    rel_sst_anom.name = "relative_sst_anomaly"

    return msl_anom, lapse_anom, rel_sst_anom, rh_anom

print("Computing JJASO anomalies…")
msl_JJASO, lapse_JJASO, relsst_JJASO, rh_JJASO = compute_fields(MONTHS_JJASO)

# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
print("Plotting results…")

import matplotlib as mpl
import matplotlib.font_manager as fm

fe = fm.FontEntry(
    fname='/home/cwp29/Documents/inigo/retro_2025/mslp_lapse/Overpass/static/Overpass-Regular.ttf',
    name='Overpass')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'

plt.style.use("inigo.mplstyle")

fig, axes = plt.subplots(2, 2, figsize=(10,6), subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True)

def plot_panel(ax, da, title, cmap, levels, units):
    im = ax.contourf(da.longitude, da.latitude, da, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend="both")
    ax.set_extent([-100, 0, 2, 50], crs=ccrs.PlateCarree())
    ax.coastlines(); ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, location="right", pad=0.05, shrink=0.5)
    cb.set_label(units)

msl_levs = np.linspace(-2, 2, 11)
lr_levs = np.linspace(-0.2, 0.2, 11)
sst_levs = np.linspace(-1, 1, 11)
rh_levs = np.linspace(-5, 5, 11)

plot_panel(axes[0, 0], msl_JJASO, "MSLP Anomaly (hPa)",
        "RdBu_r", msl_levs, "hPa")
plot_panel(axes[1, 0], lapse_JJASO, "Lapse-rate Anomaly (K km⁻¹)",
        "BrBG", lr_levs, "K km⁻¹")
plot_panel(axes[0, 1], relsst_JJASO, "Relative SST Anomaly (K)",
        "RdYlBu_r", sst_levs, "K")
plot_panel(axes[1, 1], rh_JJASO,    "500hPa RH Anomaly (%)",
        "BrBG",  rh_levs, "%")

plt.suptitle("ERA5 Jun-Aug anomalies vs 1990-2020", fontsize=15, y=0.98)
out_fig = os.path.join(OUTDIR, "mslp_lapse_relSST_RH_JJA_2025.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print("Saved figure to:", out_fig)
plt.show()

