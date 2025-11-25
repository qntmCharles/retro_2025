import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.transforms import ScaledTranslation

# SETTINGS

OUTDIR = "./output"
MSL_FILE_CLIM = os.path.join(OUTDIR, "era5_msl_sst_climatology.nc")
MSL_FILE_2025 = os.path.join(OUTDIR, "era5_msl_sst_2025.nc")
PL_FILE_CLIM  = os.path.join(OUTDIR, "era5_pl_climatology.nc")
PL_FILE_2025  = os.path.join(OUTDIR, "era5_pl_2025.nc")

CLIM_YEARS     = list(range(1990, 2021))
TARGET_YEAR    = 2025
ATL_LONGITUDE  = (-100, 20)            # for Atlantic tropical mean
TROP_LATITUDE  = (20, -20)

print("Opening datasets...")


ds_msl_clim = xr.open_dataset(MSL_FILE_CLIM)
ds_msl_2025 = xr.open_dataset(MSL_FILE_2025)
ds_pl_clim  = xr.open_dataset(PL_FILE_CLIM)
ds_pl_2025  = xr.open_dataset(PL_FILE_2025)

# Rename "valid_time" to "time" for convenience
ds_msl_clim = ds_msl_clim.rename({"valid_time": "time"})
ds_msl_2025 = ds_msl_2025.rename({"valid_time": "time"})
ds_pl_clim = ds_pl_clim.rename({"valid_time": "time"})
ds_pl_2025 = ds_pl_2025.rename({"valid_time": "time"})

# Rename "pressure_level" to "level" for convenience
ds_pl_clim = ds_pl_clim.rename({"pressure_level": "level"})
ds_pl_2025 = ds_pl_2025.rename({"pressure_level": "level"})

# Single level
msl_clim = ds_msl_clim["msl"] / 100.0
msl_2025 = ds_msl_2025["msl"] / 100.0
sst_clim = ds_msl_clim["sst"]
sst_2025 = ds_msl_2025["sst"]

# Pressure levels
t_clim  = ds_pl_clim["t"]
z_clim  = ds_pl_clim["z"]
rh_clim = ds_pl_clim["r"]
t_2025  = ds_pl_2025["t"]
z_2025  = ds_pl_2025["z"]
rh_2025 = ds_pl_2025["r"]

# Lapse rate K/km
def compute_lapse_rate(t, z):
    g0 = 9.80665
    z_m = z / g0
    t850 = t.sel(level=850)
    t200 = t.sel(level=200)
    z850 = z_m.sel(level=850)
    z200 = z_m.sel(level=200)
    lapse = (t850 - t200) / (z200 - z850) * 1000.0
    lapse.name = "lapse_rate"
    return lapse

lapse_clim_all = compute_lapse_rate(t_clim, z_clim)
lapse_2025_all = compute_lapse_rate(t_2025, z_2025)

def seasonal_mean_clim(da, months):
    return da.sel(time=da.time.dt.month.isin(months)).groupby("time.year").mean("time").mean("year")

def seasonal_mean_year(da, months):
    return da.sel(time=da.time.dt.month.isin(months)).mean("time")

def compute_fields(months):
    msl_c, msl_y       = seasonal_mean_clim(msl_clim, months), seasonal_mean_year(msl_2025, months)
    lapse_c, lapse_y   = seasonal_mean_clim(lapse_clim_all, months), seasonal_mean_year(lapse_2025_all, months)
    sst_c, sst_y       = seasonal_mean_clim(sst_clim, months), seasonal_mean_year(sst_2025, months)
    rh_c, rh_y         = seasonal_mean_clim(rh_clim.sel(level=500), months), \
                         seasonal_mean_year(rh_2025.sel(level=500), months)

    msl_anom   = msl_y - msl_c
    lapse_anom = lapse_y - lapse_c
    sst_anom   = sst_y - sst_c
    rh_anom    = rh_y - rh_c

    # Atlantic tropical relative SST
    lat_rad = np.deg2rad(sst_anom.latitude)
    weights = np.cos(lat_rad)
    weights.name = "weights"
    trop_mean = (sst_anom.sel(latitude=slice(20, -20), longitude=slice(-100, 20))
                 .weighted(weights.sel(latitude=slice(20, -20)))
                 .mean(("latitude","longitude")))
    rel_sst_anom = sst_anom - trop_mean
    rel_sst_anom.name = "relative_sst_anomaly"

    return msl_anom, lapse_anom, rel_sst_anom, rh_anom

print("Plotting results…")

month_sets = [[6, 7], [8, 9, 10]]

month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, axs = plt.subplots(len(month_sets), 4, figsize=(8, 3), constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()})

def plot_panel(ax, da, title, cmap, levels, units, cbar=True, showtitle=True):
    im = ax.contourf(da.longitude, da.latitude, da, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend="both")
    ax.set_extent([-95, -10, 2, 45], crs=ccrs.PlateCarree())
    ax.coastlines(); ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    if showtitle:
        ax.set_title(title, fontsize=10)
    if cbar:
        cb = plt.colorbar(im, ax=ax, location="bottom", pad=0.05, shrink=0.9)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(units)

msl_levs = np.linspace(-2, 2, 11)
lr_levs = np.linspace(-0.2, 0.2, 11)
sst_levs = np.linspace(-1, 1, 11)
rh_levs = np.linspace(-5, 5, 11)

for i, MONTHS in enumerate(month_sets):
    msl, lapse, relsst, rh = compute_fields(MONTHS)

    if i == 0:
        cbar = False
        title = True
    elif i < len(month_sets) - 1:
        cbar = False
        title = False
    else:
        cbar = True
        title = False

    plot_panel(axs[i, 0], msl, "MSLP",
            "RdBu_r", msl_levs, "hPa", cbar=cbar, showtitle=title)
    plot_panel(axs[i, 2], lapse, "Lapse rate",
            "BrBG", lr_levs, "K km⁻¹", cbar=cbar, showtitle=title)
    plot_panel(axs[i, 1], relsst, "Relative SST",
            "RdYlBu_r", sst_levs, "K", cbar=cbar, showtitle=title)
    plot_panel(axs[i, 3], rh,    "Mid-level RH",
            "BrBG",  rh_levs, "%", cbar=cbar, showtitle=title)

    axs[i, 0].text(-0.07, 0.55, ', '.join([month_names[j] for j in MONTHS]), va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=axs[i,0].transAxes)

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
for label, ax in zip(labels, axs.ravel()):
    ax.text(0.0, 1.0, label,
            transform=(ax.transAxes + ScaledTranslation(-10/72, +7/72, fig.dpi_scale_trans)),
            fontsize='medium', va='bottom')

out_fig = "mslp_lapse_relSST_RH_2025.png"
plt.savefig(out_fig, dpi=300)
plt.show()
