import os, sys
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append("../")

from atcf_2025_loader2 import load_atcf_dataset_for_2025

a = 6.371e6 # Earth radius
g = 9.81

CENTER_DATE      = "2025-08-10"   # centre of 10-day window
buffer_days = 30
DAYS_BEFORE_FFT  = buffer_days
DAYS_AFTER_FFT   = buffer_days

EQ_LAT_RANGE     = (-20, 20)

# User-specified lags
if CENTER_DATE == "2025-10-15": # Melissa formation
    KW_LAGS_LIST  = [-5]
    MJO_LAGS_LIST = [-1]
elif CENTER_DATE == "2025-09-22": # Precursor to September cluster
    KW_LAGS_LIST  = [-4, -2]
    MJO_LAGS_LIST = [-3]
elif CENTER_DATE == "2025-08-10": # Erin
    KW_LAGS_LIST  = [-2]
    MJO_LAGS_LIST = [-2]
else:
    KW_LAGS_LIST  = [-5]
    MJO_LAGS_LIST = [-2]

# 0 = simultaneous; negative = wave envelope leads storms

CFS_DIR = "cfs_chi200_2025"

def wind_to_col(w):
    # Simple Saffir–Simpson style colouring
    if w < 34:
        return "lightgrey"   # disturbance / depression
    elif w < 64:
        return "orange"      # TS
    elif w < 96:
        return "red"         # hurricane
    else:
        return "magenta"     # major hurricane

def lat_mean(da: xr.DataArray):
    """
    Compute cosine-weighted latitudinal mean of an xarray DataArray or Dataset.
    Supports latitude dimension named 'lat' or 'latitude'.
    """
    # detect latitude dim
    for lat_name in ["lat", "latitude"]:
        if lat_name in da.dims:
            lat_dim = lat_name
            break
    else:
        raise ValueError("No latitude dimension found ('lat' or 'latitude').")

    # get latitude values and compute weights
    lat = da[lat_dim]
    weights = np.cos(np.deg2rad(lat))

    # normalize weights to avoid scaling the data
    weights = weights / weights.mean()

    # apply weighted mean along latitude dimension
    return da.weighted(weights).mean(dim=lat_dim)

# Helper download utilities
def daterange(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += dt.timedelta(days=1)


def download_cfs_vp_window(center_date, outdir=CFS_DIR,
                           days_before=40, days_after=40):

    os.makedirs(outdir, exist_ok=True)
    base = "https://noaa-cfs-pds.s3.amazonaws.com"

    center = dt.datetime.strptime(center_date, "%Y-%m-%d")
    start_date = center - dt.timedelta(days=days_before)
    end_date   = center + dt.timedelta(days=days_after)

    files = []
    for d in daterange(start_date, end_date):
        ymd = d.strftime("%Y%m%d")
        fname = f"chi200.01.{ymd}00.daily.grb2"
        url = f"{base}/cfs.{ymd}/00/time_grib_01/{fname}"
        dest = os.path.join(outdir, fname)

        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            print(f"Downloading {url}")
            status = os.system(f"wget -q '{url}' -O '{dest}'")
            if status != 0 or os.path.getsize(dest) == 0:
                print(f"WARNING: failed or empty: {url}")
                if os.path.exists(dest):
                    os.remove(dest)
                continue

        files.append(dest)

    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError("No CFS vp files downloaded.")

    return files


# Load CFS velocity potential

def load_cfs_vp(files, latmin=-40, latmax=40):

    ds_list = []
    for f in files:
        print(f"Opening {f}")

        ds = xr.open_dataset(
            f,
            engine="cfgrib",
            chunks={"latitude": 181, "longitude": 360},
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "level": 200
                }
            }
        )

        if "step" in ds.dims:
            ds = ds.isel(step=0)

        # Identify vp
        vp_var = None
        for v in ds.data_vars:
            if v.lower() in ("vp", "chi", "chi200"):
                vp_var = v
                break
        if vp_var is None:
            raise ValueError(f"No vp variable in {f}")

        latname = "latitude" if "latitude" in ds.coords else "lat"
        lonname = "longitude" if "longitude" in ds.coords else "lon"

        lat_vals = ds[latname].values
        ascending = lat_vals[0] < lat_vals[-1]
        if ascending:
            ds = ds.sel({latname: slice(latmin, latmax)})
        else:
            ds = ds.sel({latname: slice(latmax, latmin)})

        if ds[vp_var].size == 0:
            raise RuntimeError("Latitude slice produced empty array.")

        ds[vp_var] = ds[vp_var].astype("float32")
        ds_list.append(ds[[vp_var]])

    all_ds = xr.concat(ds_list, dim="time").sortby("time")

    latname = "latitude" if "latitude" in all_ds.coords else "lat"
    lonname = "longitude" if "longitude" in all_ds.coords else "lon"
    all_ds = all_ds.rename({latname: "lat", lonname: "lon"})

    return all_ds[vp_var]

def band_limited_latwise_filters(chi, lat_range=(-20, 20)):
    """
    Apply Wheeler–Kiladis style band-limited filters at each latitude.

    Returns
    -------
    mjo : xarray.DataArray (time, lat, lon)
        MJO-filtered chi200 anomalies.
    kw  : xarray.DataArray (time, lat, lon)
        Kelvin-wave-filtered chi200 anomalies.
    """

    # Remove season mean
    chi_anom = chi - chi.mean(dim="time")

    data = chi_anom.values  # (time, lat, lon)
    nt, nlat, nx = data.shape

    # Spectral axes
    freq = np.fft.fftfreq(nt, d=1.0)          # cycles per day
    k_ind = np.fft.fftfreq(nx, d=1.0) * nx    # zonal wavenumber index

    FREQ, K = np.meshgrid(freq, k_ind, indexing="ij")
    period = np.where(FREQ != 0, 1.0/np.abs(FREQ), np.inf)

    # For debug
    global kw_mask, mjo_mask

    # MJO mask, 30–96 day period, wavenumber 1-9
    mjo_mask = (
        (period >= 30) & (period <= 96) &
        (K >= 1) & (K <= 9)
        #(K >= 0) & (K <= 5)
    ).astype("float32")


    # Kelvin mask: band between dispersion curves
    def disp_curve(Hk):
        c = np.sqrt(g * Hk)
        return (c * (K / a)) * (86400.0 / (2 * np.pi))

    kw_mask = (
        (np.abs(FREQ) >= disp_curve(8.0)) &
        (np.abs(FREQ) <= disp_curve(90.0)) &
        (K >= 1) &
        (K <= 14)
    ).astype(float)

    mjo_out = np.zeros_like(data, dtype="float32")
    kw_out  = np.zeros_like(data, dtype="float32")

    for j in range(nlat):

        ts_lon = data[:, j, :]

        # FFT
        fhat = np.fft.fft2(ts_lon)

        # Apply masks
        mjo_spec = fhat * mjo_mask
        kw_spec  = fhat * kw_mask

        # IFFT
        mjo_out[:, j, :] = np.real(np.fft.ifft2(mjo_spec))
        kw_out[:,  j, :] = np.real(np.fft.ifft2(kw_spec))


    mjo_da = xr.DataArray(
        mjo_out,
        coords=chi_anom.coords,
        dims=chi_anom.dims,
        name="MJO_LATWISE"
    )

    kw_da = xr.DataArray(
        kw_out,
        coords=chi_anom.coords,
        dims=chi_anom.dims,
        name="KW_LATWISE"
    )

    return mjo_da, kw_da


def plot_wave_spectrum(chi_anom, lat_band=(-15, 15), fname="wk_spectrum.png"):
    """
    Plot k–ω spectrum of equatorial χ200 with WK masks.
    Produces a diagnostic figure to verify that the filtering masks are correct.
    """

    lat_vals = chi_anom.lat.values
    ascending = lat_vals[0] < lat_vals[-1]
    if ascending:
        chi_eqband = chi_anom.sel(lat=slice(lat_band[0], lat_band[1]))
    else:
        chi_eqband = chi_anom.sel(lat=slice(lat_band[1], lat_band[0]))

    w = np.cos(np.deg2rad(chi_eqband.lat))
    chi_eq = (chi_eqband * w).sum(dim="lat") / w.sum()

    data = chi_eq.values  # (time, lon)
    nt, nx = data.shape

    print(data)

    # 2D FFT (time, lon)
    fhat = np.fft.fft2(data)
    power = np.power(np.abs(fhat), 2)

    freq = np.fft.fftfreq(nt, d=1.0)               # 1/day
    k = np.fft.fftfreq(nx, d=1.0) * nx         # integer wavenumber indices

    FREQ, K = np.meshgrid(freq, k, indexing="ij")
    period = np.where(FREQ != 0, 1.0/FREQ, np.inf)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    pc = ax.pcolormesh(
        np.fft.fftshift(k), np.fft.fftshift(freq),
        np.fft.fftshift(np.log10(power + 1e-12)),
        shading="auto"
    )
    pc.set_clim(10, 20)
    cbar = plt.colorbar(pc, ax=ax, pad=0.02)
    cbar.set_label("log10 Power")

    # Mask overlays
    ax.contour(np.fft.fftshift(k), np.fft.fftshift(freq), np.fft.fftshift(mjo_mask), levels=[0.5],
               colors="magenta", linewidths=1.5)
    ax.contour(np.fft.fftshift(k), np.fft.fftshift(freq), np.fft.fftshift(kw_mask), levels=[0.5],
               colors="cyan", linewidths=1.5)

    ax.set_xlabel("Zonal wavenumber (k)")
    ax.set_ylabel("Frequency (cycles per day)")
    #ax.set_title("Equatorial χ200 k–ω Spectrum with MJO & Kelvin Masks")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()

# ===================================================================
# Mean helpers
# ===================================================================


def symmetric_mean(field, center_date, halfwidth=5, return_bounds=False):
    center = pd.Timestamp(center_date)
    s = center - pd.Timedelta(days=halfwidth)
    e = center + pd.Timedelta(days=halfwidth)
    sel = field.sel(time=slice(s, e))
    if sel.time.size == 0:
        raise RuntimeError("No data in window")
    if return_bounds:
        return sel.mean(dim="time"), s, e
    else:
        return sel.mean(dim="time")


# ===================================================================
# Storm segmentation & labelling
# ===================================================================

def segment_tracks(df):
    """Group tracks by the ATCF-derived storm_id."""
    segs = []
    for sid, seg in df.groupby("storm_id"):
        segs.append(seg.sort_values("datetime").copy())
    return segs

# ===================================================================
# Plotter
# ===================================================================

def plot_panel(chi10, kw_lags, kw_dates, mjo_lags, mjo_dates, lat, lon, tracks, start, end):

    lon2 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon2)
    lon2 = lon2[order]

    chi_plot = (chi10 / 1e6)[:, order]

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 4), constrained_layout=True)
    ax = plt.axes(projection=proj)

    # Atlantic domain
    lon_min, lon_max = -90, 0
    lat_min, lat_max = 0, 40

    # Background χ200
    levels = np.linspace(-9, 9, 19)
    cf = ax.contourf(
        lon2, lat, chi_plot,
        levels=levels, cmap="BrBG", extend="both",
        transform=proj
    )

    # MJO/KW colours & alphas per lag
    # Now only one lag each
    kw_colors  = ["blue"]
    mjo_colors = ["purple"]
    kw_alphas     = np.linspace(1.0, 0.3, len(KW_LAGS_LIST))
    mjo_alphas     = np.linspace(1.0, 0.3, len(MJO_LAGS_LIST))

    # Use the first lag field as a proxy for amplitude
    kw0 = kw_lags[0] / 1e6
    mjo0 = mjo_lags[0] / 1e6

    #kw_amp  = np.nanmax(np.abs(kw0)) if np.isfinite(kw0).any() else 0.0
    #mjo_amp = np.nanmax(np.abs(mjo0)) if np.isfinite(mjo0).any() else 0.0
    """
    kw_amp  = np.nanmin(kw0) if np.isfinite(kw0).any() else 0.0
    mjo_amp  = np.nanmin(mjo0) if np.isfinite(mjo0).any() else 0.0
    print(kw_amp, mjo_amp)
    kw_amp = np.abs(kw_amp)
    mjo_amp = np.abs(mjo_amp)

    # Fall back to something small if everything is zero / NaN
    if kw_amp == 0 or not np.isfinite(kw_amp):
        kw_amp = 0.1
    if mjo_amp == 0 or not np.isfinite(mjo_amp):
        mjo_amp = 0.1

    # We want *negative* anomalies (divergence aloft) – use fractions of amp
    kw_levels  = [-0.66 * kw_amp, -0.33 * kw_amp]
    mjo_levels = [-0.66 * mjo_amp, -0.33 * mjo_amp]
    """
    kw_amp = np.nanstd(kw0)
    mjo_amp = np.nanstd(mjo0)
    kw_levels = [-2*kw_amp, -1*kw_amp]
    mjo_levels = [-2*mjo_amp, -1*mjo_amp]
    print(kw_levels)
    print(mjo_levels)

    from matplotlib.lines import Line2D

    legend_elements = []
    # Plot all lag sets
    for i in range(len(KW_LAGS_LIST)):
        KW = (kw_lags[i] / 1e6)[:, order]

        # Kelvin
        cs_kw = ax.contour(
            lon2, lat, KW,
            levels=kw_levels,
            colors="blue",
            alpha=kw_alphas[i],
            linestyles=["-.", "-"],
            transform=proj
        )

        if cs_kw.levels.size > 0:
            date_str = dt.datetime.strptime(kw_dates[i], '%Y-%m-%d').strftime("%b %d")
            #ax.clabel(cs_kw, fmt=lambda v: f"KW ({date_str})", fontsize=7)
            legend_elements.append(Line2D([0], [0], color="blue", alpha=kw_alphas[i], label=f"KW ({date_str})"))
    for i in range(len(MJO_LAGS_LIST)):
        MJO= (mjo_lags[i]/ 1e6)[:, order]
        # MJO
        cs_mjo = ax.contour(
            lon2, lat, MJO,
            levels=mjo_levels,
            colors="purple",
            alpha=mjo_alphas[i],
            linestyles=["-.", "-"],
            transform=proj,
            label=f"MJO ({date_str})"
        )

        if cs_mjo.levels.size > 0:
            date_str = dt.datetime.strptime(mjo_dates[i], '%Y-%m-%d').strftime("%b %d")
            #ax.clabel(cs_mjo, fmt=lambda v: f"MJO ({date_str})", fontsize=7)
            legend_elements.append(Line2D([0], [0], color="purple", alpha=mjo_alphas[i], label=f"MJO ({date_str})"))

    ax.legend(handles=legend_elements, loc='upper left')

    # Storm tracks: separate storms, daily markers, intensity colours
    if not tracks.empty:

        lon_min, lon_max = -90, 20
        lat_min, lat_max = -5, 40


        segments = segment_tracks(tracks)

        for seg in segments:
            seg = seg.sort_values("datetime")

            # Does this storm exist in the analysis window?
            in_window = seg[
                (seg["datetime"] >= start) &
                (seg["datetime"] <= end)
            ]
            if in_window.empty:
                continue  # storm never in the window

            genesis_time = seg["datetime"].min()

            # If genesis is within the window, show the full track;
            ## otherwise, only the segment that lies inside the window.
            if (genesis_time >= start) and (genesis_time <= end):
                seg_use = seg
            else:
                continue # Only plot if genesis occurred in window

            # Clip to map domain
            seg_use = seg_use[
                (seg_use["lon_plot"] >= lon_min) & (seg_use["lon_plot"] <= lon_max) &
                (seg_use["lat"]      >= lat_min) & (seg_use["lat"]      <= lat_max)
            ].copy()

            if seg_use.empty:
                continue  # storm never actually appears in the plotted box

            # ---- Continuous track line (neutral colour) ----
            ax.plot(
                seg_use["lon_plot"], seg_use["lat"],
                color="dimgray", linewidth=1.0,
                transform=proj, zorder=4
            )

            # ---- Daily markers with intensity-based colours ----
            seg_use["date"] = seg_use["datetime"].dt.floor("D")
            daily = (
                seg_use.sort_values("datetime")
                       .groupby("date")
                       .agg({
                           "lon_plot": "first",
                           "lat": "first",
                           "wind_kts": "max"
                       })
            )

            colors = [wind_to_col(w) for w in daily["wind_kts"]]

            ax.scatter(
                daily["lon_plot"], daily["lat"],
                s=20, c=colors, edgecolor="black",
                linewidth=0.4, zorder=5, transform=proj
            )

            # ---- Date labels along the track (one per day) ----
            storm_name = None
            if "name" in seg.columns:
                ts_points = seg[seg["wind_kts"] > 40]
                names = ts_points.get("name", None)

                if ts_points.empty:
                    names = ts_points.get("name", None)
                    ts_points = seg[seg["wind_kts"] > 17]
                else:
                    first_ts = ts_points.iloc[0]
                    name = first_ts.get("name", None)
                    if isinstance(name, str) and name.strip():
                        storm_name = name.strip()

            for idx, (d, row) in enumerate(daily.iterrows()):
                txt = d.strftime("%b %d")
                nameFlag = False
                if idx == 0 and storm_name:
                    txt = f"{txt} ({storm_name})"
                    nameFlag = True

                    lat, lon = row["lat"], row["lon_plot"]

                    if storm_name == "JERRY":
                        offset = -2
                    else:
                        offset = -1

                    ax.text(
                        lon, lat+offset, txt,
                            color="red", fontsize=8, fontweight="bold",
                            ha="right", va="top",
                            transform=proj, zorder=6,
                            bbox=dict(facecolor="white", alpha=0.5,
                                  edgecolor="none", pad=0.5)
                    )

    # Map
    ax.coastlines()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Title
    ax.set_title(f"{start:%d %b} - {end:%d %b %Y}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label(r"$\chi_{200}$ ($\times$ 10$^{6}$ m$^{2}$ s$^{-1}$)")

    plt.savefig(f"tracks_kw_mjo_{start:%d-%b}_{end:%d-%Y}.png", dpi=300)
    plt.show()


# ===================================================================
# MAIN
# ===================================================================

def main():

    # Download + load CFS
    files = download_cfs_vp_window(
        CENTER_DATE, CFS_DIR, DAYS_BEFORE_FFT, DAYS_AFTER_FFT
    )
    vp = load_cfs_vp(files, latmin=-40, latmax=40)

    # Latwise wave filtering
    base_mjo_env, base_kw_env = band_limited_latwise_filters(vp)

    # Remove season mean
    chi_anom = vp - vp.mean(dim="time")

    # Get mean
    chi10, start, end = symmetric_mean(chi_anom, CENTER_DATE, return_bounds=True)

    # Lag data and get mean
    mjo_lagmeans = []
    mjo_lagdates = []
    for lag in MJO_LAGS_LIST:
        lag_center = (pd.Timestamp(CENTER_DATE) + pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        mjo_lagmeans.append(symmetric_mean(base_mjo_env, lag_center, halfwidth=1.5).values)
        mjo_lagdates.append(lag_center)

    kw_lagmeans = []
    kw_lagdates = []
    for lag in KW_LAGS_LIST:
        lag_center = (pd.Timestamp(CENTER_DATE) + pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        kw_lagmeans.append(symmetric_mean(base_kw_env, lag_center, halfwidth=1.5).values)
        kw_lagdates.append(lag_center)

    # Storm tracks
    tracks, _ = load_atcf_dataset_for_2025()
    tracks["datetime"] = pd.to_datetime(tracks["datetime"])
    tracks["lon_plot"] = np.where(
        tracks["lon"] > 180,
        tracks["lon"] - 360,
        tracks["lon"]
    )


    # 7. Plot
    plot_panel(
        chi10.values,
        kw_lagmeans,
        kw_lagdates,
        mjo_lagmeans,
        mjo_lagdates,
        vp.lat.values, vp.lon.values,
        tracks, start, end
    )


    # Diagnostic wave spectrum
    #plot_wave_spectrum(chi_anom, lat_band=(-15, 15), fname="wk_spectrum.png")

if __name__ == "__main__":
    main()

