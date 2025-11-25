import os, sys
import datetime as dt
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scipy.signal
from matplotlib.transforms import ScaledTranslation


sys.path.append("../")

from atcf_2025_loader2 import load_atcf_dataset_for_2025
from plot_waves import (
    load_cfs_vp,
    segment_tracks
)

SEASON_START = pd.Timestamp("2025-06-20")
SEASON_END   = pd.Timestamp("2025-11-05")

CFS_DIR = "cfs_chi200_2025"
RMM_FILE = "rmm.74toRealtime.txt"

LAT_BAND = (5.0, 25.0)
LON_BAND_0_360 = (300.0, 360.0, 0.0, 10.0)
LON_BAND_PM180 = (-60.0, 10.0)

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

def load_daily_ace():
    tracks, daily = load_atcf_dataset_for_2025()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.loc[(daily.index >= SEASON_START) & (daily.index <= SEASON_END)]
    if "daily_ACE" not in daily.columns:
        raise RuntimeError("Expected 'daily_ACE' in ATCF daily output.")
    return daily["daily_ACE"]

def load_rmm(path=RMM_FILE):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) < 7:
                continue
            try:
                y, m, d = map(int, p[:3])
            except ValueError:
                continue
            rows.append([y, m, d] + p[3:7])

    if not rows:
        raise RuntimeError(f"No valid RMM rows in {path}.")

    df = pd.DataFrame(
        rows,
        columns=["year", "month", "day", "RMM1", "RMM2", "phase", "amplitude"],
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").sort_index()

    for c in ["RMM1", "RMM2", "amplitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce").astype("Int64")

    return df.loc[(df.index >= SEASON_START) & (df.index <= SEASON_END)]

def daterange(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += dt.timedelta(days=1)

def download_cfs_vp_season(start, end, outdir=CFS_DIR):
    os.makedirs(outdir, exist_ok=True)
    base = "https://noaa-cfs-pds.s3.amazonaws.com"

    files = []
    for d in daterange(start.to_pydatetime(), end.to_pydatetime()):
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
    if not files:
        raise RuntimeError("No CFS vp files downloaded for the requested season.")
    return files

def band_limited_latwise_filters_tapered(
    chi200,
    lat_range=(-20, 20),
    half_window=30,
    mjo_period_range=(30.0, 96.0),   # days
    kw_period_range=(2.5, 20.0),     # days
    mjo_k_range=(1, 9),              # eastward wavenumbers for MJO
    kw_k_range=(1, 14),              # eastward wavenumbers for Kelvin
    min_bins=3                       # minimum number of freq bins in band
):
    """
    Wave filter with tapered time windows.

    For each latitude:
      • Take a time×lon window around each time index (±half_window days,
        truncated at ends).
      • Apply a Tukey taper in time.
      • Do a 2-D FFT over (time, lon) → (ω, k).
      • Apply separate (ω,k)-space masks for MJO and Kelvin.
      • Inverse FFT to get filtered segment.
      • Assign the centre of that segment as the filtered value at that time.

    Parameters
    ----------
    chi200 : xr.DataArray(time, lat, lon)
        χ200 (velocity potential) anomalies or fields.
    lat_range : tuple
        Latitude range for filtering.
    half_window : int
        Half-size of time window in days (default 30 → 61-day window).
    mjo_period_range : (min_days, max_days)
        MJO period band in days (default 30–96).
    kw_period_range : (min_days, max_days)
        Kelvin period band in days (default 2.5–20).
    mjo_k_range : (kmin, kmax)
        Zonal wavenumber range for MJO (eastward).
    kw_k_range : (kmin, kmax)
        Zonal wavenumber range for Kelvin (eastward).
    min_bins : int
        Minimum number of discrete frequency bins to include in each band.

    Returns
    -------
    mjo_da, kw_da : xr.DataArray
        Filtered χ200 fields for MJO and Kelvin, same shape as chi200.sel(lat=lat_range).
    """

    # --- Latitude subset with order robustness ---
    lat0, lat1 = lat_range
    if chi200.lat.values[0] < chi200.lat.values[-1]:
        # ascending
        chi = chi200.sel(lat=slice(lat0, lat1))
    else:
        # descending
        chi = chi200.sel(lat=slice(lat1, lat0))

    time = chi.time.values
    nt   = chi.sizes["time"]
    nlat = chi.sizes["lat"]
    nlon = chi.sizes["lon"]

    mjo_filtered = np.zeros_like(chi.values, dtype=float)
    kw_filtered  = np.zeros_like(chi.values, dtype=float)

    mjo_fmin_base = 1.0 / mjo_period_range[1]
    mjo_fmax_base = 1.0 / mjo_period_range[0]
    kw_fmin_base  = 1.0 / kw_period_range[1]
    kw_fmax_base  = 1.0 / kw_period_range[0]

    for j in range(nlat):
        # 2D field for this latitude: (time, lon)
        lat_series = chi.values[:, j, :]  # shape (nt, nlon)

        for i in range(nt):
            # Select time window
            i0 = max(0, i - half_window)
            i1 = min(nt - 1, i + half_window)
            segment = lat_series[i0:i1+1, :]  # (T, nlon)
            T = segment.shape[0]

            if T < 5:
                # too short to filter; just use original value
                mjo_filtered[i, j, :] = lat_series[i, :]
                kw_filtered[i, j, :]  = lat_series[i, :]
                continue

            # Tukey window in time
            t_win = scipy.signal.tukey(T, alpha=0.5)
            seg_taper = segment * t_win[:, None]

            # 2D FFT (time, lon)
            F = np.fft.fft2(seg_taper, axes=(0, 1))

            # freq in cycles per day
            freqs = np.fft.fftfreq(T, d=1.0)
            # integer-ish zonal wavenumbers
            k = np.fft.fftfreq(nlon, d=1.0 / nlon)  # gives 0,1,2,...,-1 etc

            freq2d, k2d = np.meshgrid(freqs, k, indexing="ij")
            freq_abs = np.abs(freq2d)

            df = np.abs(freqs[1] - freqs[0]) if T > 1 else 1.0

            # Adaptive MJO band
            mjo_fmin = mjo_fmin_base
            mjo_fmax = mjo_fmax_base
            if (mjo_fmax - mjo_fmin) < min_bins * df:
                extra = 0.5 * (min_bins * df - (mjo_fmax - mjo_fmin))
                mjo_fmin = max(0.0, mjo_fmin - extra)
                mjo_fmax = mjo_fmax + extra

            mjo_kmin, mjo_kmax = mjo_k_range

            mjo_mask = (
                (freq_abs >= mjo_fmin) & (freq_abs <= mjo_fmax) &
                (k2d >= mjo_kmin) & (k2d <= mjo_kmax)
            )

            # Adaptive Kelvin band
            kw_fmin = kw_fmin_base
            kw_fmax = kw_fmax_base
            if (kw_fmax - kw_fmin) < min_bins * df:
                extra = 0.5 * (min_bins * df - (kw_fmax - kw_fmin))
                kw_fmin = max(0.0, kw_fmin - extra)
                kw_fmax = kw_fmax + extra

            kw_kmin, kw_kmax = kw_k_range

            kw_mask = (
                (freq_abs >= kw_fmin) & (freq_abs <= kw_fmax) &
                (k2d >= kw_kmin) & (k2d <= kw_kmax)
            )

            # Apply masks
            mjo_fft = np.zeros_like(F)
            kw_fft  = np.zeros_like(F)

            mjo_fft[mjo_mask] = F[mjo_mask]
            kw_fft[kw_mask]   = F[kw_mask]

            mjo_seg = np.fft.ifft2(mjo_fft, axes=(0, 1)).real
            kw_seg  = np.fft.ifft2(kw_fft,  axes=(0, 1)).real

            # Take the centre of the window as filtered value at time i
            center_idx = T // 2
            if center_idx >= T:
                center_idx = T - 1

            mjo_filtered[i, j, :] = mjo_seg[center_idx, :]
            kw_filtered[i, j, :]  = kw_seg[center_idx, :]

    mjo_da = xr.DataArray(
        mjo_filtered,
        coords=chi.coords,
        dims=chi.dims,
        name="mjo"
    )
    kw_da = xr.DataArray(
        kw_filtered,
        coords=chi.coords,
        dims=chi.dims,
        name="kw"
    )
    return mjo_da, kw_da

def compute_wave_indices(start, end):

    buffer_days = 30
    start_ext = start - pd.Timedelta(days=buffer_days)
    end_ext   = end   + pd.Timedelta(days=buffer_days)

    files = download_cfs_vp_season(start_ext, end_ext, outdir=CFS_DIR)
    print("files downloaded:", len(files))

    chi200 = load_cfs_vp(files, latmin=-40, latmax=40)

    # Compute anomaly
    chi200 = chi200 - chi200.mean(dim="time")

    # Wave filtering
    mjo_filt, kw_filt = band_limited_latwise_filters_tapered(chi200, mjo_period_range=(30.0, 96.0),
        kw_period_range=(2.0, 20.0),
        mjo_k_range=(1, 9),
        kw_k_range=(1,14),
        min_bins=2,
        lat_range=(-20, 20), half_window=30)

    if mjo_filt.lat.values[0] > mjo_filt.lat.values[-1]:
        mjo_filt = mjo_filt.sortby("lat")
        kw_filt  = kw_filt.sortby("lat")

    # 3-day running mean in time BEFORE any spatial stats
    mjo_smooth = mjo_filt.rolling(time=3, center=True).mean()
    kw_smooth  = kw_filt.rolling(time=3, center=True).mean()

    # keep edges from the original if you don't want NaNs at ends
    mjo_smooth = mjo_smooth.fillna(mjo_filt)
    kw_smooth  = kw_smooth.fillna(kw_filt)

    # Select MDR
    lat0, lat1 = LAT_BAND
    mjo_lat = mjo_smooth.sel(lat=slice(lat0, lat1))
    kw_lat  = kw_smooth.sel(lat=slice(lat0, lat1))

    # Handle longitude boundaries for both grids
    lons = mjo_lat.lon.values
    if lons.min() >= 0:
        w1,e1,w2,e2 = LON_BAND_0_360
        mjo_box = xr.concat(
            [mjo_lat.sel(lon=slice(w1,e1)),
             mjo_lat.sel(lon=slice(w2,e2))], dim="lon")
        kw_box = xr.concat(
            [kw_lat.sel(lon=slice(w1,e1)),
             kw_lat.sel(lon=slice(w2,e2))], dim="lon")
    else:
        w,e = LON_BAND_PM180
        mjo_box = mjo_lat.sel(lon=slice(w,e))
        kw_box  = kw_lat .sel(lon=slice(w,e))

    # MDR mean & minimum anomalies
    mjo_mean = lat_mean(mjo_box.mean(dim="lon"))
    kw_mean  = lat_mean(kw_box .mean(dim="lon"))

    mjo_min  = mjo_box.min(dim=("lat","lon"))
    kw_min   = kw_box .min(dim=("lat","lon"))

    # Convert to Series and restrict to season
    mjo_mean = pd.Series(mjo_mean.values, index=pd.to_datetime(mjo_mean.time.values)).loc[start:end]
    kw_mean  = pd.Series(kw_mean.values,  index=pd.to_datetime(kw_mean.time.values)).loc[start:end]

    mjo_min  = pd.Series(mjo_min.values, index=pd.to_datetime(mjo_min.time.values)).loc[start:end]
    kw_min   = pd.Series(kw_min.values,  index=pd.to_datetime(kw_min.time.values)).loc[start:end]

    mjo_sigma = mjo_smooth.sel(lat=slice(-20,20)).std(dim=("lat","lon"))
    kw_sigma  = kw_smooth.sel(lat=slice(-20,20)).std(dim=("lat","lon"))

    mjo_sigma = pd.Series(mjo_sigma.values, index=pd.to_datetime(mjo_sigma.time.values)).loc[start:end]
    kw_sigma  = pd.Series(kw_sigma.values,  index=pd.to_datetime(kw_sigma.time.values)).loc[start:end]

    mjo_fav = (mjo_min <= -mjo_sigma)
    kw_fav  = (kw_min <= -kw_sigma)

    chi_lat = chi200.sel(lat=slice(lat1, lat0))
    if chi_lat.lon.min() >= 0:
        chi_box = xr.concat(
            [chi_lat.sel(lon=slice(w1,e1)),
             chi_lat.sel(lon=slice(w2,e2))],
            dim="lon"
        )
    else:
        chi_box = chi_lat.sel(lon=slice(w,e))

    chi_mean = lat_mean(chi_box.mean(dim="lon"))
    chi_mean = pd.Series(
        chi_mean.values,
        index=pd.to_datetime(chi_mean.time.values)
    ).loc[start:end]

    return mjo_mean, kw_mean, mjo_fav, kw_fav, chi_mean, (start_ext, end_ext)

def shade_regions(ax, dates, mask, color, label):
    in_reg = False
    start = None
    for d, flag in zip(dates, mask):
        if flag and not in_reg:
            in_reg = True
            start = d
        elif not flag and in_reg:
            ax.axvspan(start, d, color=color, alpha=0.25, linewidth=0, label=label)
            in_reg = False
    if in_reg:
        ax.axvspan(start, dates[-1], color=color, alpha=0.25, linewidth=0, label=label)

def wind_to_col_label(max_wind):
    if max_wind < 34:
        return 'grey', 'TD'
    elif max_wind < 64:
        return 'orange', 'TS'
    elif max_wind < 96:
        return 'red', 'H'
    else:
        return 'magenta', 'MH'

def create_plot(tracks, ace, mjo_idx, kw_idx, mjo_fav, kw_fav, rmm,
                shear=None, chi_mean=None, dates=(None, None)):

    if dates[0] and dates[1]:
        full_idx = pd.date_range(dates[0], dates[1], freq="D")
    else:
        full_idx = pd.date_range(SEASON_START, SEASON_END, freq="D")

    df = pd.DataFrame(index=full_idx)
    df["MJO_idx"] = mjo_idx.reindex(full_idx)
    df["KW_idx"]  = kw_idx.reindex(full_idx)
    df["MJO_fav"] = mjo_fav.reindex(full_idx, fill_value=False)
    df["KW_fav"]  = kw_fav.reindex(full_idx, fill_value=False)
    df["chi_mean"] = chi_mean.reindex(full_idx) if chi_mean is not None else np.nan


    fig, axs = plt.subplots(3, 1, figsize=(8, 5), constrained_layout=True)

    # Daily ACE
    ax = axs[0]
    shade_regions(ax, df.index, df["MJO_fav"], "purple", "MJO fav")
    shade_regions(ax, df.index, df["KW_fav"],  "blue",   "KW fav")

    ax.plot(ace.index, ace.values, "k-", lw=1.3)

    ax.set_ylabel("Daily ACE")
    ax.set_ylim(0, 1250)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_yticklabels([0, 200, 400, 600, 800])

    ax.set_xlim(dates[0], dates[1])
    ax.set_xticklabels([])

    bar_ax = inset_axes(ax, width="100%", height="28%",
                        loc='upper left', bbox_to_anchor=(0, 0, 1, 1),
                        bbox_transform=ax.transAxes, borderpad=0)

    bar_ax.set_xlim(dates[0], dates[1])
    bar_ax.set_xticklabels([])

    bar_ax.set_ylim(0, 1)
    bar_ax.set_yticks([])

    segments = segment_tracks(tracks)
    y0 = 0.2
    y = y0
    dy = 0.3

    for i, seg in enumerate(segments):
        seg = seg.sort_values("datetime")
        time_pts = seg["datetime"].values
        wind = seg["wind_kts"].values

        offset = pd.Timedelta(days=1)

        name = seg["name"].unique()[-1]
        if name in ['ERIN', 'GABRIELLE', 'HUMBERTO', 'IMELDA']:
            bar_ax.text(seg["datetime"].values[0]-offset, y, name, va='center_baseline', ha='right',
                    fontsize=8, fontweight='bold')
        elif name in ['MELISSA']:
            bar_ax.text(seg["datetime"].values[0], y+0.1, name, va='bottom', ha='left',
                    fontsize=8, fontweight='bold')

        for j in range(len(time_pts)-1):
            plt.plot([time_pts[j], time_pts[j+1]], [y, y],
                    color=wind_to_col_label(wind[j])[0], lw=5)

        y += dy
        if y > 0.9:
            y = y0

    legend_handles = [Patch(facecolor=wind_to_col_label(w)[0], edgecolor=None, label=wind_to_col_label(w)[1]) for w in [10, 50, 80, 100] ]

    # Wave indices
    ax = axs[1]
    ax.axhline(0,color='k')
    ax.plot(df.index, df["MJO_idx"]/1e6, color="purple", label=r"MJO ($\times$ 10$^6$)")
    ax.plot(df.index, df["KW_idx"]/1e6, color="blue", label=r"KW ($\times$ 10$^6$)")
    ax.plot(df.index, df["chi_mean"]/3e6, color="green", label=r"Full $\chi_{200}$ ($\times 3\cdot10^{6}$)")
    ax.legend(loc='upper right', ncol=3)

    ax.set_xlim(dates[0], dates[1])
    ax.set_xticklabels([])

    ax.set_ylim(-3, 4)
    ax.set_ylabel("MDR mean")

    # MJO phase and amplitude
    ax=axs[2]
    ax.plot(rmm.index, rmm["amplitude"],"k-")
    ax.set_ylim(0, 2.5)

    ax2=ax.twinx()
    ax2.plot(rmm.index, rmm["phase"],"r.",ms=3)

    ax.set_ylabel("MJO amplitude")
    ax.set_xlim(dates[0], dates[1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    ax.set_xlabel("Date")

    ax2.set_xlim(dates[0], dates[1])
    ax2.set_ylabel("MJO phase", color="red")
    ax2.set_yticklabels(ax2.get_yticklabels(), color='r')
    ax2.set_ylim(0.5, 8.5)

    # panel labels
    labels = ["(a)", "(b)", "(c)"]
    for label, ax in zip(labels, axs):
        ax.text(0.0, 1.0, label,
                transform=(ax.transAxes + ScaledTranslation(-10/72, +7/72, fig.dpi_scale_trans)),
                fontsize='medium', va='bottom')

    plt.savefig("ACE_MJO_KW_timeseries.png", dpi=300)
    plt.show()

def main():
    tracks, _ = load_atcf_dataset_for_2025()
    tracks["datetime"] = pd.to_datetime(tracks["datetime"])
    tracks["lon_plot"] = np.where(
        tracks["lon"] > 180,
        tracks["lon"] - 360,
        tracks["lon"]
    )

    ace = load_daily_ace()
    rmm = load_rmm()
    print("RMM coverage:", rmm.index.min(), rmm.index.max())

    start = max(SEASON_START, ace.index.min())
    end   = min(SEASON_END,   ace.index.max())
    dates = (start, end)

    print(f"Season used: {start.date()} to {end.date()}")

    mjo_idx, kw_idx, mjo_fav, kw_fav, chi_mean, _ = compute_wave_indices(start, end)

    create_plot(tracks, ace, mjo_idx, kw_idx, mjo_fav, kw_fav, rmm, chi_mean=chi_mean, dates=dates)

if __name__ == "__main__":
    main()

