import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, minimum_filter, label
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.transforms import ScaledTranslation

INPUT_FILE = "era5_u_v_z_200_850hPa_monthly_2025_JJAS.nc"

def segment_to_line(seg_mask, lats, lons):
    """
    Return contour-line vertices for one mask segment
    """
    fig = Figure()
    ax = fig.add_subplot(111)

    # Generate contour on the off-screen axes
    cs = ax.contour(lons, lats, seg_mask.astype(int), levels=[0.5])

    paths = cs.collections[0].get_paths()
    lines = []

    for p in paths:
        v = p.vertices
        xs, ys = v[:, 0], v[:, 1]
        lines.append((xs, ys))

    return lines

def extract_segments(mask, lats, lons, connectivity=1):
    """
    Extracts contiguous segments of 1s from a mask array.

    Parameters
    ----------
    mask : 2D array of 0/1 or NaN
        The segmentation mask.
    lats, lons : 2D arrays
        Latitude & longitude arrays aligned with mask.
    connectivity : int (1 or 2)
        1 = 4-connectivity, 2 = 8-connectivity.

    Returns
    -------
    segments : list of dict
        Each dict contains 'lat', 'lon', 'mask' arrays for that segment.
    """
    # Convert NaN to 0
    binmask = np.nan_to_num(mask).astype(bool)

    # Label connected components
    structure = np.ones((3,3)) if connectivity == 2 else None
    labeled, n_segments = label(binmask, structure=structure)

    segments = []
    for seg_id in range(1, n_segments + 1):
        seg_mask = labeled == seg_id
        segments.append({
            "lat": lats[seg_mask],
            "lon": lons[seg_mask],
            "mask": seg_mask,
        })

    return segments

def smooth_da(da, sigma=1.0):
    arr = gaussian_filter(da.values, sigma=sigma)
    return xr.DataArray(arr, coords=da.coords, dims=da.dims)

def compute_relative_vorticity(u_da, v_da):
    u2 = u_da.squeeze()
    v2 = v_da.squeeze()
    if u2.ndim != 2 or v2.ndim != 2:
        raise ValueError(f"compute_relative_vorticity expects 2D arrays, got {u2.shape} and {v2.shape}")
    lat_rad = np.deg2rad(u2.latitude.values)
    lon_rad = np.deg2rad(u2.longitude.values)
    a = 6.371e6
    dudlat = np.gradient(u2.values, axis=0) / np.gradient(lat_rad)[:, None]
    vcos = v2.values * np.cos(lat_rad)[:, None]
    dvcos_dlon = np.gradient(vcos, axis=1) / np.gradient(lon_rad)[None, :]
    zeta = (1/(a*np.cos(lat_rad)[:,None]))*dvcos_dlon - (1/a)*dudlat
    return xr.DataArray(zeta, coords=u2.coords, dims=u2.dims)

def find_trough_minima(z_sm, min_distance=5):
    data = z_sm.values
    footprint = np.ones((min_distance, min_distance))
    local_min = minimum_filter(data, footprint=footprint, mode="nearest") == data
    thresh = np.percentile(data[~np.isnan(data)], 40)
    mask = local_min & (data < thresh)
    return xr.DataArray(mask, coords=z_sm.coords, dims=z_sm.dims)

ds = xr.open_dataset(INPUT_FILE)
sigma, vort_thresh = 1.0, 1e-5

proj = ccrs.PlateCarree()
fig, axs = plt.subplots(2, 2, figsize=(8,4), subplot_kw={"projection": proj}, constrained_layout=True)
axs = axs.flatten()

n_plot = 4
month_labels=["June", "July", "August", "September"]

for i in range(n_plot):
    vt = ds["valid_time"].isel({"valid_time": i}).values
    ax = axs[i]

    # extract 2D slices explicitly by index (guarantees 2D arrays)
    u200 = ds["u"].isel({"valid_time": i}).sel(pressure_level=200.0).squeeze()
    v200 = ds["v"].isel({"valid_time": i}).sel(pressure_level=200.0).squeeze()
    u850 = ds["u"].isel({"valid_time": i}).sel(pressure_level=850.0).squeeze()
    v850 = ds["v"].isel({"valid_time": i}).sel(pressure_level=850.0).squeeze()
    z200 = ds["z"].isel({"valid_time": i}).sel(pressure_level=200.0).squeeze()
    # compute shear and geopotential height (m)
    shear = np.sqrt((u200 - u850)**2 + (v200 - v850)**2)
    z200_m = z200 / 9.80665

    # smooth (keep coords)
    u_s = smooth_da(u200, sigma)
    v_s = smooth_da(v200, sigma)
    shear_s = smooth_da(shear, sigma)
    z_s = smooth_da(z200_m, 2*sigma)

    # vorticity (2D) and smoothed
    zeta = compute_relative_vorticity(u200, v200)
    zeta_s = smooth_da(zeta, sigma)

    # trough mask
    trough_mask = find_trough_minima(z_s, min_distance=5)
    trough_mask = trough_mask & (zeta_s > vort_thresh)

    # meshgrid (use coordinate values)
    Lon = np.meshgrid(u_s.longitude.values, u_s.latitude.values)[0]
    Lat = np.meshgrid(u_s.longitude.values, u_s.latitude.values)[1]  # shapes (lat, lon) and (lat, lon)

    # --- map base ---
    ax.set_extent([-100, 0, 2, 50], crs=proj)

    # tick positions
    lat_ticks = np.arange(5, 55, 10)
    lon_ticks = np.arange(-100, 10, 20)

    ax.set_yticks(lat_ticks, crs=proj)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    if i % 2 != 0:
        ax.set_yticklabels([])

    ax.set_xticks(lon_ticks, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    if i // 2 != 1:
        ax.set_xticklabels([])

    # Make sure Matplotlib draws the ticks
    ax.tick_params(length=4, width=0.8)

    ax.coastlines("50m", color='k', linewidth=0.7)

    # --- filled shear ---
    cf = ax.contourf(Lon, Lat, shear_s.values,
                     levels=np.arange(0, 40, 3), cmap="Reds",
                     transform=proj, extend="max")

    # --- geopotential contours (coarser & lighter) ---
    gph_step = 50
    levels = np.arange(np.nanmin(z_s.values)//60*60,
                       np.nanmax(z_s.values)//60*60 + gph_step, gph_step)
    ax.contour(Lon, Lat, z_s.values, levels=levels,
               colors="gray", linewidths=1.5, linestyles="--")

    # --- streamlines ---
    ax.streamplot(Lon, Lat, u_s.values, v_s.values,
                  color="lightgreen", linewidth=1.0, density=0.5, arrowsize=0.8)

    # --- Compute smoothed height field (already done earlier) ---
    Z = z_s.values  # geopotential height (m)
    lat = z_s.latitude.values
    lon = z_s.longitude.values

    # --- Compute meridional gradient dZ/dφ ---
    lat_rad = np.deg2rad(lat)
    dZ_dlat = np.gradient(Z, lat_rad, axis=0)

    # --- Identify trough axes as zero crossings ---
    # mask where sign of dZ/dlat changes with latitude
    sign_change = np.sign(dZ_dlat[1:, :] * dZ_dlat[:-1, :]) < 0

    # Assign latitudes midway between the sign changes
    lat_mid = 0.5 * (lat[1:] + lat[:-1])

    # Create DataArrays for plotting
    trough_lat = np.full_like(Z, np.nan)
    for j in range(len(lon)):
        inds = np.where(sign_change[:, j])[0]
        if len(inds) > 0:
            for idx in inds:
                if dZ_dlat[idx, j] > 0:   # trough (height decreases then increases)
                    trough_lat[idx, j] = lat_mid[idx]

    # Convert to DataArrays for convenience
    trough_mask = xr.DataArray(~np.isnan(trough_lat), coords=z_s.coords, dims=z_s.dims)

    segments = extract_segments(np.isnan(trough_lat).astype(int), Lat, Lon, connectivity=2)
    for seg in segments:
        lines = segment_to_line(seg["mask"], Lat, Lon)
        for xs, ys in lines:
            ax.plot(xs, ys, color='b', linewidth=2, transform=proj)

    ax.set_title(month_labels[i])

labels = ["(a)", "(b)", "(c)", "(d)"]
for label, ax in zip(labels, axs):
    ax.text(0.0, 1.0, label,
            transform=(ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='medium', va='bottom')


# shared colorbar
cb = fig.colorbar(cf, ax=[axs[1],axs[3]], location='right', shrink=0.7)
cb.set_label(r"850–200 hPa vertical wind shear (m s$^{-1}$)")

OUT_PNG = "ERA5_JJAS2025_TUTTs_composite.png"
plt.savefig(OUT_PNG, dpi=300)
plt.show()
