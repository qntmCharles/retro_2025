#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# SETTINGS

lat_min, lat_max = 10, 20
lon_min, lon_max = -85, -20

LOCAL_ZARR = "oisst_mdr.zarr"

# Kerchunk reference to the global OISST dataset on S3
REF_URL = (
    "https://ncsa.osn.xsede.org/Pangeo/pangeo-forge/"
    "pangeo-forge/aws-noaa-oisst-feedstock/"
    "aws-noaa-oisst-avhrr-only.zarr/reference.json"
)

# Load data

remote = xr.open_dataset(
    "reference://",
    engine="zarr",
    backend_kwargs={
        "consolidated": False,
        "storage_options": {
            "fo": REF_URL,
            "remote_protocol": "s3",
            "remote_options": {"anon": True},
        },
    },
    chunks={},
)

# Fix longitude to -180–180
remote = remote.assign_coords(lon=((remote.lon + 180) % 360) - 180)

# Restrict to 1991 onwards
remote = remote.sel(time=slice("1991-01-01", None))

mdr = remote.where(
    (remote.lat >= lat_min) & (remote.lat <= lat_max) &
    (remote.lon >= lon_min) & (remote.lon <= lon_max),
    drop=True
)

mdr.to_zarr(LOCAL_ZARR, mode="w")

# Re-open locally to ensure clean state
ds = xr.open_zarr(LOCAL_ZARR)

print(ds)


# Cos-weighted MDR mean

weights = np.cos(np.deg2rad(ds.lat))
weights = weights / weights.mean()

sst_weighted = (ds["sst"] * weights).mean(dim=["lat", "lon", "zlev"])
sst_weighted = sst_weighted.rename("sst")   # needed for to_dataframe()

df = sst_weighted.to_dataframe().reset_index()
df["year"] = df["time"].dt.year
df["doy"]  = df["time"].dt.dayofyear

# Save data

OUT_CSV = "oisst_mdr_daily.csv"
df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
