import cdsapi

PRESS_LEVELS   = ["850","200","500"]  # added 500 hPa for RH
MONTHS = [6, 7, 8, 9, 10]
CLIM_YEARS     = list(range(1990, 2021))
TARGET_YEAR    = 2025

OUTDIR = "./output"
MSL_FILE_CLIM = os.path.join(OUTDIR, "era5_msl_sst_climatology.nc")
MSL_FILE_2025 = os.path.join(OUTDIR, "era5_msl_sst_2025.nc")
PL_FILE_CLIM  = os.path.join(OUTDIR, "era5_pl_climatology.nc")
PL_FILE_2025  = os.path.join(OUTDIR, "era5_pl_2025.nc")

# Download ERA5 single-level monthly means
def cds_download_single_monthly(years, months, target, variable):
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


# Download ERA5 pressure-level monthly means
def cds_download_pressure_monthly(years, months, target, levels=None):
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
cds_download_single_monthly(CLIM_YEARS, MONTHS, MSL_FILE_CLIM,
                            variable=["mean_sea_level_pressure","sea_surface_temperature"])
cds_download_pressure_monthly(CLIM_YEARS, MONTHS, PL_FILE_CLIM,
                              levels=PRESS_LEVELS)

# Download 2025 data
cds_download_single_monthly([TARGET_YEAR], MONTHS, MSL_FILE_2025,
                            variable=["mean_sea_level_pressure","sea_surface_temperature"])
cds_download_pressure_monthly([TARGET_YEAR], MONTHS, PL_FILE_2025,
                              levels=PRESS_LEVELS)
