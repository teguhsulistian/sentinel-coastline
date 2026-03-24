import os
import numpy as np
import xarray as xr
from coastlines.utils import extract_contours, tide_cutoffs
from datacube import Datacube
from datacube.utils.masking import create_mask_value, valid_data_mask
from dea_tools.spatial import points_on_line
from dea_tools.datahandling import mostcommon_crs
from eo_tides import pixel_tides
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation

#Load and Masking Sentinel Datasheet
def load_mask(product,start_date,end_date,lon_range,lat_range,max_cloud_cover):
    dc = Datacube(app="coastlines")
    
    # Find and load Landsat datasets
    sentinel2_dataset = dc.find_datasets(
        product=[product],
        time=(start_date, end_date),
        longitude=lon_range,
        latitude=lat_range,
        cloud_cover=(0, max_cloud_cover),
    )

    # Identify the most common projection system in the input query
    query = {
        'y': lat_range,
        'x': lon_range,
        'time': (start_date, end_date),
        'resolution': (-20, 20),
    }
    output_crs = mostcommon_crs(dc=dc, product="s2_l2a", query=query)
    
    
    print(f"Found {len(sentinel2_dataset)} Sentinel datasets")
    
    data = dc.load(
        datasets=sentinel2_dataset,
        longitude=lon_range,
        latitude=lat_range,
        resolution=20,
        output_crs=output_crs,
        measurements=["red", "green", "blue", "nir08", "swir16", "swir22", "scl"],
        group_by="solar_day",
        dask_chunks={"time": 1, "x": 3200, "y": 3200},
        resampling={
            "*": "cubic",
            "scl": "nearest",
        },
        driver="rio",
    )

    # Mask Sentinel-2 data
    # 3 is cloud shadow, 8 is medium probability cloud, 9 is high probability cloud
    cloud_mask = data.scl.isin([3, 8, 9])
    valid_data = valid_data_mask(data)
    mask = cloud_mask | ~valid_data
    
    s2_ds_masked = data.where(~mask).drop_vars("scl")
    
    # Scale Sentinel-2 data so that values are between 0 and 1
    masked = (s2_ds_masked * 0.0001).clip(
        0, 1
    )  # Scale Sentinel-2 data to match Landsat scale
  
    return masked

#Tide Masking
def tide_mask(masked,tide_data_location,model):
    # Add tide height to the data
    tides = pixel_tides(
        masked,
        resample=True,
        directory=tide_data_location,
        model=model,
        dask_compute=True,
    )
    
    # Determine tide cutoff
    tide_cutoff_min, tide_cutoff_max = tide_cutoffs(masked, tides, tide_centre=0.0)
    
    tide_bool = (tides >= tide_cutoff_min) & (tides <= tide_cutoff_max)
    data_filtered = masked.sel(time=tide_bool.sum(dim=["x", "y"]) > 0)
    
    # Apply mask, and load in corresponding tide masked data
    data_tide_masked = data_filtered.where(tide_bool)
 
    return data_tide_masked

#Water Index Processing
def water_index(wi,data_tide_masked):
    if wi == "combined":
        # # # Create Combined index (the mean of MNDWI and NDWI)
        index_name = wi
        data_tide_masked["mndwi"] = (data_tide_masked.green - data_tide_masked.swir16) / (data_tide_masked.green + data_tide_masked.swir16)
        data_tide_masked["ndwi"] = (data_tide_masked.green - data_tide_masked.nir08) / (data_tide_masked.green + data_tide_masked.nir08)
        data_tide_masked[wi] = (data_tide_masked.mndwi + data_tide_masked.ndwi) / 2
    else:
        # # Create MNDWI_NIR index
        index_name = wi
        scaled_green = (
            data_tide_masked.green
            + (data_tide_masked.nir08.max(dim="time") - data_tide_masked.nir08)
        ) / 2
        scaled_swir1 = (data_tide_masked.swir16 + data_tide_masked.nir08) / 2
        data_tide_masked[wi] = (scaled_green - scaled_swir1) / (
            scaled_green + scaled_swir1
        )

    #Group data by year
    grouped_by_year = (
        data_tide_masked[index_name]
        .groupby("time.year")
        .median()
        .to_dataset(name=index_name)
    )
    
    return grouped_by_year

#Masking
def inland_water_mask(
    da,
    threshold=0.1,
    min_area=2000,
    dilation_iter=2,
    fill_value=-1,
    wi="combined"
):
    """
    Extract coastal water mask from multi-temporal water index data and
    prepare a contour-ready dataset with inland water (e.g., fish ponds) removed.

    -------------------------------------------------------------------------
    🌍 ALGORITHM OVERVIEW
    -------------------------------------------------------------------------
    This function separates ocean water from inland water (e.g., aquaculture ponds)
    using a combination of spectral, spatial, and connectivity filtering:

    1. Spectral filtering (threshold)
       - Convert water index into binary water mask
       - Water = water index > threshold

    2. Spatial filtering (min_area)
       - Label connected water regions
       - Remove small regions (typically fish ponds, small lakes)

    3. Connectivity filtering (border connection)
       - Keep only water connected to image boundary
       - Assumption: ocean touches image edge, inland water does not

    4. Morphological smoothing (dilation)
       - Smooth coastline and reduce pixel-level noise

    5. Mask application
       - Apply coastal mask to original water index
       - Inland water is removed (set to fill_value)

    -------------------------------------------------------------------------
    📌 PARAMETER INTERPRETATION
    -------------------------------------------------------------------------
    threshold:
        Controls sensitivity of water detection.
        Lower → more water detected (may include noise/fish ponds)
        Higher → stricter water detection (may miss shallow water)

    min_area:
        Controls removal of small water bodies.
        Based on pixel count (Landsat: 1 px ≈ 900 m²).
        Increase → more aggressive removal of fish ponds.

    dilation_iter:
        Number of smoothing iterations.
        Higher → smoother coastline but may expand edges.

    fill_value:
        Value used to replace masked pixels.
        Required for stable contour extraction (avoid NaN gaps).

    wi:
        Output variable name for water index (used in downstream processing).

    -------------------------------------------------------------------------
    📤 OUTPUTS
    -------------------------------------------------------------------------
    coastal_da : xarray.DataArray
        Binary coastal mask (1 = ocean, 0 = others)

    masked_ds : xarray.Dataset
        Masked water index dataset (inland water removed),
        ready for contour extraction (extract_contours)

    -------------------------------------------------------------------------
    🎯 USE CASE
    -------------------------------------------------------------------------
    - Coastline extraction from Landsat/Sentinel
    - Removing aquaculture ponds
    - Multi-year shoreline analysis
    - Pre-processing for sub-pixel contouring
    """
    
    if not isinstance(da, xr.DataArray):
        raise TypeError("❌ Input must be xarray.DataArray")

    if da.ndim != 3:
        raise ValueError("❌ Expected 3D DataArray (year, y, x)")

    coastal_masks = np.zeros_like(da.values, dtype=np.uint8)

    for i, yr in enumerate(da["year"].values):

        mndwi = da.sel(year=yr).values

        water_mask = mndwi > threshold
        labeled = label(water_mask)

        clean_mask = np.zeros_like(water_mask, dtype=bool)
        for region in regionprops(labeled):
            if region.area >= min_area:
                clean_mask[labeled == region.label] = True

        labeled_clean = label(clean_mask)

        border_mask = np.zeros_like(clean_mask)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        coastal_labels = np.unique(labeled_clean[border_mask & clean_mask])
        coastal_mask = np.isin(labeled_clean, coastal_labels)

        coastal_mask = binary_dilation(coastal_mask, iterations=dilation_iter)

        coastal_masks[i] = coastal_mask.astype(np.uint8)

    # --- coastal mask (DataArray)
    coastal_da = xr.DataArray(
        coastal_masks,
        coords=da.coords,
        dims=da.dims,
        name="coastal_mask",
        attrs={
            "description": "Coastal water mask (fish ponds removed)",
            "threshold": threshold,
            "min_area": min_area
        }
    )

    # --- masked dataset (for contour)
    masked_da = (
        da.where(coastal_da == 1)
          .fillna(fill_value)
          .rename(wi)
    )

    masked_ds = xr.Dataset({wi: masked_da})

    return coastal_da, masked_ds