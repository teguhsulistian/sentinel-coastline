import os
from coastlines.utils import extract_contours, tide_cutoffs
from datacube import Datacube
from datacube.utils.masking import create_mask_value, valid_data_mask
from dea_tools.spatial import points_on_line
from dea_tools.datahandling import mostcommon_crs
from eo_tides import pixel_tides

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