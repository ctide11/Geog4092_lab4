import glob
import rasterio
import numpy as num
import os
from rasterio.plot import show
import rasterio, os
from rasterio.warp import calculate_default_transform, reproject, Resampling

# define variable

filepath = "C:\data"
raster_file = glob.glob(filepath + '/*tif')

# open and prep raster files
urban_areas = rasterio.open(os.path.join(filepath, 'urban_areas.tif'))
water_bodies = rasterio.open(os.path.join(filepath, 'water_bodies.tif'))
protected_areas = rasterio.open(os.path.join(filepath, 'protected_areas.tif'))
slope = rasterio.open(os.path.join(filepath, 'slope.tif'))
wind_speed = rasterio.open(os.path.join(filepath, 'ws80m.tif'))


urbanA = urban_areas.read(1)
waterA = water_bodies.read(1)
protectedA = protected_areas.read(1)
slopeA = slope.read(1)
windA = wind_speed.read(1)

slopeA = num.where(slopeA < 0, 0, slopeA)
windA = num.where(windA < 0, 0, windA)

window_rows = 11
window_colms = 9

mask = num.ones((window_rows, window_colms))

def mean_filter(ma, mask):
    pct_array = num.zeros(ma.shape)
    wind_area = float(mask.sum())
    row_dim = mask.shape[0]//2
    col_dim = mask.shape[1]//2
    for row in range(row_dim,ma.shape[0]-row_dim):
        for col in range(col_dim,ma.shape[1]-col_dim):
            win = ma[row-row_dim:row+row_dim+1,col-col_dim:col+col_dim+1]
            pct_array[row,col] = win.sum()
    return pct_array/wind_area


    wind_loca = mean_filter(windA, mask)
wind_loca = num.where(wind_loca > 8.5, 1, 0)

slope_loca = mean_filter(slopeA, mask)
slope_loca = num.where(slope_loca < 15, 1, 0)

water_loca = mean_filter(waterA, mask)
water_loca = num.where(water_loca < 0.02, 1, 0)

urban_loca = mean_filter(urbanA, mask)
urban_loca = num.where(urban_loca != 1, 1, 0)

protected_loca = mean_filter(protectedA, mask)
protected_loca = num.where(protected_loca < 0.05, 1, 0)



srcRst = rasterio.open("C:\data\ws80m.tif")

def transform_rast(srcRst):
    """Transforms a raster file, reprojecting it to the master crs defined in the function,
    and outputs an array of pixel values from the raster
        Input:
            raster file (*.tif)
        Returns:
            numpy array of the pixel values of the input raster
    """
    master_crs = 'ESRI:102028'
    data = srcRst.read(1) 
    transform, width, height = calculate_default_transform(
        srcRst.crs, master_crs, srcRst.width, srcRst.height, *srcRst.bounds)
    kwargs = srcRst.meta.copy()
    kwargs.update({
        'crs': master_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    destination = num.zeros((1765, 1121), num.uint8)
    for i in range(1, srcRst.count + 1):
        reproject(
            source = rasterio.band(srcRst, i),
            destination = destination,
            src_crs = srcRst.crs,
            dst_crs = master_crs,
            resampling = Resampling.nearest)
    return destination


urban_loca = transform_rast(urban_areas)
water_loca = transform_rast(water_bodies)
slope_loca = transform_rast(slope)
wind_loca = transform_rast(wind_speed)
protected_loca = transform_rast(protected_areas)


wind_loca = mean_filter(wind_loca, mask)
wind_loca = num.where(wind_loca > 8.5, 1, 0)

slope_loca = mean_filter(slope_loca, mask)
slope_loca = num.where(slope_loca < 15, 1, 0)

water_loca = mean_filter(water_loca, mask)
water_loca = num.where(water_loca < 0.02, 1, 0)

urban_loca = mean_filter(urban_loca, mask)
urban_loca = num.where(urban_loca != 1, 1, 0)

protected_loca = mean_filter(protected_loca, mask)
protected_loca = num.where(protected_loca < 0.05, 1, 0)


final = wind_loca + protected_loca + urban_loca + slope_loca + water_loca


suit_arr = num.where(final == 5, 1, 0)

meta = wind_speed.meta 


with rasterio.open(os.path.join(filepath, 'suitables_sites.tif'), 'w', **meta) as dest:
    dest.write(suit_arr.astype('int16'), indexes=1)


suitables = rasterio.open("C:\data\suitables_sites.tif")
show(suitables)


print(suit_arr.sum())


cell_size = suitables.transform[0]


# coordinates of the raster
suitables.transform


# coordinates
x = suitables.bounds[0] + (cell_size / 2)
y = suitables.bounds[3] - (cell_size / 2)

bounds = suitables.bounds


x_coords = num.arange(bounds[0] + cell_size/2, bounds[2], cell_size)
y_coords = num.arange(bounds[1] + cell_size/2, bounds[3], cell_size)

# Getting same number of x and y coordinates
x, y = num.meshgrid(x_coords, y_coords)


x.flatten().shape, y.flatten().shape

# column stacking of arrays
coords = num.c_[x.flatten(), y.flatten()]

import pandas as pd
coords = pd.DataFrame(coords)
coords.rename(columns={0: 'lon', 1: 'lat'}, inplace=True)

from scipy.spatial import cKDTree, KDTree
points = pd.read_csv(r"C:\data\transmission_stations.txt")
tree = cKDTree(coords)
dist, indexes = tree.query(points, k=1)


print("The max distance for the suitable raster is 696.04")
print("The min distance for the suitable raster is 13.4")

