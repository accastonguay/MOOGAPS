#    Copyright (C) 2010 Adam Charette-Castonguay
#    the University of Queensland
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
__author__ = "Adam Charette-Castonguay"

from rasterstats import zonal_stats
from rasterstats import point_query
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import time
from glob import glob
import multiprocessing
import logging
import rasterio
import os.path

LOG_FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(
    filename="/home/uqachare/model_file/gridinit.log",
    level=logging.INFO,
    format=LOG_FORMAT,
    filemode='w')

logger = logging.getLogger()

######################### Load tables #########################

# Fuel cost by country
fuel_cost = pd.read_csv("tables/fuel_costs.csv")

# Load transition costs for grass/grain
transition = pd.read_csv("tables/transitioning_costs.csv", index_col="current")

# Load GLPS regions
regions = pd.read_csv("tables/glps_regions.csv")

# Load landuse coding to get land use names
landuse_code = pd.read_csv("tables/landuse_coding.csv", index_col="code")
cmap = landuse_code.to_dict()['landuse']

# Beef production for different locations
beef_table = pd.read_csv("tables/beef_production.csv", index_col="Code")

######################### Set parameters #########################

# List of land covers for which livestock production is allowed
suitable_landcovers = ["area_tree", "area_sparse", "area_shrub", "area_mosaic", "area_grass", "area_crop",
                       "area_protected", "area_intact", "area_barren"]

list_suitable_cover = [i for i in cmap if 'area_'+cmap[i] in suitable_landcovers]

# k landuses to include in the simulation
landuses = ['grass_low', 'grass_high', 'alfalfa_high', 'maize', 'soybean', 'wheat']

no_transition_lu = [i for i in cmap if cmap[i] in ['crop', 'grass']]
grass = [i for i in cmap if cmap[i] == 'grass']
crop = [i for i in cmap if cmap[i] == 'crop']
forest = [i for i in cmap if cmap[i] =='tree']

def lc_summary(landcovers, cell_area):
    est_cost, area_pasture, area_crop, area_forest, suitable_area = 0,0,0,0,0
    for lc_code in landcovers:
        area = float(landcovers[lc_code])/sum(landcovers.values())* cell_area
        if lc_code in list_suitable_cover:
            suitable_area += area
            if lc_code not in no_transition_lu:
                est_cost += area
            if lc_code in grass:
                area_pasture += area
            if lc_code in crop:
                area_crop += area
            if lc_code in forest:
                area_forest += area
    d = (est_cost,  suitable_area, area_pasture, area_crop, area_forest)
    return d

def zstats_partial(feats):
    """
    Imports raster values into a dataframe partition and returns the partition

    Arguments:
    feats (array)-> partion of dataframe

    Output: returns a gridded dataframe
    """

    # Get all tif rasters in folder 'rasters'
    folder = 'rasters/'
    rasters = glob(folder + '*.tif')
    feats['centroid_column'] = feats.centroid

    # Loop over rasters to create a dictionary of raster path and raster name for column names and add values to grid
    for i in rasters:
        # Get data type of raster
        with rasterio.open(i) as dataset:
            col_dtype = dataset.meta['dtype']

        # Extract column name from raster name
        colname= os.path.basename(i).split('.')[0]
        logger.info("   Col name {} dtype {}".format(colname, col_dtype))

        # Query effective temperature
        if 'efftemp' in colname:
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate='nearest')
            feats[colname] = pd.Series([d for d in stats], index=feats.index, dtype = col_dtype)

        # Query landcover and apply lc_summary function
        elif colname =='landcover':
            # For land cover raster, count the number of covers in each cell
            stats =  zonal_stats(feats.set_geometry('geometry'), i, categorical=True)
            result = list(map(lc_summary, stats, feats['area'].values))
            for cols, pos in zip(['est_area', 'suitable_area', "pasture_area", "crop_area", 'tree_area'], range(5)):
                feats[cols] = [i[pos] for i in result]

        # Get mean accessibility
        elif colname =='accessibility':
            stats =  zonal_stats(feats.set_geometry('geometry'), i, stats = 'mean', nodata=-9999)
            feats[colname] = pd.Series([d['mean'] for d in stats], index=feats.index)

        #For all other rasters do a point query instead of zonal statistics and replace negative values by NaN
        else:
            stats = point_query(feats.set_geometry('centroid_column'), i, interpolate = 'nearest')
            feats[colname] = pd.Series([0 if d is None else 0 if d < 0 else d for d in stats], index=feats.index, dtype = col_dtype)
        logger.info("   Done with "+colname)

    # Establishment cost of 8 ('000)$ per ha where land cover requires a transition (not grass or crop) from
    # (Dietrich et al 2019 Geosci. Model Dev.
    feats['est_cost'] = feats['est_area'] * 8

    feats["opp_cost"] = feats["agri_opp_cost"] + feats['ls_opp_cost']*0.01
    logger.info("Done with opp_cost")

    feats["nutrient_availability"] = feats['nutrient_availability'].replace(0, 2)

    return feats

def parallelize(df, func, ncores):
    """
    Splits the dataframe into a number of partitions corresponding to the number of cores,
    applies a function to each partition and returns the dataframe.

    Arguments:
    df (pandas dataframe)-> Dataframe on which to apply function
    func (function)-> function to apply to partitions
    ncores (int)-> number of cores to use

    Output: returns a gridded dataframe
    """
    num_cores = int(ncores)
    # number of partitions to split dataframe based on the number of cores
    num_partitions = num_cores
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.imap(func, df_split))
    pool.close()
    pool.join()
    return df

def create_grid(location, resolution):
    """
    Create grid for a defined location and resolution
    
    Arguments:
    location (str)-> extent of simulation at country level using country code or 'Global'
    resolution (float)-> resolution of cells in degrees

    Output: returns an empty grid
    """

    # Load countries file
    extent = gpd.read_file('map/world.gpkg')
    extent.crs = {'init': 'epsg:4326'}

    # Only keep three columns
    extent = extent[['geometry', 'ADM0_A3']]

    # Filter countries on location argument
    if "Global" in location:
        extent = extent[extent.ADM0_A3.notnull() & (extent.ADM0_A3 != "ATA")]
    elif location in beef_table.index:
        extent = extent[extent.ADM0_A3 == location]
    else:
        print("Location not in choices")

    # Create grid based on extent bounds
    xmin, ymin, xmax, ymax = extent.total_bounds
    # if "Global" in location:
    #     xmin, xmax = -180, 180
    ################ Vectorized technique ################
    # resolution = float(resolution)

    resolution = 360 / 4321.
    rows = abs(int(np.ceil((ymax - ymin) / resolution)))
    cols = abs(int(np.ceil((xmax - xmin) / resolution)))
    x1 = np.cumsum(np.full((rows, cols), resolution), axis=1) + xmin - resolution
    x2 = np.cumsum(np.full((rows, cols), resolution), axis=1) + xmin
    y1 = np.cumsum(np.full((rows, cols), resolution), axis=0) + ymin - resolution
    y2 = np.cumsum(np.full((rows, cols), resolution), axis=0) + ymin
    polys = [Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) for x1, x2, y1, y2 in
             zip(x1.flatten(), x2.flatten(), y1.flatten(), y2.flatten())]
    grid = gpd.GeoDataFrame({'geometry': polys})
    grid.crs = {'init': 'epsg:4326'}
    # extent.crs = "EPSG:4326"

    # Find which grid fall within a country or on land
    grid['centroid_column'] = grid.centroid

    grid = grid.set_geometry('centroid_column')

    start = time.time()

    grid = gpd.sjoin(grid, extent, how='left', op='within')
    grid.drop(['index_right'], axis=1, inplace=True)

    print('Joined grid to country in {} seconds'.format(time.time()-start))

    # Filter cells to keep those on land
    grid = grid.merge(regions, how='left')

    if "Global" in location:
        grid = grid[(grid.ADM0_A3.notnull())]
    elif location in beef_table.index:
        grid = grid[(grid.ADM0_A3 == location) & (grid.ADM0_A3.notnull())]
    else:
        print("no beef demand for location")

    grid = grid.set_geometry('geometry')

    print('### Done Creating grid in {} seconds. '.format(time.time()-start))

    return grid

def main(location = 'TLS', resolution = 0.1, ncores =1, export_folder ='.'):
    """
    Main function that optimises beef production for a given location and resolution, using a given number of cores.
    
    Arguments:
    location (str)-> extent of simulation at country level using country code or 'Global'
    resolution (float)-> resolution of cells in degrees
    ncores (int)-> number of cores to use
    export_folder (str)-> folder where the output file is exported
    constraint (str)-> Whether the production should be constrained to equal actual country-specific production, or unconstrained 
    
    Output: Writes the grid as GPKG file
    """

    grid = create_grid(location, resolution)
    logger.info('Shape of grid: {}'.format(grid.shape[0]))

    # Measure area of cells in hectare
    grid = grid.to_crs({'proj': 'cea'})
    grid["area"] = grid['geometry'].area / 10. ** 4
    grid = grid.to_crs({'init': 'epsg:4326'})

    grid = grid.loc[grid.area < 1000000]

    logger.info("Done calculating area")

    # Parallelise the input data collection
    grid = parallelize(grid, zstats_partial, ncores)

    logger.info("Done Collecting inputs")

    # Get net cropping area to 'protect'
    grid['net_fodder_area'] = grid['sum_area'] - grid['current_cropping']

    ### Only keep cells where there is feed ###
    foddercrop_list = ['barley', 'maize', 'rapeseed', 'rice', 'sorghum', 'soybean', 'wheat']
    feeds = [c for c in grid.columns if 'grass' in c or c in foddercrop_list]
    grid = grid.loc[(grid[feeds].sum(axis=1) > 0) & (grid['suitable_area'] - grid['net_fodder_area'] > 0)]

    ######### Export #########
    # Columns to drop
    cols_drop = [c for c in ['centroid_column', 'soilmoisture', "sum_area", "gdd", 'ls_opp_cost', 'agri_opp_cost','est_area'] if c in grid.columns]

    grid.drop(cols_drop, axis = 1).set_geometry('geometry').to_file(export_folder+"/grid.gpkg", driver="GPKG")
    logger.info("Exporting results finished")

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('location', help='Spatial extent of simulation')
    argparser.add_argument('resolution', help='Resolution of pixels (in degrees)')
    argparser.add_argument('ncores', help='Number of cores for multiprocessing')
    argparser.add_argument('export_folder', help='Name of exported file')

    args = argparser.parse_args()
    location = args.location
    resolution = args.resolution
    ncores = args.ncores
    export_folder = args.export_folder

    main(location, resolution, ncores, export_folder)