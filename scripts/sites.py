"""
Preprocess sites scripts.

Written by Ed Oughton.

Winter 2020

"""
import os
import configparser
import json
import csv
import math
import glob
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import Polygon, MultiPolygon, mapping, shape, MultiLineString, LineString
from shapely.ops import transform, unary_union, nearest_points
import fiona
from fiona.crs import from_epsg
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import networkx as nx
from rtree import index
import numpy as np
import random
from datetime import datetime

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_INTERMEDIATE = os.path.join(BASE_PATH, 'intermediate')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def find_country_list(continent_list):
    """
    This function produces country information by continent.

    Parameters
    ----------
    continent_list : list
        Contains the name of the desired continent, e.g. ['Africa']

    Returns
    -------
    countries : list of dicts
        Contains all desired country information for countries in
        the stated continent.

    """
    glob_info_path = os.path.join(BASE_PATH, 'global_information.csv')
    countries = pd.read_csv(glob_info_path, encoding = "ISO-8859-1")

    countries = countries[countries.exclude != 1]

    if len(continent_list) > 0:
        data = countries.loc[countries['continent'].isin(continent_list)]
    else:
        data = countries

    output = []

    for index, country in data.iterrows():

        output.append({
            'country_name': country['country'],
            'iso3': country['ISO_3digit'],
            'iso2': country['ISO_2digit'],
            'regional_level': country['lowest'],
            'region': country['region'],
            'exclude': country['exclude'],
            'operators': country['operators'],
        })

    return output


def get_regional_data(country):
    """
    Extract regional data including luminosity and population.

    Parameters
    ----------
    country : string
        Three digit ISO country code.

    """
    iso3 = country['iso3']
    level = country['regional_level']
    gid_level = 'GID_{}'.format(level)

    path_output = os.path.join(DATA_INTERMEDIATE, iso3, 'regional_coverage.csv')

    # if os.path.exists(path_output):
    #     return #print('Regional data already exists')

    path_in = os.path.join(DATA_INTERMEDIATE, iso3, 'regional_data.csv')
    regions = pd.read_csv(path_in)

    results = []

    for index, region in regions.iterrows():

        results.append({
            'GID_0': region['GID_0'],
            'GID_id': region['GID_id'],
            'GID_level': region['GID_level'],
            'population': region['population'],
            'area_km2': region['area_km2'],
            'population_km2': region['population_km2'],
        })

    results = estimate_sites(results, country)

    results_df = pd.DataFrame(results)

    results_df.to_csv(path_output, index=False)

    return


def estimate_sites(data, country):
    """
    Estimate the sites by region.

    Parameters
    ----------
    data : dataframe
        Pandas df with regional data.
    iso3 : string
        ISO3 country code.

    Returns
    -------
    output : list of dicts
        All regional data with estimated sites.

    """
    iso3 = country['iso3']

    output = []

    population = 0

    for region in data:

        if region['population'] == None:
            continue

        population += int(region['population'])

    path = os.path.join(DATA_RAW, 'site_counts', 'hybrid_site_data.csv')
    site_data = pd.read_csv(path, encoding = "ISO-8859-1")
    site_data['estimated_towers'] = pd.to_numeric(site_data['estimated_towers'])

    site_data = site_data.loc[site_data['ISO3'] == iso3]
    if len(site_data) >= 1:
        coverage_2G = site_data['coverage_2G_perc'].sum()
        coverage_4G = site_data['coverage_4G_perc'].sum()
        towers = site_data['estimated_towers'].sum()
    else:
        coverage_2G = 0
        coverage_4G = 0
        towers = 0

    population_covered_2G = population * (coverage_2G / 100)
    population_covered_4G = population * (coverage_4G / 100)

    if population_covered_2G < population_covered_4G:
        population_covered_2G = population_covered_4G

    towers_2G = towers * (coverage_2G / 100)
    if np.isnan(towers_2G) or population_covered_2G == 0 or towers_2G == 0:
        towers_2G = 0
        towers_per_pop_2G = 0
    else:
        towers_per_pop_2G = towers_2G / population_covered_2G

    towers_4G = towers * (coverage_4G / 100)
    if np.isnan(towers_4G) or population_covered_4G == 0 or towers_4G == 0:
        towers_4G = 0
        towers_per_pop_4G = 0
    else:
        towers_per_pop_4G = towers_4G / population_covered_4G

    data = sorted(data, key=lambda k: k['population_km2'], reverse=True)

    covered_pop_so_far_2G = 0
    covered_pop_so_far_4G = 0

    for region in data:

        if covered_pop_so_far_2G < population_covered_2G:
            total_existing_sites_2G = round(region['population'] * towers_per_pop_2G)
        else:
            total_existing_sites_2G = 0

        if covered_pop_so_far_4G < population_covered_4G:
            total_existing_sites_4G = round(region['population'] * towers_per_pop_4G)
        else:
            total_existing_sites_4G = 0

        output.append({
                'GID_0': region['GID_0'],
                'GID_id': region['GID_id'],
                'GID_level': region['GID_level'],
                'population': region['population'],
                'area_km2': region['area_km2'],
                'population_km2': round(region['population_km2'],1),
                'total_existing_sites_2G': total_existing_sites_2G,
                'total_existing_sites_4G': total_existing_sites_4G,
            })

        if region['population'] == None:
            continue

        covered_pop_so_far_2G += region['population']
        covered_pop_so_far_4G += region['population']

    return output


def collect_data(countries):
    """
    Collect data into single file.

    """
    output = []

    for country in countries:

        iso3 = country['iso3']
        path = os.path.join(DATA_INTERMEDIATE, iso3, 'regional_coverage.csv')

        sites = pd.read_csv(path)

        output.append({
            'country_name': country['country_name'],
            'iso3': country['iso3'],
            'population': sites['population'].sum(),
            'area_km2': sites['area_km2'].sum(),
            'total_existing_sites_2G': sites['total_existing_sites_2G'].sum(),
            'total_existing_sites_4G': sites['total_existing_sites_4G'].sum(),
        })

    output = pd.DataFrame(output)
    path_output = os.path.join(DATA_INTERMEDIATE, 'total_sites.csv')
    output.to_csv(path_output, index=False)


if __name__ == '__main__':

    countries = find_country_list([])#[::-1]

    for country in countries:

        if country['exclude'] == 1:
            continue

        # if not country['iso3'] == 'GBR':
        #     continue

        print('-Working on {}: {}'.format(country['iso3'], country['country_name']))

        get_regional_data(country)

    collect_data(countries)
