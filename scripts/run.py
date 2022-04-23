"""
EBSR run script.

Written by Ed Oughton.

March 2022.

"""
import os
import configparser
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import random, uniform
from itertools import tee
from tqdm import tqdm

from inputs import parameters

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_INTERMEDIATE = os.path.join(BASE_PATH, 'intermediate')
RESULTS = os.path.join(BASE_PATH, '..', 'results')


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

    if len(continent_list) > 0:
        data = countries.loc[countries['continent'].isin(continent_list)]
    else:
        data = countries

    countries['income'] = countries['income'].fillna('lic')

    output = []

    for index, country in data.iterrows():

        output.append({
            'country_name': country['country'],
            'iso3': country['ISO_3digit'],
            'iso2': country['ISO_2digit'],
            'regional_level': country['lowest'],
            'income': country['income'].lower(),
        })

    return output


def import_hourly_distribution(hourly_distribution, parameters):
    """
    Select hours to run model for.

    """
    output = {}

    # for index, item in hourly_distribution.iterrows():

    #     key = int(item['hour'])
    #     value = item['percentage_share']#.values[0]

    #     if not key == 17:
    #         continue

    #     output[key] = value

    key = int(17)
    value = parameters['busy_hour_traffic_perc']
    output[key] = value

    return output


def find_spectrum_portfolio(country, parameters):
    """
    Function to find the spectrum bands to be used.

    """
    handle = 'dl_spectrum_GHz_{}'.format(country['income'])
    dl_spectrum_GHz = parameters[handle]

    if dl_spectrum_GHz == 10:
        spectrum_portfolio = [0.8]
    elif dl_spectrum_GHz == 20:
        spectrum_portfolio = [0.8, 1.8]
    elif dl_spectrum_GHz == 30:
        spectrum_portfolio = [0.8, 1.8, 2.6]
    else:
        print('Did not recognize spectrum portfolio size')

    return spectrum_portfolio


def per_user_hourly_data(data_per_month_gb, percentage_share):
    """
    Estimate the per user data demand in Mbps.

    """

    per_user_mbps = (
        data_per_month_gb *
        1000 *
        8 *
        (1/30) *
        (percentage_share/100) *
        (1/3600)
    )

    return per_user_mbps


def get_active_users(network_sp_users, active_users_perc, area_km2):
    """
    Estimate the number of active users.

    """
    active_users = round(
        network_sp_users *
        # (smartphone_users_perc/100) *
        active_users_perc /
        area_km2
    )

    return active_users


def find_site_density(traffic_km2, spectrum_portfolio,
    capacity_lookup_table, confidence_intervals):
    """
    Given a traffic capacity, find a site density to serve
    this traffic.

    """
    output = {}

    for frequency in spectrum_portfolio:
        for confidence_interval in confidence_intervals:

            density_lut = []

            for item in capacity_lookup_table:

                if item['frequency_GHz'] == frequency:
                    if confidence_interval == item['confidence_interval']:

                        density_lut.append(
                            (item['sites_per_km2'], item['capacity_mbps_km2'])
                        )

            density_lut = sorted(density_lut, key=lambda tup: tup[0])

            max_density, max_capacity = density_lut[-1]
            min_density, min_capacity = density_lut[0]

            key = '{}_{}'.format(frequency, confidence_interval)

            if traffic_km2 > max_capacity:

                output[key] = {
                    'frequency': frequency,
                    'sites_per_km2': max_density,
                    'sites_per_km2_upper': max_density,
                    'capacity_mbps_km2_upper': max_capacity,
                    'sites_per_km2_lower': max_density,
                    'capacity_mbps_km2_lower': max_capacity,
                }

            elif traffic_km2 < min_capacity:

                output[key] = {
                    'frequency': frequency,
                    'sites_per_km2': min_density,
                    'sites_per_km2_upper': min_density,
                    'capacity_mbps_km2_upper': min_capacity,
                    'sites_per_km2_lower': min_density,
                    'capacity_mbps_km2_lower': min_capacity,

                }

            else:

                for a, b in pairwise(density_lut):

                    lower_density, lower_capacity  = a
                    upper_density, upper_capacity  = b

                    if lower_capacity <= traffic_km2 < upper_capacity:

                        site_density = interpolate(
                            lower_capacity, lower_density,
                            upper_capacity, upper_density,
                            traffic_km2
                        )

                        output[key] = {
                            'frequency': frequency,
                            'sites_per_km2': site_density,
                            'sites_per_km2_upper': upper_density,
                            'capacity_mbps_km2_upper': upper_capacity,
                            'sites_per_km2_lower': lower_density,
                            'capacity_mbps_km2_lower': lower_capacity,
                        }
    print(output)
    return output


def interpolate(x0, y0, x1, y1, x):
    """
    Linear interpolation between two values.
    """
    y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    return y


def pairwise(iterable):
    """
    Return iterable of 2-tuples in a sliding window.
    >>> list(pairwise([1,2,3,4]))
    [(1,2),(2,3),(3,4)]
    """
    a, b = tee(iterable)
    next(b, None)

    return zip(a, b)


def capacity_metrics(capacity_lookup_table, confidence_interval, site_density_km2):
    """

    """
    output = {}

    for item in capacity_lookup_table:
        if item['confidence_interval'] == confidence_interval:
            if item['sites_per_km2'] == site_density_km2['sites_per_km2_upper']:
                output['inter_site_distance_m'] = item['inter_site_distance_m']
                output['site_area_km2'] = item['site_area_km2']
                output['sites_per_km2'] = item['sites_per_km2']
                output['frequency_GHz'] = item['frequency_GHz']
                output['bandwidth_MHz'] = item['bandwidth_MHz']
                output['generation'] = item['generation']
                output['path_loss_dB'] = item['path_loss_dB']
                output['received_power_dBm'] = item['received_power_dBm']
                output['interference_dBm'] = item['interference_dBm']
                output['noise_dB'] = item['noise_dB']
                output['sinr_dB'] = item['sinr_dB']
                output['spectral_efficiency_bps_hz'] = item['spectral_efficiency_bps_hz']
                output['capacity_mbps'] = item['capacity_mbps']
                output['capacity_mbps_km2'] = item['capacity_mbps_km2']

    return output


def calc_costs(site_density_km2):
    """
    Calculate cost.

    """
    cost = site_density_km2['sites_per_km2'] * 100000

    return cost


def collect_results(parameters):
    """
    Get final results.

    """
    output = []

    directory = os.path.join(RESULTS, 'regional_results')
    path = os.path.join(directory, '*.csv')
    paths = glob.glob(path)

    for path in paths:

        if not country['iso3'] == 'GBR':
            continue

        data = pd.read_csv(path)
        data = data[[
            'country_name',
            'iso3',
            'iso2',
            'hour',
            'traffic_perc',
            'traffic_per_user_gb',
            'confidence_interval',
            'generation',
            'population',
            'area_km2',
            'smartphone_users_perc',
            'sp_users',
            'total_regional_cost',
        ]]

        data = data.groupby([
            'country_name',
            'iso3',
            'iso2',
            'hour',
            'traffic_perc',
            'traffic_per_user_gb',
            'confidence_interval',
            'generation',
            'smartphone_users_perc']).agg(
            population = ('population','sum'),
            area_km2 = ('area_km2','sum'),
            sp_users = ('sp_users','sum'),
            total_cost = ('total_regional_cost','sum'),
            ).reset_index()

        data = data.sort_values('smartphone_users_perc')

        interim = []

        quant = 0

        for idx, item in data.iterrows():

            total_cost = item['total_cost']

            interim.append({
                'country_name': item['country_name'],
                'iso3': item['iso3'],
                'iso2': item['iso2'],
                'traffic_perc': item['traffic_perc'],
                'traffic_per_user_gb': item['traffic_per_user_gb'],
                'confidence_interval': item['confidence_interval'],
                'generation': item['generation'],
                'population': item['population'],
                'area_km2': item['area_km2'],
                'smartphone_users_perc': item['smartphone_users_perc'],
                'sp_users': item['sp_users'],
                'total_cost': total_cost,
                'incremental_cost': total_cost - quant,
                'cost_per_sp_user': (total_cost - quant) / item['sp_users'],
            })

            quant = item['total_cost']

        output = output + interim

    output = pd.DataFrame(output)

    path = os.path.join(RESULTS, 'country_results.csv')
    output.to_csv(path, index=False)

    return


if __name__ == "__main__":

    #Load countries list
    countries = find_country_list([])

    #Load demand data inputs
    path = os.path.join(DATA_RAW, 'hourly_demand', 'hourly_demand.csv')
    hourly_distribution = pd.read_csv(path)
    hourly_distribution = import_hourly_distribution(hourly_distribution, parameters)

    #Load supply data inputs
    path = os.path.join(DATA_INTERMEDIATE, 'luts', 'capacity_lut_by_frequency.csv')
    capacity_lookup_table = pd.read_csv(path)#[:1]
    # confidence_intervals = capacity_lookup_table['confidence_interval'].unique()[2:3] #50%
    capacity_lookup_table = capacity_lookup_table[capacity_lookup_table['generation'] == '4G']
    confidence_intervals = [parameters['confidence_level']]
    capacity_lookup_table = capacity_lookup_table[[
        'confidence_interval', 'inter_site_distance_m', 'site_area_km2',
        'sites_per_km2', 'frequency_GHz', 'bandwidth_MHz', 'generation',
        'path_loss_dB', 'received_power_dBm', 'interference_dBm', 'noise_dB', 'sinr_dB',
        'spectral_efficiency_bps_hz', 'capacity_mbps', 'capacity_mbps_km2'
    ]]
    capacity_lookup_table = capacity_lookup_table.to_dict('records')

    active_users_perc = parameters['active_users_perc']
    market_share_perc = parameters['market_share_perc']

    for country in tqdm(countries):

        if not country['iso3'] == 'GBR':
            continue

        output = []

        #Load population data
        path = os.path.join(DATA_INTERMEDIATE, country['iso3'], 'regional_data.csv')

        if not os.path.exists(path):
            # missing_regional_data.add(country['country_name'])
            continue

        handle = 'traffic_per_user_gb_{}'.format(country['income'])
        traffic_per_user_gb = parameters[handle]

        spectrum_portfolio = find_spectrum_portfolio(country, parameters)

        regional_data = pd.read_csv(path)#[:1]

        #Load coverage data
        path = os.path.join(DATA_INTERMEDIATE, country['iso3'], 'regional_coverage.csv')
        regional_coverage = pd.read_csv(path)[:1]
        regional_coverage = regional_coverage[['GID_id', 'total_estimated_sites', 'sites_4G']]
        regional_data = regional_data.merge(regional_coverage, on='GID_id')

        for i in tqdm(range(1, 100+1)):

            if not i == 100:
                continue

            smartphone_users_perc = i

            for idx, region in regional_data.iterrows():

                if region['area_km2'] == 0:
                    continue

                sp_users = int(round(region['population'] * (smartphone_users_perc/100)))

                network_sp_users = sp_users * (market_share_perc/100)

                if round(network_sp_users) == 0:
                    continue

                active_users_km2 = get_active_users(
                    network_sp_users,
                    active_users_perc,  # % of active users
                    region['area_km2'],
                    )

                for hour, traffic_perc in hourly_distribution.items():

                    per_user_mbps = per_user_hourly_data(
                        traffic_per_user_gb,
                        traffic_perc,
                        )

                    traffic_km2 = active_users_km2 * per_user_mbps

                    site_density_km2 = find_site_density(
                        traffic_km2,
                        spectrum_portfolio,
                        capacity_lookup_table,
                        confidence_intervals
                    )

                    for confidence_interval in confidence_intervals:

                        metrics = capacity_metrics(
                            capacity_lookup_table,
                            confidence_interval,
                            site_density_km2[confidence_interval]
                        )

                        costs = calc_costs(
                            site_density_km2[confidence_interval]
                            )

                        output.append({
                            'country_name': country['country_name'],
                            'GID_id': region['GID_id'],
                            'iso3': country['iso2'],
                            'iso2': country['iso2'],
                            'regional_level': country['regional_level'],
                            'population': region['population'],
                            'area_km2': region['area_km2'],
                            'smartphone_users_perc': smartphone_users_perc,
                            'sp_users': sp_users,
                            'network_sp_users': network_sp_users,
                            'traffic_per_user_gb': traffic_per_user_gb,
                            'hour': hour,
                            'traffic_perc': traffic_perc,
                            'active_users_km2': active_users_km2,
                            'per_user_mbps': per_user_mbps,
                            'traffic_km2': traffic_km2,
                            'confidence_interval': confidence_interval,
                            'inter_site_distance_m': metrics['inter_site_distance_m'],
                            'site_area_km2': metrics['site_area_km2'],
                            'sites_per_km2': metrics['sites_per_km2'],
                            'total_sites': metrics['site_area_km2'] * region['area_km2'],
                            'pop_per_sites': region['population'] / (metrics['site_area_km2'] * region['area_km2']),
                            'network_sp_users_per_site': network_sp_users / (metrics['site_area_km2'] * region['area_km2']),
                            'active_users_per_site': active_users_km2 / (metrics['site_area_km2'] * region['area_km2']),
                            'frequency_GHz': metrics['frequency_GHz'],
                            'bandwidth_MHz': metrics['bandwidth_MHz'],
                            'generation': metrics['generation'],
                            'path_loss_dB': metrics['path_loss_dB'],
                            'received_power_dBm': metrics['received_power_dBm'],
                            'interference_dBm': metrics['interference_dBm'],
                            'noise_dB': metrics['noise_dB'],
                            'sinr_dB': metrics['sinr_dB'],
                            'spectral_efficiency_bps_hz': metrics['spectral_efficiency_bps_hz'],
                            'capacity_mbps': metrics['capacity_mbps'],
                            'capacity_mbps_km2': metrics['capacity_mbps_km2'],
                            'cost_km2': costs * (100/market_share_perc),
                            'cost_per_sp_user': int(round((costs * region['area_km2']) / sp_users)),
                            'total_regional_cost': int(round((costs * region['area_km2']) / sp_users) * sp_users),
                        })

        output = pd.DataFrame(output)

        directory = os.path.join(RESULTS, 'regional_results')
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = '{}.csv'.format(country['iso3'])
        path = os.path.join(directory, filename)
        output.to_csv(path, index=False)

    collect_results(parameters)
