"""
EBSR run script.

Written by Ed Oughton.

March 2022.

"""
import os
import configparser
from math import ceil
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import random, uniform
from itertools import tee
from tqdm import tqdm

from inputs import PARAMETERS, COSTS

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
        (active_users_perc/100) /
        area_km2
    )

    return active_users


def find_site_density(region, parameters, traffic_km2, spectrum_portfolio,
    capacity_lookup_table, confidence_interval):
    """
    For a given region, estimate the number of needed sites.

    Parameters
    ----------
    region : dicts
        Data for a single region.
    option : dict
        Contains the scenario and strategy. The strategy string controls
        the strategy variants being tested in the model and is defined based
        on the type of technology generation, core and backhaul, and the
        strategy for infrastructure sharing, the number of networks in each
        geotype, spectrum and taxation.
    global_parameters : dict
        All global model parameters.
    country_parameters : dict
        All country specific parameters.
    capacity_lut : dict
        A dictionary containing the lookup capacities.
    ci : int
        Confidence interval.

    Return
    ------
    site_density : float
        Estimated site density.

    """
    if region['geotype'] == 'rural':
        spectrum_portfolio = spectrum_portfolio[:1]

    unique_densities = set()

    capacity = 0

    ### Get a unique set of site densities
    for item in spectrum_portfolio:

        density_capacities = lookup_capacity(
            capacity_lookup_table,
            item,
            10,
            '4G',
            confidence_interval
        )

        for item in density_capacities:
            site_density, capacity = item
            unique_densities.add(site_density)

    density_lut = []

    ### Now get capacity for each unique site density + frequency combination
    for density in list(unique_densities):

        capacity = 0

        for item in spectrum_portfolio:

            density_capacities = lookup_capacity(
                capacity_lookup_table,
                item,
                10,
                '4G',
                confidence_interval
            )

            for density_capacity in density_capacities:

                if density_capacity[0] == density:
                    capacity += density_capacity[1]

        density_lut.append((density, capacity))

    density_lut = sorted(density_lut, key=lambda tup: tup[0])

    max_density, max_capacity = density_lut[-1]
    min_density, min_capacity = density_lut[0]

    # max_capacity = max_capacity * bandwidth
    # min_capacity = min_capacity * bandwidth

    if traffic_km2 > max_capacity:

        return max_density

    elif traffic_km2 < min_capacity:

        return min_density

    else:

        for a, b in pairwise(density_lut):

            lower_density, lower_capacity  = a
            upper_density, upper_capacity  = b

            # lower_capacity = lower_capacity * bandwidth
            # upper_capacity = upper_capacity * bandwidth

            if lower_capacity <= traffic_km2 < upper_capacity:

                site_density = interpolate(
                    lower_capacity, lower_density,
                    upper_capacity, upper_density,
                    traffic_km2
                )

                return site_density


def lookup_capacity(capacity_lut, frequency, bandwidth,
    generation, confidence_interval):
    """
    Use lookup table to find the combination of spectrum bands
    which meets capacity by frequency, bandwidth, technology
    generation and site density.

    Parameters
    ----------
    capacity_lut : dict
        A dictionary containing the lookup capacities.
    frequency : string
        The frequency band in Megahertz.
    bandwidth : string
        Channel bandwidth.
    generation : string
        The cellular generation such as 4G or 5G.
    confidence_interval : int
        Confidence interval.

    Returns
    -------
    site_densities_to_capacities : list of tuples
        Returns a list of site density to capacity tuples.

    """
    output = []

    for item in capacity_lut:
        if item['frequency_GHz'] == frequency:
            if item['bandwidth_MHz'] == bandwidth:
                if item['generation'] == generation:
                    if item['confidence_interval'] == confidence_interval:

                        tup = (item['sites_per_km2'], item['capacity_mbps_km2'])

                        output.append(tup)

    return output


def interpolate(x0, y0, x1, y1, x):
    """
    Linear interpolation between two values.
    """
    y = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    return y


def estimate_site_upgrades(region, generation, total_sites_required,
    parameters):
    """
    Estimate the number of greenfield sites and brownfield upgrades for the
    single network being modeled.

    Parameters
    ----------
    region : dict
        Contains all regional data.
    total_sites_required : int
        Number of sites needed to meet demand.
    parameters : dict
        All country specific parameters.

    Returns
    -------
    region : dict
        Contains all regional data.

    """
    region['existing_mno_sites'] = ceil(region['total_estimated_sites'] *
        (parameters['market_share_perc']/100))

    region['existing_4G_sites'] = ceil(region['sites_4G'] *
        (parameters['market_share_perc']/100))

    if total_sites_required > region['existing_mno_sites']:

        region['new_mno_sites'] = (int(round(total_sites_required -
            region['existing_mno_sites'])))

        if region['existing_mno_sites'] > 0:
            if generation == '4G' and region['existing_4G_sites'] > 0 :
                region['upgraded_mno_sites'] = (region['existing_mno_sites'] -
                    region['existing_4G_sites'])
            else:
                region['upgraded_mno_sites'] = region['existing_mno_sites']
        else:
            region['upgraded_mno_sites'] = 0

    else:
        region['new_mno_sites'] = 0

        if generation == '4G' and region['existing_4G_sites'] > 0 :
            to_upgrade = total_sites_required - region['existing_4G_sites']
            region['upgraded_mno_sites'] = to_upgrade if to_upgrade >= 0 else 0
        else:
            region['upgraded_mno_sites'] = total_sites_required

    return region


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


def calc_costs(region, costs):
    """
    Calculate cost.

    """
    cost = 0

    for i in range(0, region['new_mno_sites']):
        for key, item in costs.items():
            cost += item

    for i in range(0, region['upgraded_mno_sites']):
        for key, item in costs.items():

            if key == 'site_build':
                continue
            if key == 'backhaul':
                continue

            cost += item

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

        # if not path.endswith('AFG.csv'):
        #     continue

        data = pd.read_csv(path)

        data = data[[
            'country_name',
            'iso3',
            'iso2',
            'population',
            'area_km2',
            'smartphone_users_perc',
            'sp_users',
            'traffic_per_user_gb',
            'traffic_perc',
            'confidence_interval',
            'total_regional_cost',
        ]]

        data = data.groupby([
            'country_name',
            'iso3',
            'iso2',
            # 'hour',
            'traffic_perc',
            'traffic_per_user_gb',
            'confidence_interval',
            # 'generation',
            'smartphone_users_perc']).agg(
            population = ('population','sum'),
            area_km2 = ('area_km2','sum'),
            sp_users = ('sp_users','sum'),
            total_cost = ('total_regional_cost','sum'),
            ).reset_index()

        data['sp_users_km2'] = data['sp_users'] / data['area_km2']

        data = data.sort_values('smartphone_users_perc')

        interim = []

        quant = 0

        for idx, item in data.iterrows():

            total_cost = item['total_cost']
            incremental_cost = total_cost - quant
            cost_per_sp_user = (total_cost - quant) / item['sp_users']

            interim.append({
                'country_name': item['country_name'],
                'iso3': item['iso3'],
                'iso2': item['iso2'],
                'traffic_perc': item['traffic_perc'],
                'traffic_per_user_gb': item['traffic_per_user_gb'],
                'confidence_interval': item['confidence_interval'],
                'population': item['population'],
                'area_km2': item['area_km2'],
                'smartphone_users_perc': item['smartphone_users_perc'],
                'sp_users': item['sp_users'],
                'sp_users_km2': item['sp_users_km2'],
                'total_cost': total_cost,
                'incremental_cost': incremental_cost if incremental_cost > 0 else 0,
                'cost_per_sp_user': cost_per_sp_user if cost_per_sp_user > 0 else 0,
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

    #Load supply data inputs
    path = os.path.join(DATA_INTERMEDIATE, 'luts', 'capacity_lut_by_frequency.csv')
    capacity_lookup_table = pd.read_csv(path)#[:1]
    # confidence_intervals = capacity_lookup_table['confidence_interval'].unique()[2:3] #50%
    capacity_lookup_table = capacity_lookup_table[capacity_lookup_table['generation'] == '4G']
    confidence_intervals = [PARAMETERS['confidence_level']]
    capacity_lookup_table = capacity_lookup_table[[
        'confidence_interval', 'inter_site_distance_m', 'site_area_km2',
        'sites_per_km2', 'frequency_GHz', 'bandwidth_MHz', 'generation',
        'path_loss_dB', 'received_power_dBm', 'interference_dBm', 'noise_dB', 'sinr_dB',
        'spectral_efficiency_bps_hz', 'capacity_mbps', 'capacity_mbps_km2'
    ]]
    capacity_lookup_table = capacity_lookup_table.to_dict('records')

    active_users_perc = PARAMETERS['active_users_perc']
    market_share_perc = PARAMETERS['market_share_perc']

    for country in tqdm(countries):

        # if not country['iso3'] == 'AFG':
        #     continue

        output = []

        #Load population data
        path = os.path.join(DATA_INTERMEDIATE, country['iso3'], 'regional_data.csv')

        if not os.path.exists(path):
            # missing_regional_data.add(country['country_name'])
            continue

        spectrum_portfolio = find_spectrum_portfolio(country, PARAMETERS)

        handle = 'traffic_per_user_gb_{}'.format(country['income'])
        traffic_per_user_gb = PARAMETERS[handle]

        regional_data = pd.read_csv(path)#[:1]

        #Load coverage data
        path = os.path.join(DATA_INTERMEDIATE, country['iso3'], 'regional_coverage.csv')
        if not os.path.exists(path):
            # missing_regional_data.add(country['country_name'])
            continue
        regional_coverage = pd.read_csv(path)#[:1]
        regional_coverage = regional_coverage[['GID_id', 'total_estimated_sites', 'sites_4G']]
        regional_data = regional_data.merge(regional_coverage, on='GID_id')#[:1]

        for i in tqdm(range(0, 100+1, 5)):

            smartphone_users_perc = i

            for idx, region in regional_data.iterrows():

                # if not region['GID_id'] == 'GBR.3.26.1_1':
                #     continue

                if region['area_km2'] == 0:
                    continue

                if smartphone_users_perc == 0:
                    sp_users = 1
                    network_sp_users = 1
                else:
                    sp_users = int(round(region['population'] * (smartphone_users_perc/100)))
                    network_sp_users = sp_users / (100/market_share_perc)

                if region['population_km2'] > PARAMETERS['urban_rural_pop_density_km2']:
                    region['geotype'] = 'urban'
                else:
                    region['geotype'] = 'rural'

                if round(network_sp_users) == 0:
                    continue

                active_users_km2 = get_active_users(
                    network_sp_users,
                    active_users_perc,  # % of active users
                    region['area_km2'],
                    )

                per_user_mbps = per_user_hourly_data(
                    traffic_per_user_gb,
                    PARAMETERS['busy_hour_traffic_perc'],
                    )

                traffic_km2 = active_users_km2 * per_user_mbps

                for confidence_interval in confidence_intervals:

                    site_density_km2 = find_site_density(
                        region,
                        PARAMETERS,
                        traffic_km2,
                        spectrum_portfolio,
                        capacity_lookup_table,
                        confidence_interval
                    )

                    total_sites_required = ceil(site_density_km2 * region['area_km2'])

                    region = estimate_site_upgrades(
                        region,
                        '4G',
                        total_sites_required,
                        PARAMETERS
                    )

                #     metrics = capacity_metrics(
                #         capacity_lookup_table,
                #         confidence_interval,
                #         site_density_km2[confidence_interval]
                #     )

                    cost = calc_costs(region, COSTS)

                    output.append({
                        'country_name': country['country_name'],
                        'GID_id': region['GID_id'],
                        'iso3': country['iso2'],
                        'iso2': country['iso2'],
                        'regional_level': country['regional_level'],
                        'population': region['population'],
                        'area_km2': region['area_km2'],
                        'population_km2': region['population_km2'],
                        'smartphone_users_perc': smartphone_users_perc,
                        'sp_users': sp_users,
                        'network_sp_users': network_sp_users,
                        'traffic_per_user_gb': traffic_per_user_gb,
                        # 'hour': hour,
                        'traffic_perc': PARAMETERS['busy_hour_traffic_perc'],
                        'active_users_km2': active_users_km2,
                        'per_user_mbps': per_user_mbps,
                        'traffic_km2': traffic_km2,
                        'confidence_interval': confidence_interval,
                        'total_estimated_sites': region['total_estimated_sites'],
                        'sites_4G': region['sites_4G'],
                        'existing_mno_sites': region['existing_mno_sites'],
                        'existing_4G_sites ': region['existing_4G_sites'],
                        'new_mno_sites ': region['new_mno_sites'],
                        'upgraded_mno_sites ': region['upgraded_mno_sites'],
                        'cost_km2': cost / region['area_km2'],
                        'cost_per_sp_user': int(round(cost / sp_users)),
                        'total_regional_cost': int(round(cost / sp_users) * sp_users),
                    })

        output = pd.DataFrame(output)

        directory = os.path.join(RESULTS, 'regional_results')
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = '{}.csv'.format(country['iso3'])
        path = os.path.join(directory, filename)
        output.to_csv(path, index=False)

    collect_results(PARAMETERS)
