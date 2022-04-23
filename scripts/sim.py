"""
Runner for system_simulator.py

Written by Edward Oughton

Adapted from the India5G repository.

January 2022

"""
import os
import sys
import configparser
import csv

import math
import fiona
from shapely.geometry import shape, Point, LineString, mapping
import numpy as np
from random import choice
from rtree import index
import pandas as pd

from collections import OrderedDict

from ebsr.generate_hex import produce_sites_and_site_areas
from ebsr.system_simulator import SimulationManager

np.random.seed(42)

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']
DATA_INTERMEDIATE = os.path.join(BASE_PATH, 'intermediate')

def generate_receivers(site_area, parameters, grid):
    """

    Generate receiver locations as points within the site area.

    Sampling points can either be generated on a grid (grid=1)
    or more efficiently between the transmitter and the edge
    of the site (grid=0) area.

    Parameters
    ----------
    site_area : polygon
        Shape of the site area we want to generate receivers within.
    parameters : dict
        Contains all necessary simulation parameters.
    grid : int
        Binary indicator to dictate receiver generation type.

    Output
    ------
    receivers : List of dicts
        Contains the quantity of desired receivers within the area boundary.

    """
    receivers = []

    if grid == 1:

        geom = shape(site_area[0]['geometry'])
        geom_box = geom.bounds

        minx = geom_box[0]
        miny = geom_box[1]
        maxx = geom_box[2]
        maxy = geom_box[3]

        id_number = 0

        x_axis = np.linspace(
            minx, maxx, num=(
                int(math.sqrt(geom.area) / (math.sqrt(geom.area)/50))
                )
            )
        y_axis = np.linspace(
            miny, maxy, num=(
                int(math.sqrt(geom.area) / (math.sqrt(geom.area)/50))
                )
            )

        xv, yv = np.meshgrid(x_axis, y_axis, sparse=False, indexing='ij')
        for i in range(len(x_axis)):
            for j in range(len(y_axis)):
                receiver = Point((xv[i,j], yv[i,j]))
                indoor_outdoor_probability = np.random.rand(1,1)[0][0]
                if geom.contains(receiver):
                    receivers.append({
                        'type': "Feature",
                        'geometry': {
                            "type": "Point",
                            "coordinates": [xv[i,j], yv[i,j]],
                        },
                        'properties': {
                            'ue_id': "id_{}".format(id_number),
                            "misc_losses": parameters['rx_misc_losses'],
                            "gain": parameters['rx_gain'],
                            "losses": parameters['rx_losses'],
                            "ue_height": float(parameters['rx_height']),
                            "indoor": (True if float(indoor_outdoor_probability) < \
                                float(0.5) else False),
                        }
                    })
                    id_number += 1

                else:
                    pass

    else:

        centroid = shape(site_area[0]['geometry']).centroid

        coord = site_area[0]['geometry']['coordinates'][0][0]
        path = LineString([(coord), (centroid)])
        length = int(path.length)
        increment = int(length / 20)

        indoor = parameters['indoor_users_percentage'] / 100

        id_number = 0
        for increment_value in range(1, 11):
            point = path.interpolate(increment * increment_value)
            indoor_outdoor_probability = np.random.rand(1,1)[0][0]
            receivers.append({
                'type': "Feature",
                'geometry': mapping(point),
                'properties': {
                    'ue_id': "id_{}".format(id_number),
                    "misc_losses": parameters['rx_misc_losses'],
                    "gain": parameters['rx_gain'],
                    "losses": parameters['rx_losses'],
                    "ue_height": float(parameters['rx_height']),
                    "indoor": (True if float(indoor_outdoor_probability) < \
                        float(indoor) else False),
                }
            })
            id_number += 1

    return receivers


def obtain_percentile_values(results, transmission_type, parameters, confidence_intervals,
    environment, site_radius, frequency, bandwidth, generation, ant_type):
    """

    Get the threshold value for a metric based on a given percentiles.

    Parameters
    ----------
    results : list of dicts
        All data returned from the system simulation.

    parameters : dict
        Contains all necessary simulation parameters.

    Output
    ------
    percentile_site_results : dict
        Contains the percentile value for each site metric.

    """
    inter_site_distance = site_radius * 2
    site_area_km2 = math.sqrt(3) / 2 * inter_site_distance ** 2 / 1e6
    sites_per_km2 = 1 / site_area_km2

    sectors = parameters['sectorization']

    output = []

    path_loss_values = []
    received_power_values = []
    interference_values = []
    noise_values = []
    sinr_values = []
    spectral_efficiency_values = []
    estimated_capacity_values = []
    estimated_capacity_values_km2 = []

    for result in results:

        path_loss_values.append(result['path_loss'])

        received_power_values.append(result['received_power'])

        interference_values.append(result['interference'])

        noise_values.append(result['noise'])

        sinr = result['sinr']
        if sinr == None:
            sinr = 0
        else:
            sinr_values.append(sinr)

        spectral_efficiency = result['spectral_efficiency']
        if spectral_efficiency == None:
            spectral_efficiency = 0
        else:
            spectral_efficiency_values.append(spectral_efficiency)

        estimated_capacity = result['capacity_mbps']
        if estimated_capacity == None:
            estimated_capacity = 0
        else:
            estimated_capacity_values.append(estimated_capacity)

        estimated_capacity_km2 = result['capacity_mbps_km2']
        if estimated_capacity_km2 == None:
            estimated_capacity_km2 = 0
        else:
            estimated_capacity_values_km2.append(estimated_capacity_km2)

    for confidence_interval in confidence_intervals:

        output.append({
            'confidence_interval': confidence_interval,
            'environment': environment,
            'inter_site_distance_m': inter_site_distance,
            'site_area_km2': site_area_km2,
            'sites_per_km2': sites_per_km2,
            'frequency_GHz': frequency,
            'bandwidth_MHz': bandwidth,
            'number_of_sectors': sectors,
            'generation': generation,
            'ant_type': ant_type,
            'tranmission_type': transmission_type,
            'path_loss_dB': np.percentile(
                path_loss_values, confidence_interval #<- low path loss is better
            ),
            'received_power_dBm': np.percentile(
                received_power_values, 100 - confidence_interval
            ),
            'interference_dBm': np.percentile(
                interference_values, confidence_interval #<- low interference is better
            ),
            'noise_dB': np.percentile(
                noise_values, confidence_interval #<- low interference is better
            ),
            'sinr_dB': np.percentile(
                sinr_values, 100 - confidence_interval
            ),
            'spectral_efficiency_bps_hz': np.percentile(
                spectral_efficiency_values, 100 - confidence_interval
            ),
            'capacity_mbps': np.percentile(
                estimated_capacity_values, 100 - confidence_interval
            ),
            'capacity_mbps_km2': np.percentile(
                estimated_capacity_values_km2, 100 - confidence_interval
            ) * sectors
        })

    return output


if __name__ == '__main__':

    PARAMETERS = {
        'iterations': 1,
        'seed_value1_3G': 1,
        'seed_value2_3G': 2,
        'seed_value1_4G': 3,
        'seed_value2_4G': 4,
        'seed_value1_5G': 5,
        'seed_value2_5G': 6,
        'seed_value1_urban': 7,
        'seed_value2_urban': 8,
        'seed_value1_suburban': 9,
        'seed_value2_suburban': 10,
        'seed_value1_rural': 11,
        'seed_value2_rural': 12,
        'seed_value1_free-space': 13,
        'seed_value2_free-space': 14,
        'indoor_users_percentage': 50,
        'los_breakpoint_m': 500,
        'tx_macro_baseline_height': 30,
        'tx_macro_power': 40,
        'tx_macro_gain': 16,
        'tx_macro_losses': 1,
        'tx_micro_baseline_height': 10,
        'tx_micro_power': 24,
        'tx_micro_gain': 5,
        'tx_micro_losses': 1,
        'rx_gain': 0,
        'rx_losses': 4,
        'rx_misc_losses': 4,
        'rx_height': 1.5,
        'building_height': 5,
        'street_width': 20,
        'above_roof': 0,
        'network_load': 100,
        'percentile': 50,
        'sectorization': 3,
        'mnos': 2,
        'asset_lifetime': 10,
        'discount_rate': 3.5,
        'opex_percentage_of_capex': 10,
    }

    SPECTRUM_PORTFOLIO = [
        # (1.9, 10, '3G', '1x1'),
        # (2.1, 10, '3G', '1x1'),
        (0.8, 10, '4G', '2x2'),
        (1.8, 10, '4G', '2x2'),
        (2.6, 10, '4G', '2x2'),
    ]

    ANT_TYPES = [
        ('macro'),
    ]

    MODULATION_AND_CODING_LUT = {
        # ETSI. 2018. ‘5G; NR; Physical Layer Procedures for Data
        # (3GPP TS 38.214 Version 15.3.0 Release 15)’. Valbonne, France: ETSI.
        # Generation MIMO CQI Index	Modulation	Coding rate
        # Spectral efficiency (bps/Hz) SINR estimate (dB)
        '3G': [
            ('3G', '1x1', 'NA', 'NA', 78, 0.15, -6.7),
            ('3G', '1x1', 'NA', 'NA', 120, 0.23, -4.7),
            ('3G', '1x1', 'NA', 'NA', 193, 0.37, -2.3),
            ('3G', '1x1', 'NA', 'NA', 308, 0.6, 0.2),
            ('3G', '1x1', 'NA', 'NA', 449, 0.8, 2.4),
            ('3G', '1x1', 'NA', 'NA', 602, 1.1, 4.3),
            ('3G', '1x1', 'NA', 'NA', 378, 1.4, 5.9),
            ('3G', '1x1', 'NA', 'NA', 490, 1.9, 8.1),
            ('3G', '1x1', 'NA', 'NA', 616, 2.4, 10.3),
            ('3G', '1x1', 'NA', 'NA', 466, 2.7, 11.7),
            ('3G', '1x1', 'NA', 'NA', 567, 3.3, 14.1),
            ('3G', '1x1', 'NA', 'NA', 666, 3.9, 16.3),
            ('3G', '1x1', 'NA', 'NA', 772, 4.5, 18.7),
            ('3G', '1x1', 'NA', 'NA', 973, 5.1, 21),
            ('3G', '1x1', 'NA', 'NA', 948, 5.5, 22.7),
        ],
        '4G': [
            ('4G', '2x2', 1, 'QPSK', 78, 0.3, -6.7),
            ('4G', '2x2', 2, 'QPSK', 120, 0.46, -4.7),
            ('4G', '2x2', 3, 'QPSK', 193, 0.74, -2.3),
            ('4G', '2x2', 4, 'QPSK', 308, 1.2, 0.2),
            ('4G', '2x2', 5, 'QPSK', 449, 1.6, 2.4),
            ('4G', '2x2', 6, 'QPSK', 602, 2.2, 4.3),
            ('4G', '2x2', 7, '16QAM', 378, 2.8, 5.9),
            ('4G', '2x2', 8, '16QAM', 490, 3.8, 8.1),
            ('4G', '2x2', 9, '16QAM', 616, 4.8, 10.3),
            ('4G', '2x2', 10, '64QAM', 466, 5.4, 11.7),
            ('4G', '2x2', 11, '64QAM', 567, 6.6, 14.1),
            ('4G', '2x2', 12, '64QAM', 666, 7.8, 16.3),
            ('4G', '2x2', 13, '64QAM', 772, 9, 18.7),
            ('4G', '2x2', 14, '64QAM', 973, 10.2, 21),
            ('4G', '2x2', 15, '64QAM', 948, 11.4, 22.7),
        ],
        '5G': [
            ('5G', '4x4', 1, 'QPSK', 78, 0.15, -6.7),
            ('5G', '4x4', 2, 'QPSK', 193, 1.02, -4.7),
            ('5G', '4x4', 3, 'QPSK', 449, 2.21, -2.3),
            ('5G', '4x4', 4, '16QAM', 378, 3.20, 0.2),
            ('5G', '4x4', 5, '16QAM', 490, 4.00, 2.4),
            ('5G', '4x4', 6, '16QAM', 616, 5.41, 4.3),
            ('5G', '4x4', 7, '64QAM', 466, 6.20, 5.9),
            ('5G', '4x4', 8, '64QAM', 567, 8.00, 8.1),
            ('5G', '4x4', 9, '64QAM', 666, 9.50, 10.3),
            ('5G', '4x4', 10, '64QAM', 772, 11.00, 11.7),
            ('5G', '4x4', 11, '64QAM', 873, 14.00, 14.1),
            ('5G', '4x4', 12, '256QAM', 711, 16.00, 16.3),
            ('5G', '4x4', 13, '256QAM', 797, 19.00, 18.7),
            ('5G', '4x4', 14, '256QAM', 885, 22.00, 21),
            ('5G', '4x4', 15, '256QAM', 948, 25.00, 22.7),
        ]
    }

    CONFIDENCE_INTERVALS = [
        # 5,
        # 50,
        95,
    ]

    def generate_site_radii(min, max, increment):
        for n in range(min, max, increment):
            yield n

    INCREMENT_MA = (125, 50000, 125)#50000,125) #(400, 40400, 1000)

    SITE_RADII = {
        'macro': {
            # 'urban':
            #     generate_site_radii(INCREMENT_MA[0],INCREMENT_MA[1],INCREMENT_MA[2]),
            # 'suburban':
            #     generate_site_radii(INCREMENT_MA[0],INCREMENT_MA[1],INCREMENT_MA[2]),
            # 'rural':
            #     generate_site_radii(INCREMENT_MA[0],INCREMENT_MA[1],INCREMENT_MA[2]),
            'free-space':
                generate_site_radii(INCREMENT_MA[0],INCREMENT_MA[1],INCREMENT_MA[2])
            },
        }

    unprojected_point = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': (0, 0),
            },
        'properties': {
            'site_id': 'Radio Tower'
            }
        }

    unprojected_crs = 'epsg:4326'
    projected_crs = 'epsg:3857'

    environments =[
        # 'urban',
        # 'suburban',
        # 'rural'
        'free-space'
    ]

    output = []

    for environment in environments:
        for ant_type in ANT_TYPES:
            site_radii_generator = SITE_RADII[ant_type]
            for site_radius in site_radii_generator[environment]:

                # if not site_radius == 4400:
                #     continue

                if environment == 'urban' and site_radius > 5000:
                    continue
                if environment == 'suburban' and site_radius > 15000:
                    continue

                print('--working on {}: {}'.format(environment, site_radius))

                transmitter, interfering_transmitters, site_area, int_site_areas = \
                    produce_sites_and_site_areas(
                        unprojected_point['geometry']['coordinates'],
                        site_radius,
                        unprojected_crs,
                        projected_crs
                        )

                receivers = generate_receivers(site_area, PARAMETERS, 1)

                for frequency, bandwidth, generation, transmission_type in SPECTRUM_PORTFOLIO:

                    print('{}, {}, {}, {}'.format(frequency, bandwidth, generation, transmission_type))

                    MANAGER = SimulationManager(
                        transmitter, interfering_transmitters, ant_type,
                        receivers, site_area, PARAMETERS
                        )

                    results = MANAGER.estimate_link_budget(
                        frequency,
                        bandwidth,
                        generation,
                        ant_type,
                        transmission_type,
                        environment,
                        MODULATION_AND_CODING_LUT,
                        PARAMETERS
                        )

                    percentile_results = obtain_percentile_values(
                        results, transmission_type, PARAMETERS, CONFIDENCE_INTERVALS,
                        environment, site_radius, frequency, bandwidth, generation, ant_type,
                    )

                    output = output + percentile_results

    results_directory = os.path.join(DATA_INTERMEDIATE, 'luts')
    path = os.path.join(results_directory, 'capacity_lut_by_frequency.csv')
    output = pd.DataFrame(output)
    output.to_csv(path, index=False)
