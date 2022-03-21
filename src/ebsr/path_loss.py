"""
Taken from the pysim5G repository.

Path loss calculator

Author: Edward Oughton
Date: Adapted June 2021

An implementation of a path loss calculator utilising (i) a Free Space model,
(ii) the Extended Hata model (150 MHz - 3 GHz) as found in the following
documents:

ITU-R SM.2028-2
Monte Carlo simulation methodology for the use in sharing and compatibility
studies between different radio services or systems.

"""
import numpy as np
from math import pi, sqrt, log10


# def path_loss_calculator(frequency, distance, ant_height, ant_type,
#     building_height, street_width, settlement_type, type_of_sight,
#     ue_height, above_roof, indoor, seed_value, iterations):
#     """
#     Calculate the correct path loss given a range of critera.

#     Parameters
#     ----------
#     frequency : float
#         Frequency band given in GHz.
#     distance : float
#         Distance between the transmitter and receiver in km.
#     ant_height:
#         Height of the antenna.
#     ant_type : string
#         Indicates the type of site antenna (hotspot, micro, macro).
#     building_height : int
#         Height of surrounding buildings in meters (m).
#     street_width : float
#         Width of street in meters (m).
#     settlement_type : string
#         Gives the type of settlement (urban, suburban or rural).
#     type_of_sight : string
#         Indicates whether the path is (Non) Line of Sight (LOS or NLOS).
#     ue_height : float
#         Height of the User Equipment.
#     above_roof : int
#         Indicates if the propagation line is above or below building roofs.
#         Above = 1, below = 0.
#     indoor : binary
#         Indicates if the user is indoor (True) or outdoor (False).
#     seed_value : int
#         Dictates repeatable random number generation.
#     iterations : int
#         Specifies how many iterations a specific calculation should be run for.

#     Returns
#     -------
#     path_loss : float
#         Path loss in decibels (dB)
#     model : string
#         Type of model used for path loss estimation.

#     """
#     if 0.05 < frequency <= 100:

#         path_loss = etsi_tr_138_901(frequency, distance, ant_height, ant_type,
#             building_height, street_width, settlement_type, type_of_sight,
#             ue_height, above_roof, indoor, seed_value, iterations
#         )

#         path_loss = path_loss + outdoor_to_indoor_path_loss(
#             frequency, indoor, seed_value
#         )

#         model = 'etsi_tr_138_901'

#     else:

#         raise ValueError (
#             "frequency of {} is NOT within correct range".format(frequency)
#         )

#     return round(path_loss), model

def path_loss_calculator(distance, frequency, simulation_parameters):
    """
    Calculate the free space path loss in decibels.

    FSPL(dB) = 20log(d) + 20log(f) + 32.44

    Where distance (d) is in km and frequency (f) is MHz.

    Parameters
    ----------
    distance : float
        Distance between transmitter and receiver in metres.
    simulation_parameters : dict
        Contains all simulation parameters.
    i : int
        Iteration number.
    random_variation : list
        List of random variation components.

    Returns
    -------
    path_loss : float
        The free space path loss over the given distance.
    random_variation : float
        Stochastic component.

    """

    # frequency_MHz = frequency / 1e6
    # print(frequency, frequency_MHz)
    path_loss = 20*log10(distance) + 20*log10(frequency) + 32.44

    random_variations = generate_log_normal_dist_value(
        frequency,
        2, #simulation_parameters['mu'],
        10, #simulation_parameters['sigma'],
        39, #simulation_parameters['seed_value'],
        10, #simulation_parameters['iterations']
    )

    random_variation = random_variations[0]

    return path_loss + random_variation, random_variation


def generate_log_normal_dist_value(frequency, mu, sigma, seed_value, draws):
    """
    Generates random values using a lognormal distribution, given a specific mean (mu)
    and standard deviation (sigma).
    Original function in pysim5G/path_loss.py.
    The parameters mu and sigma in np.random.lognormal are not the mean and STD of the
    lognormal distribution. They are the mean and STD of the underlying normal distribution.
    Parameters
    ----------
    frequency : float
        Carrier frequency value in Hertz.
    mu : int
        Mean of the desired distribution.
    sigma : int
        Standard deviation of the desired distribution.
    seed_value : int
        Starting point for pseudo-random number generator.
    draws : int
        Number of required values.
    Returns
    -------
    random_variation : float
        Mean of the random variation over the specified itations.
    """
    if seed_value == None:
        pass
    else:
        frequency_seed_value = seed_value * frequency * 100
        np.random.seed(int(str(frequency_seed_value)[:2]))

    normal_std = np.sqrt(np.log10(1 + (sigma/mu)**2))
    normal_mean = np.log10(mu) - normal_std**2 / 2

    random_variation  = np.random.lognormal(normal_mean, normal_std, draws)

    return random_variation
