"""
Visualize results.

Written by Ed Oughton.

May 5th 2022.

"""
import os
import sys
import configparser
import numpy as np
import pandas as pd
# import geopandas as gpd
# import glob
import matplotlib.pyplot as plt
# from matplotlib.dates import YearLocator
# from matplotlib.ticker import ScalarFormatter
# from matplotlib.patches import Patch
# import matplotlib.patches
# import matplotlib.transforms as transforms
# from matplotlib.colors import LinearSegmentedColormap

# import contextily as ctx

# import matplotlib.gridspec as gridspec

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
RESULTS = os.path.join(BASE_PATH, '..', 'results')
EXPORT_FIGURES = os.path.join(BASE_PATH, '..', 'vis', 'figures')

if not os.path.exists(EXPORT_FIGURES):
    os.makedirs(EXPORT_FIGURES)



if __name__ == "__main__":

    path = os.path.join(RESULTS, 'country_results.csv')
    data = pd.read_csv(path)

    print(data)
