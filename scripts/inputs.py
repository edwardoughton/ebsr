"""
Model inputs.

Written by Ed Oughton.

April 2022.

"""

PARAMETERS = {
    'confidence_level': 90,
    'active_users_perc': 10,
    # 'market_share_perc': 25,
    # 'cells_per_site_2G': 3,
    # 'cells_per_site_4G': 3,
    'busy_hour_traffic_perc': 10,
    'dl_spectrum_GHz_hic': 30,
    'dl_spectrum_GHz_umc': 20,
    'dl_spectrum_GHz_lmc': 20,
    'dl_spectrum_GHz_lic': 10,
    'traffic_per_user_gb_hic': 30,
    'traffic_per_user_gb_umc': 30,
    'traffic_per_user_gb_lmc': 30,
    'traffic_per_user_gb_lic': 30,
    'urban_rural_pop_density_km2': 500,
}


COSTS = {
    #all costs in $USD
    'equipment': 40000,
    'site_build': 30000,
    'installation': 30000,
    'backhaul': 20000,
}
