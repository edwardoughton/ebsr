"""
Model inputs.

Written by Ed Oughton.

April 2022.

"""

PARAMETERS = {
    'confidence_level': 95,
    'active_users_perc': 5,
    'market_share_perc': 25,
    'busy_hour_traffic_perc': 15,
    'dl_spectrum_GHz_hic': 30,
    'dl_spectrum_GHz_umc': 20,
    'dl_spectrum_GHz_lmc': 20,
    'dl_spectrum_GHz_lic': 10,
    'traffic_per_user_gb_hic': 20,
    'traffic_per_user_gb_umc': 15,
    'traffic_per_user_gb_lmc': 10,
    'traffic_per_user_gb_lic': 5,
    'urban_rural_pop_density_km2': 500,
}


COSTS = {
    #all costs in $USD
    'equipment': 40000,
    'site_build': 30000,
    'installation': 30000,
    'backhaul': 20000,
}
