import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sys
import os

sys.path.append(os.path.abspath('C:/Users/Tony/Documents/GitHub/EAE-593-Project'))
from main_script import (
    plot_vorticity_adv, load_datasets, plot_250_isotachs, plot_iwv, plot_ivt, 
    plot_q_uv_zeta, plot_pressure_pert, plot_250_isotachs_ageo, 
    plot_250_isotachs_ageo_stream, theta_pv_cross_section, plot_pressure_pert_new, 
    plot_sfc, plot_fgen, fgen_pv_cross_section, plot_new_thetae_grad, plot_ivt_panel, 
    plot_pv_vadv, theta_pv_cross_section_2x2, plot_250_isotachs_thickness, isentropic_sfc)

if __name__ == '__main__':
    variable_name = 'Isentropic'
    year = 2019
    month = 2
    first_day = 13
    last_day = 15
    directions = {'North': 55, 
                'East': 250, 
                'South': 20, 
                'West': 200} # units: degrees North, degrees East
    start_point = (30, 360 - 134) # units: degrees North, degrees East
    end_point = (36, 360 - 127) # units: degrees North, degrees East
    g = 9.81 # units: m/s^2
    isentropic_level = 296 # units: K

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Append the desired folder to the path
    path = os.path.join(script_dir, variable_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    for start_day in range(first_day, last_day):
        ds_pl, ds_sfc = load_datasets(year, month, start_day, start_hour=0, end_day=None, end_hour=23)
        #plot_250_isotachs(ds_pl=ds_pl, directions=directions, g=g, path=path)
        #plot_ivt(g=g, ds_pl=ds_pl, ds_sfc=ds_sfc, directions=directions, path=path)
        #plot_iwv(g=g, ds_pl=ds_pl, directions=directions, path=path)
        #plot_q_uv_zeta(g=g, ds_pl=ds_pl, directions=directions, path=path)
        #plot_pressure_pert(ds_sfc=ds_sfc, directions=directions, path=path)  
        #plot_250_isotachs_ageo(ds_pl=ds_pl, directions=directions, g=g, path=path)
        #plot_250_isotachs_ageo_stream(ds_pl=ds_pl, directions=directions, g=g, path=path) # not wrrking
        #theta_pv_cross_section(start_point=start_point, end_point=end_point, ds_pl=ds_pl, directions=directions, g=g, path=path)
        #plot_pressure_pert_new(g=g, ds_pl=ds_pl, ds_sfc=ds_sfc, directions=directions, path=path, threshold=500)
        #plot_sfc(ds_sfc=ds_sfc, directions=directions, path=path)
        #plot_fgen(g=g, ds_pl=ds_pl, directions=directions, path=path)
        #fgen_pv_cross_section(start_point=start_point, end_point=end_point, ds_pl=ds_pl, ds_sfc=ds_sfc, directions=directions, g=g, path=path)
        #plot_new_thetae_grad(ds_sfc, ds_pl, directions, path, g)
        #plot_ivt_panel(g, ds_pl, ds_sfc, directions, path)
        #plot_pv_vadv(start_point, end_point, ds_pl, directions, g, path)
        #theta_pv_cross_section_2x2(start_point, end_point, ds_pl, directions, g, path)
        #plot_250_isotachs_thickness(ds_pl=ds_pl, directions=directions, g=g, path=path)
        isentropic_sfc(ds_pl=ds_pl, directions=directions, g=g, path=path, level=isentropic_level)