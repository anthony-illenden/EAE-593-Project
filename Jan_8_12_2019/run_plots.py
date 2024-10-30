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
from main_script import plot_vorticity_adv, load_datasets, plot_250_isotachs

if __name__ == '__main__':
    year = 2019
    month = 1
    start_day = 12
    ds_pl, ds_sfc = load_datasets(year, month, start_day, start_hour=0, end_day=None, end_hour=23)
    directions = {'North': 55, 
                  'East': 250, 
                  'South': 20, 
                  'West': 200} # units: degrees North, degrees East
    start_point = (42.0, 360-136) # units: degrees North, degrees East
    end_point = (38.0, 360-124) # units: degrees North, degrees East
    g = 9.81 # units: m/s^2
    # Get the current working directory
    current_dir = os.getcwd()
    # Append the desired folder to the path
    path = os.path.join(current_dir, 'Isotachs')
    plot_250_isotachs(ds_pl=ds_pl, directions=directions, g=g, path=path)
    #plot_vorticity_adv(ds_pl=ds_pl, directions=directions, g=g, path=path)