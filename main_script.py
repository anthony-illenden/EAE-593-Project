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

def load_datasets(year, month, start_day, start_hour=0, end_day=None, end_hour=23):
    """
    Load datasets for specified start day, month, and year.
    
    Parameters:
        year (int): The year of the data.
        month (int): The month of the data.
        start_day (int): The start day of the data.
        start_hour (int, optional): The hour of the start day. Defaults to 0.
        end_day (int, optional): The end day of the data. Defaults to start_day.
        end_hour (int, optional): The hour of the end day. Defaults to 23.
        
    Returns:
        tuple: Xarray Datasets for pressure level and surface datasets.
    """
    
    # Set end_day to start_day if not provided
    if end_day is None:
        end_day = start_day
    
    # Get the last day of the month
    last_day_of_month = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    last_day_str = f"{last_day_of_month.day:02d}"  # format last day as two digits
    
    # Format date and time strings
    year_month = f'{year}{month:02d}'
    start_time = f'{year}{month:02d}{start_day:02d}{start_hour:02d}'  # yyyymmddhh (start)
    end_time = f'{year}{month:02d}{end_day:02d}{end_hour:02d}'  # yyyymmddhh (end)
    
    # Define URLs for pressure level datasets with specific time ranges
    urls = {
        'temperature_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_130_t.ll025sc.{start_time}_{end_time}.nc',
        'geopotential_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_129_z.ll025sc.{start_time}_{end_time}.nc',
        'humidity_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_133_q.ll025sc.{start_time}_{end_time}.nc',
        'v_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_132_v.ll025uv.{start_time}_{end_time}.nc',
        'u_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_131_u.ll025uv.{start_time}_{end_time}.nc',
        'w_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_135_w.ll025sc.{start_time}_{end_time}.nc',
        'pv_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_060_pv.ll025sc.{start_time}_{end_time}.nc',
        
        # Define URLs for surface datasets to cover the full month using last_day_of_month
        'mslp_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.128_151_msl.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc',
        'u_wind_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.228_131_u10n.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc',
        'v_wind_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.228_132_v10n.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc'
    }

    # Initialize empty dictionaries for datasets
    datasets = {}
    
    # Try to load datasets from the URLs
    for var, url in urls.items():
        try:
            tds_catalog = TDSCatalog(url)
            ds_url = tds_catalog.datasets[0].access_urls['OPENDAP']
            ds = xr.open_dataset(ds_url).metpy.parse_cf()
            datasets[var] = ds
            print(f"Successfully loaded {var}")

        except Exception as e:
            print(f"Error loading {var}: {e}")

    # Merge pressure level datasets if available
    ds_pl, ds_sfc = None, None
    
    try:
        ds_pl = xr.merge([datasets['temperature_pl'], datasets['geopotential_pl'], datasets['humidity_pl'], 
                          datasets['v_wind_pl'], datasets['u_wind_pl'], datasets['w_wind_pl'], datasets['pv_pl']])
        print("Successfully merged pressure level datasets")
    except KeyError as e:
        print(f"Error merging pressure level datasets: {e}")

    # Merge surface datasets if available
    try:
        ds_sfc = xr.merge([datasets['mslp_sfc'], datasets['v_wind_sfc'], datasets['u_wind_sfc']])
        print("Successfully merged surface datasets")
    except KeyError as e:
        print(f"Error merging surface datasets: {e}")

    # Synchronize time dimensions
    try:
        if ds_pl is not None and ds_sfc is not None:
            first_time_pl, last_time_pl = ds_pl['time'].min().values, ds_pl['time'].max().values
            ds_sfc = ds_sfc.sel(time=slice(first_time_pl, last_time_pl))
    except KeyError as e:
        print(f"Error accessing 'time' in the datasets: {e}")
    except Exception as e:
        print(f"An error occurred during slicing: {e}")
        
    return ds_pl, ds_sfc

def plot_vorticity_adv(g, ds_pl, directions, path):
    # Loop over the reanlysis time steps
    for i in range(0, len(ds_pl.time.values)):
        # Slice the dataset to get the data for the current time step
        ds_pl_sliced = ds_pl.isel(time=i)
        
        # Slice the dataset to get the data for the region of interest
        ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North']+10, directions['South']-10), longitude=slice(directions['West']-10, directions['East']+10))

        # Slice the dataset to get the data for the pressure levels at 850 hPa
        t_sliced = ds_pl_sliced['T'].sel(level=slice(500, 900)) # units: K
        u_sliced = ds_pl_sliced['U'].sel(level=slice(500, 900)) # units: m/s
        v_sliced = ds_pl_sliced['V'].sel(level=slice(500, 900)) # units: m/s
        z_sliced = ds_pl_sliced['Z'].sel(level=slice(500, 900)) / g # units: m
        q_sliced = ds_pl_sliced['Q'].sel(level=900) * 1000 # units: g/kg

        # Smoothing the geopotential height field (z_sliced)
        n_reps = 80
        u_smooth_raw = xr.apply_ufunc(mpcalc.smooth_n_point, u_sliced, 9, n_reps, dask='parallelized',
                                output_dtypes=[u_sliced.dtype])
        v_smooth_raw = xr.apply_ufunc(mpcalc.smooth_n_point, v_sliced, 9, n_reps, dask='parallelized',
                                output_dtypes=[v_sliced.dtype])


        # Convert the raw geostrophic wind output (Pint Quantities) back to xarray DataArray
        # This step ensures that the coordinates (level, lat, lon) are retained
        u = xr.DataArray(u_smooth_raw, coords=u_smooth_raw.coords, dims=u_smooth_raw.dims) * units('m/s')
        v = xr.DataArray(v_smooth_raw, coords=v_smooth_raw.coords, dims=v_smooth_raw.dims) * units('m/s')

        # Calculate absolute vorticity and select the 500 and 700 hPa levels
        absolute_vorticity = mpcalc.absolute_vorticity(u, v)
        absolute_vorticity_500 = absolute_vorticity.sel(level=500)
        abs_vort_adv = mpcalc.advection(absolute_vorticity, u, v)
        abs_vort_adv_500 = abs_vort_adv.sel(level=500)
        z_500 = z_sliced.sel(level=500)

        # Select the 500-hPa u-wind and v-wind
        u_500 = u_sliced.sel(level=500)
        v_500 = v_sliced.sel(level=500)

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_pl_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        isohypses = plt.contour(z_500['longitude'], z_500['latitude'], z_500, colors='black', levels=np.arange(5000, 6200, 60), linewidths=1)
        try:
            plt.clabel(isohypses, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label for geopotential heights.")
        cs2 = ax.contour(absolute_vorticity_500['longitude'], absolute_vorticity_500['latitude'], absolute_vorticity_500*10**5, np.arange(-40, 50, 3),colors='grey', linewidths=1.0, linestyles='dotted')
        try:
            plt.clabel(cs2, fontsize=10, inline=1, inline_spacing=3, fmt='%d', rightside_up=True, use_clabeltext=True)
        except IndexError:
            print("No contours to label for absolute vorticity.")
        
        cf = plt.contourf(abs_vort_adv_500['longitude'], abs_vort_adv_500['latitude'], abs_vort_adv_500*10**8, cmap=plt.cm.BrBG, levels=np.arange(-2, 2.2, 0.2), extend='both')
        plt.colorbar(cf, orientation='vertical', label = 'Absolute Vorticity Advection $\\times 10^{8}$ s$^{-2}$', fraction=0.046, pad=0.04)

        # Plot the 850-hPa wind barbs
        step = 10
        ax.barbs(u_500['longitude'][::step], u_500['latitude'][::step], u_500[::step, ::step], v_500[::step, ::step], length=6, color='black')

        # Adding custom legend entries (hardcoded)
        isohypse_line = plt.Line2D([0], [0], color='black', linewidth=1, label='Geopotential Height (m)')
        vorticity_line = plt.Line2D([0], [0], color='grey', linestyle = "dotted", linewidth=1, label='Absolute Vorticity (s$^{-1}$)')

        # Creating the legend with the custom entries
        ax.legend(handles=[isohypse_line, vorticity_line], loc='upper right')


        # Add the title, set the map extent, and add map features
        plt.title(f'500-hPa Absolute Vorticity Advection, Geopotential Height, and Winds | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='white')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')

        # Add gridlines and format longitude/latitude labels
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/vadv_{formatted_datetime}.png')
        plt.close()


def plot_250_isotachs(ds_pl, directions, g, path):
    # Loop over the reanalysis time steps
    for i in range(0, len(ds_pl.time.values)):
        # Slice the dataset to get the data for the current time step
        ds_pl_sliced = ds_pl.isel(time=i)

        # Slice the dataset to get the data for the region of interest
        ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
        
        # Slice the dataset to get the data for the pressure level at 250 hPa
        u_250 = ds_pl_sliced['U'].sel(level=250) # units: m/s
        v_250 = ds_pl_sliced['V'].sel(level=250) # units: m/s
        z_250 = ds_pl_sliced['Z'].sel(level=250) / g # units: m
        pressure_levels = u_250.level * 100 # units: Pa

        # Calculate the wind speed at 250 hPa
        wind_speed_250 = np.sqrt(u_250**2 + v_250**2)

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_pl_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Smooth the mslp and wind speed
        z_smoothed = gaussian_filter(z_250, sigma=3)
        wnd_smoothed = gaussian_filter(wind_speed_250, sigma=3)

        # Define the color levels and colors for the isotachs
        levels = np.arange(20, 95, 5)
        colors = ['#daedfb', '#b7dcf6', '#91bae4', '#7099ce', '#6a999d', '#72ad63', '#77c14a', '#cad955', '#f8cf4f', '#f7953c', '#ef5f28', '#e13e26', '#cd1e28', '#b1181e', '#901617']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Create the plot 
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the geopotential heights and isotachs
        isohypses = plt.contour(z_250['longitude'], z_250['latitude'], z_smoothed, colors='black', levels=np.arange(8700, 11820, 60), linewidths=1)
        try:
            ax.clabel(isohypses, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label.")
        cf = plt.contourf(u_250['longitude'], u_250['latitude'], wnd_smoothed, cmap=cmap, norm=norm, levels=levels, extend='max')
        plt.colorbar(cf, orientation='vertical', label='Wind Speed (ms$^{-1}$)', fraction=0.046, pad=0.04)


        step = 10
        ax.barbs(u_250['longitude'][::step], u_250['latitude'][::step], u_250[::step, ::step], v_250[::step, ::step], length=6, color='black')

        # Add the title, set the map extent, and add map features
        plt.title(f'ERA5 Reanalysis 250-hPa Isoatachs, Geopotential Height, and Barbs | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North'] - 5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/isotachs_{formatted_datetime}.png')
        plt.close()

def theta_pv_cross_section(start_point, end_point, ds_pl, directions, g, path):
    # Loop over the reanalysis time steps
    for i in range(0, len(ds_pl.time.values)):
        ds_pl_sliced = ds_pl.isel(time=i)

        # Slice the dataset to get the data for the region of interest
        ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))

        # Get the u, v, t, and q fields at multiple levels
        u_sliced = ds_pl_sliced['U'].sel(level=slice(150, 1000)) # units: m/s
        v_sliced = ds_pl_sliced['V'].sel(level=slice(150, 1000)) # units: m/s
        t_sliced = ds_pl_sliced['T'].sel(level=slice(150, 1000)) # units: K
        q_sliced = ds_pl_sliced['Q'].sel(level=slice(150, 1000)) # units: kg/kg
        pressure_levels = t_sliced['level'] # units: 
        pressure_levels_ivt = u_sliced.level[::-1] * 100 # units: Pa

        # Calculate potential temperature
        theta = mpcalc.potential_temperature(pressure_levels, t_sliced) # units: K

        # Add the potential temperature (theta) to the dataset
        theta_da = xr.DataArray(theta, dims=['level', 'latitude', 'longitude'],
                                coords={'level': t_sliced['level'], 
                                        'latitude': t_sliced['latitude'], 
                                        'longitude': t_sliced['longitude']},
                                attrs={'units': 'K'})

        ds_pl_sliced['THETA'] = theta_da

        lats = u_sliced['latitude'][:]
        lons = u_sliced['longitude'][:] 


        # Calculate the integrated vapor transport (IVT) using the u- and v-wind components and the specific humidity
        u_ivt = -1 / g * np.trapz(u_sliced * q_sliced, pressure_levels_ivt, axis=0)
        v_ivt = -1 / g * np.trapz(v_sliced * q_sliced, pressure_levels_ivt, axis=0)

        # Calculate the IVT magnitude
        ivt = np.sqrt(u_ivt**2 + v_ivt**2)

        # Create an xarray DataArray for the IVT
        ivt_da = xr.DataArray(ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']})

        # Define the color levels and colors for the IVT plot
        ivt_levels = [250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800]
        ivt_colors = ['#ffff00', '#ffe400', '#ffc800', '#ffad00', '#ff8200', '#ff5000', '#ff1e00', '#eb0010', '#b8003a', '#850063', '#570088']
        ivt_cmap = mcolors.ListedColormap(ivt_colors)
        ivt_norm = mcolors.BoundaryNorm(ivt_levels, ivt_cmap.N)

        # Mask the IVT values below 250 kg/m/s and create a filtered DataArray for the u- and v-wind components
        mask = ivt_da >= 250
        u_ivt_filtered = xr.DataArray(u_ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']}).where(mask, drop=True)
        v_ivt_filtered = xr.DataArray(v_ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']}).where(mask, drop=True)

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_pl_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Now do the cross-section interpolation
        lat_values = np.linspace(start_point[0], end_point[0])
        lon_values = np.linspace(start_point[1], end_point[1])
        x = xr.DataArray(lon_values, dims='Lat_Lon')
        y = xr.DataArray(lat_values, dims='Lat_Lon')
        
        # Interpolate the sliced data for the cross-section
        ds_cross = ds_pl_sliced.interp(longitude=x, latitude=y, method='nearest')

        # Get the temperature, potential vorticity, and theta 
        pv_crossed = ds_cross['PV'].sel(level=slice(150, 1000)) # units: K
        theta_crossed = ds_cross['THETA'].sel(level=slice(150, 1000)) # units: K
        #fgen_crossed = ds_cross['FGEN'].sel(level=slice(150, 1000)) # units: K / 100km / hr

        # Get the pressure levels
        pressure_levels = pv_crossed['level'] # units: hPa

        # Define the color levels and colors for the potential vorticity
        levels = np.arange(0, 3.26, 0.25)
        colors = ['white', '#d1e9f7', '#a5cdec', '#79a3d5', '#69999b', '#78af58', '#b0cc58', '#f0d95f', '#de903e', '#cb5428', '#b6282a', '#9b1622', '#7a1419']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Smooth the PV and potential temperature 
        pv_smoothed = gaussian_filter(pv_crossed, sigma=1)
        theta_smoothed = gaussian_filter(theta_crossed, sigma=1)

        # Create the figure 
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the isentropes and potential vorticity 
        isentropes = ax.contour(ds_cross['longitude'], pressure_levels, theta_smoothed, colors='black', levels=np.arange(250, 450, 1))
        ax.clabel(isentropes, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        #fgen_c = ax.contour(ds_cross['longitude'], pressure_levels, fgen_3hr, colors='black', levels=np.arange(-10, 10, 1), linestyles='dashed')
        #ax.clabel(fgen_c, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        pv_cf = ax.contourf(ds_cross['longitude'], pressure_levels, pv_smoothed * 1e6, cmap=cmap, levels=levels, norm=norm, extend='max')
        plt.colorbar(pv_cf, orientation='vertical', label='PV (m$^2$ s$^{-1}$ K kg$^{-1}$)', fraction=0.046, pad=0.04)


        # Adding the isentropes to the legend with a custom label
        theta_line = plt.Line2D([0], [0], color='black', linewidth=1, label=r'$\theta$ Potential Temperature (K)')
        #fgen_line = plt.Line2D([0], [0], color='black', linewidth=1, linestyle='dashed', label='Frontogenesis (K / 100km / hr)')

        # Creating the legend with the custom entries
        ax.legend(handles=[theta_line], loc='upper right')
        ax.set_yticks(pressure_levels)
        ax.set_yticklabels(list(pressure_levels.values))
        plt.ylim(1000, 600)
        plt.xlabel('Longitude (degrees E)')
        plt.ylabel('Pressure (hPa)')
        plt.title(f'ERA5 Reanalysis Vertical Cross-Section of PV and $\\theta$ | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}')

        # Manually set y-axis tick marks
        custom_ticks = [1000, 900, 800, 700, 600]  
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels([str(tick) for tick in custom_ticks])

        # Add labels "A" and "A'" to the bottom left and right
        ax.text(0, -0.1, 'A', transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
        ax.text(1, -0.1, "A'", transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

        # Add an inset of the plan view to provide additional context 
        ax_inset = fig.add_axes([0.10, 0.63, 0.25, 0.25], projection=ccrs.PlateCarree())
        ax_inset.set_extent([directions['East'], directions['West'], directions['South'], directions['North']])
        ax_inset.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        ax_inset.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax_inset.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax_inset.add_feature(cfeature.OCEAN, color='white')
        ax_inset.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_inset.add_feature(cfeature.LAND, color='#fbf5e9')
        ax_inset.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]],
                            color="black", marker="o", transform=ccrs.PlateCarree())
        # Add text annotations
        ax_inset.text(start_point[1], start_point[0], "A", transform=ccrs.PlateCarree(),
                    verticalalignment='bottom', horizontalalignment='right', fontweight='bold', fontsize=12)
        ax_inset.text(end_point[1], end_point[0], "A'", transform=ccrs.PlateCarree(),
                    verticalalignment='bottom', horizontalalignment='right', fontweight='bold', fontsize=12)

        #isohypses = ax_inset.contour(z_500['longitude'], z_500['latitude'], z_500, colors='black', levels=np.arange(5000, 6000, 50), linewidths=1)
        #isobars = ax_inset.contour(mslp['longitude'], mslp['latitude'], gaussian_filter(mslp, sigma=1), colors='black', levels=np.arange(960, 1040, 4), linewidths=0.25)
        #plt.clabel(isobars, inline=True, inline_spacing=5, fontsize=6, fmt='%i')
        c = ax_inset.contour(ivt_da['longitude'], ivt_da['latitude'], gaussian_filter(ivt_da, sigma=1), colors='black', levels=ivt_levels, linewidths=0.5)
        cf = ax_inset.contourf(ivt_da['longitude'], ivt_da['latitude'], gaussian_filter(ivt_da, sigma=1), cmap=ivt_cmap, levels=ivt_levels, norm=ivt_norm, extend='max')



        # Plot the IVT vectors
        step = 5 
        plt.quiver(u_ivt_filtered['longitude'][::step], u_ivt_filtered['latitude'][::step], u_ivt_filtered[::step, ::step], v_ivt_filtered[::step, ::step], scale=500,scale_units='xy', color='black')

        #plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/theta_pv_cross_section_{formatted_datetime}.png')
        plt.close()

def thetae_pv_cross_section(start_point, end_point, ds_pl, directions, g, path):
    # Loop over the reanalysis time steps
    for i in range(0, len(ds_pl.time.values)):
        ds_pl_sliced = ds_pl.isel(time=i)

        # Slice the dataset to get the data for the region of interest
        ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))

        # Get the u, v, t, and q fields at multiple levels
        u_sliced = ds_pl_sliced['U'].sel(level=slice(150, 1000)) # units: m/s
        v_sliced = ds_pl_sliced['V'].sel(level=slice(150, 1000)) # units: m/s
        t_sliced = ds_pl_sliced['T'].sel(level=slice(150, 1000)) # units: K
        q_sliced = ds_pl_sliced['Q'].sel(level=slice(150, 1000)) # units: kg/kg
        pressure_levels = t_sliced['level'] # units: 
        pressure_levels_ivt = u_sliced.level[::-1] * 100 # units: Pa

        # Calculate dewpoint and equivalent potential temperature
        td = mpcalc.dewpoint_from_specific_humidity(pressure_levels, t_sliced, q_sliced) # units: K
        thetae = mpcalc.equivalent_potential_temperature(pressure_levels, t_sliced, td) # units: K

        # Calculate frontogenesis
        #fgen = mpcalc.frontogenesis(theta, u_sliced, v_sliced, pressure_levels) # units: K / m / s
        #fgen_3hr = fgen * 1.08e9 # units: K / 100km / hr

        # Add the potential temperature (theta) to the dataset
        thetae_da = xr.DataArray(thetae, dims=['level', 'latitude', 'longitude'],
                                coords={'level': t_sliced['level'], 
                                        'latitude': t_sliced['latitude'], 
                                        'longitude': t_sliced['longitude']},
                                attrs={'units': 'K'})

        ds_pl_sliced['THETAE'] = thetae

        # Get the lats and lons
        lats = u_sliced['latitude'][:]
        lons = u_sliced['longitude'][:] 

        # Calculate the integrated vapor transport (IVT) using the u- and v-wind components and the specific humidity
        u_ivt = -1 / g * np.trapz(u_sliced * q_sliced, pressure_levels_ivt, axis=0)
        v_ivt = -1 / g * np.trapz(v_sliced * q_sliced, pressure_levels_ivt, axis=0)

        # Calculate the IVT magnitude
        ivt = np.sqrt(u_ivt**2 + v_ivt**2)

        # Create an xarray DataArray for the IVT
        ivt_da = xr.DataArray(ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']})

        # Define the color levels and colors for the IVT plot
        ivt_levels = [250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800]
        ivt_colors = ['#ffff00', '#ffe400', '#ffc800', '#ffad00', '#ff8200', '#ff5000', '#ff1e00', '#eb0010', '#b8003a', '#850063', '#570088']
        ivt_cmap = mcolors.ListedColormap(ivt_colors)
        ivt_norm = mcolors.BoundaryNorm(ivt_levels, ivt_cmap.N)

        # Mask the IVT values below 250 kg/m/s and create a filtered DataArray for the u- and v-wind components
        mask = ivt_da >= 250
        u_ivt_filtered = xr.DataArray(u_ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']}).where(mask, drop=True)
        v_ivt_filtered = xr.DataArray(v_ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']}).where(mask, drop=True)

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_pl_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Now do the cross-section interpolation
        lat_values = np.linspace(start_point[0], end_point[0])
        lon_values = np.linspace(start_point[1], end_point[1])
        x = xr.DataArray(lon_values, dims='Lat_Lon')
        y = xr.DataArray(lat_values, dims='Lat_Lon')
        
        # Interpolate the sliced data for the cross-section
        ds_cross = ds_pl_sliced.interp(longitude=x, latitude=y, method='nearest')

        # Get the temperature, potential vorticity, and theta 
        pv_crossed = ds_cross['PV'].sel(level=slice(150, 1000)) # units: K
        thetae_crossed = ds_cross['THETAE'].sel(level=slice(150, 1000)) # units: K

        # Get the pressure levels
        pressure_levels = pv_crossed['level'] # units: hPa

        # Define the color levels and colors for the potential vorticity
        levels = np.arange(0, 3.26, 0.25)
        colors = ['white', '#d1e9f7', '#a5cdec', '#79a3d5', '#69999b', '#78af58', '#b0cc58', '#f0d95f', '#de903e', '#cb5428', '#b6282a', '#9b1622', '#7a1419']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Smooth the PV and potential temperature 
        pv_smoothed = gaussian_filter(pv_crossed, sigma=1)
        thetae_smoothed = gaussian_filter(thetae_crossed, sigma=1)

        # Create the figure 
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the isentropes and potential vorticity 
        isentropes = ax.contour(ds_cross['longitude'], pressure_levels, thetae_smoothed, colors='black', levels=np.arange(250, 450, 1))
        ax.clabel(isentropes, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        #fgen_c = ax.contour(ds_cross['longitude'], pressure_levels, fgen_3hr, colors='black', levels=np.arange(-10, 10, 1), linestyles='dashed')
        #ax.clabel(fgen_c, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        pv_cf = ax.contourf(ds_cross['longitude'], pressure_levels, pv_smoothed * 1e6, cmap=cmap, levels=levels, norm=norm, extend='max')
        plt.colorbar(pv_cf, orientation='vertical', label='PV (m$^2$ s$^{-1}$ K kg$^{-1}$)', fraction=0.046, pad=0.04)


        # Adding the isentropes to the legend with a custom label
        thetae_line = plt.Line2D([0], [0], color='black', linewidth=1, label=r'$\theta_e$: Equivalent Potential Temperature (K)')
        #fgen_line = plt.Line2D([0], [0], color='black', linewidth=1, linestyle='dashed', label='Frontogenesis (K / 100km / hr)')

        # Creating the legend with the custom entries
        ax.legend(handles=[thetae_line], loc='upper right')
        ax.set_yticks(pressure_levels)
        ax.set_yticklabels(list(pressure_levels.values))
        plt.ylim(1000, 600)
        plt.xlabel('Longitude (degrees E)')
        plt.ylabel('Pressure (hPa)')
        plt.title(f'ERA5 Reanalysis Vertical Cross-Section of PV and $\\theta_e$ | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}')

        # Manually set y-axis tick marks
        custom_ticks = [1000, 900, 800, 700, 600]  
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels([str(tick) for tick in custom_ticks])

        # Add labels "A" and "A'" to the bottom left and right
        ax.text(0, -0.1, 'A', transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
        ax.text(1, -0.1, "A'", transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

        # Add an inset of the plan view to provide additional context 
        ax_inset = fig.add_axes([0.10, 0.63, 0.25, 0.25], projection=ccrs.PlateCarree())
        ax_inset.set_extent([directions['East'], directions['West'], directions['South'], directions['North']])
        ax_inset.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        ax_inset.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax_inset.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax_inset.add_feature(cfeature.OCEAN, color='white')
        ax_inset.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_inset.add_feature(cfeature.LAND, color='#fbf5e9')
        ax_inset.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]],
                            color="black", marker="o", transform=ccrs.PlateCarree())
        # Add text annotations
        ax_inset.text(start_point[1], start_point[0], "A", transform=ccrs.PlateCarree(),
                    verticalalignment='bottom', horizontalalignment='right', fontweight='bold', fontsize=12)
        ax_inset.text(end_point[1], end_point[0], "A'", transform=ccrs.PlateCarree(),
                    verticalalignment='bottom', horizontalalignment='right', fontweight='bold', fontsize=12)

        #isohypses = ax_inset.contour(z_500['longitude'], z_500['latitude'], z_500, colors='black', levels=np.arange(5000, 6000, 50), linewidths=1)
        #isobars = ax_inset.contour(mslp['longitude'], mslp['latitude'], gaussian_filter(mslp, sigma=1), colors='black', levels=np.arange(960, 1040, 4), linewidths=0.25)
        #plt.clabel(isobars, inline=True, inline_spacing=5, fontsize=6, fmt='%i')
        c = ax_inset.contour(ivt_da['longitude'], ivt_da['latitude'], gaussian_filter(ivt_da, sigma=1), colors='black', levels=ivt_levels, linewidths=0.5)
        cf = ax_inset.contourf(ivt_da['longitude'], ivt_da['latitude'], gaussian_filter(ivt_da, sigma=1), cmap=ivt_cmap, levels=ivt_levels, norm=ivt_norm, extend='max')

        # Plot the IVT vectors
        step = 5 
        plt.quiver(u_ivt_filtered['longitude'][::step], u_ivt_filtered['latitude'][::step], u_ivt_filtered[::step, ::step], v_ivt_filtered[::step, ::step], scale=500,scale_units='xy', color='black')

        #plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/thetae_pv_cross_section_{formatted_datetime}.png')
        plt.close()
        #plt.show()

def plot_ivt(g, ds_pl, ds_sfc, directions, path):
    # Loop over the reanlysis time steps
    for i in range(0, len(ds_pl.time)):
        # Slice the dataset to get the data for the current time step
        ds_sliced = ds_pl.isel(time=i)
        ds_sfc_sliced = ds_sfc.isel(time=i)
        
        # Slice the dataset to get the data for the region of interest
        ds_sliced = ds_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
        ds_sfc_sliced = ds_sfc_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))

        # Slice the dataset to get the data for the pressure levels between 500 and 1000 hPa
        u_sliced = ds_sliced['U'].sel(level=slice(500, 1000)) # units: m/s
        v_sliced = ds_sliced['V'].sel(level=slice(500, 1000)) # units: m/s
        q_sliced = ds_sliced['Q'].sel(level=slice(500, 1000)) # units: kg/kg

        mslp = ds_sfc_sliced['MSL'] / 100 # units: hPa

        # Flip the order of the pressure levels and convert them to Pa from hPa
        pressure_levels = u_sliced.level[::-1] * 100 # units: Pa

        # Calculate the integrated vapor transport (IVT) using the u- and v-wind components and the specific humidity
        u_ivt = -1 / g * np.trapz(u_sliced * q_sliced, pressure_levels, axis=0)
        v_ivt = -1 / g * np.trapz(v_sliced * q_sliced, pressure_levels, axis=0)

        # Calculate the IVT magnitude
        ivt = np.sqrt(u_ivt**2 + v_ivt**2)

        # Create an xarray DataArray for the IVT
        ivt_da = xr.DataArray(ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']})

        # Define the color levels and colors for the IVT plot
        levels = [250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800]
        colors = ['#ffff00', '#ffe400', '#ffc800', '#ffad00', '#ff8200', '#ff5000', '#ff1e00', '#eb0010', '#b8003a', '#850063', '#570088']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the IVT
        c = plt.contour(ivt_da['longitude'], ivt_da['latitude'], gaussian_filter(ivt_da, sigma=1), colors='black', levels=levels, linewidths=0.5)
        cf = plt.contourf(ivt_da['longitude'], ivt_da['latitude'], gaussian_filter(ivt_da, sigma=1), cmap=cmap, levels=levels, norm=norm, extend='max')
        plt.colorbar(cf, orientation='vertical', label='IVT (kg/m/s)', fraction=0.046, pad=0.04)

        # Mask the IVT values below 250 kg/m/s and create a filtered DataArray for the u- and v-wind components
        mask = ivt_da >= 250
        u_ivt_filtered = xr.DataArray(u_ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']}).where(mask, drop=True)
        v_ivt_filtered = xr.DataArray(v_ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']}).where(mask, drop=True)

        # Plot the IVT vectors
        step = 5 
        plt.quiver(u_ivt_filtered['longitude'][::step], u_ivt_filtered['latitude'][::step], u_ivt_filtered[::step, ::step], v_ivt_filtered[::step, ::step], scale=500,scale_units='xy', color='black')
        
        isobars = plt.contour(mslp['longitude'], mslp['latitude'], gaussian_filter(mslp, sigma=1), colors='black', levels=np.arange(960, 1080, 4), linewidths=0.5)
        try:
            plt.clabel(isobars, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label for isobars.")

        # Adding custom legend entries (hardcoded)
        isobars_line = plt.Line2D([0], [0], color='black', linewidth=1, label='MSLP (hPa)')

        # Creating the legend with the custom entries
        ax.legend(handles=[isobars_line], loc='upper right')

        # Add the title, set the map extent, and add map features
        plt.title(f'ERA5 Reanalysis Integrated Water Vapor Transport (IVT) | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.tight_layout()

        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/ivt_{formatted_datetime}.png')
        plt.close()

def plot_iwv(g, ds_pl, directions, path):
    # Loop over the reanalysis time steps
    for i in range(0, len(ds_pl.time)):
        # Slice the dataset to get the data for the current time step
        ds_sliced = ds_pl.isel(time=i)

        # Slice the dataset to get the data for the region of interest
        ds_sliced = ds_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
        
        # Slice the dataset to get the data for the pressure levels between 500 and 1000 hPa, or specifically at 850 hPa
        u_sliced = ds_sliced['U'].sel(level=slice(500, 1000)) # units: m/s
        u_850 = ds_sliced['U'].sel(level=850) * 1.94384 # units: knots
        v_sliced = ds_sliced['V'].sel(level=slice(500, 1000)) # units: m/s
        v_850 = ds_sliced['V'].sel(level=850) * 1.94384 # units: knots
        q_sliced = ds_sliced['Q'].sel(level=slice(500, 1000)) # units: kg/kg
        z_sliced = ds_sliced['Z'].sel(level=850) # units: m
        pressure_levels = u_sliced.level[::-1] * 100 # units: Pa

        # Calculate the integrated water vapor (IWV) using the specific humidity and put it in an xarray DataArray
        iwv = -1 / g * np.trapz(q_sliced, pressure_levels, axis=0)
        iwv_da = xr.DataArray(iwv, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']})

        # Define the color levels and colors for the IWV plot
        levels = np.arange(20, 61, 2)
        colors = ['#1a2dd3', '#1a43ff', '#2486ff', '#31ccff', '#3cfbf0', '#37e5aa', '#32ce63', '#33be21', '#76d31c', '#bae814', '#fffc02',
                    '#ffe100', '#fec600', '#fdab00', '#fc7800', '#fc4100', '#fc0000','#d2002f', '#a31060', '#711e8b', '#8a51af']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Create the plot 
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the geopotential heights and IWV
        isohypses = plt.contour(z_sliced['longitude'], z_sliced['latitude'], gaussian_filter(z_sliced, sigma=1), colors='black')
        try:
            plt.clabel(isohypses, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label for geopotential heights.")
        c = plt.contour(iwv_da['longitude'], iwv_da['latitude'], gaussian_filter(iwv_da, sigma=1), colors='black', levels=levels, linewidths=0.5)
        cf = plt.contourf(iwv_da['longitude'], iwv_da['latitude'], gaussian_filter(iwv_da, sigma=1), cmap=cmap, levels=levels, norm=norm, extend='max')
        plt.colorbar(cf, orientation='vertical', label='IWV (mm)', fraction=0.046, pad=0.04)

        # Plot the wind barbs for the u- and v-wind components at 850 hPa
        step = 10 
        ax.barbs(u_850['longitude'][::step], u_850['latitude'][::step], u_850[::step, ::step], v_850[::step, ::step], length=6, color='black')
        
        # Adding custom legend entries (hardcoded)
        isohypses_line = plt.Line2D([0], [0], color='black', linewidth=1, label='Geopotential Heights (m)')

        # Creating the legend with the custom entries
        ax.legend(handles=[isohypses_line], loc='upper right')


        # Add the title, set the map extent, and add map features
        plt.title(f'IWV and 850-hPa Geopotential Heights and Winds | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/iwv_{formatted_datetime}.png')
        plt.close()

def plot_q_uv_zeta(g, ds_pl, directions, path):
    # Loop over the reanlysis time steps
        for i in range(0, len(ds_pl.time)):
            # Slice the dataset to get the data for the current time step
            ds_pl_sliced = ds_pl.isel(time=i)
            
            # Slice the dataset to get the data for the region of interest
            ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))

            # Slice the dataset to get the data for the pressure levels at 850 hPa
            t_sliced = ds_pl_sliced['T'].sel(level=850) # units: K
            u_sliced = ds_pl_sliced['U'].sel(level=850) # units: m/s
            v_sliced = ds_pl_sliced['V'].sel(level=850) # units: m/s
            q_sliced = ds_pl_sliced['Q'].sel(level=850) * 1000 # units: g/kg

            # Calculate the potential temperature and relative vorticity
            theta_sliced = mpcalc.potential_temperature(850 * units.hPa, t_sliced) # units: K
            zeta_sliced = mpcalc.vorticity(u_sliced, v_sliced) # units: 10^-5 s^-1

            # Get the time of the current time step and create a pandas DatetimeIndex
            time = ds_pl_sliced.time.values
            int_datetime_index = pd.DatetimeIndex([time])

            # Define the color levels and colors for the specific humidity
            levels = np.arange(4, 15, 1)
            colors = ['#c3e8fa', '#8bc5e9', '#5195cf', '#49a283', '#6cc04b', '#d8de5a', '#f8b348', '#f46328', '#dc352b', '#bb1b24', '#911618']
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(levels, cmap.N)

            # Smooth the specific humidity and potential temperature
            q_smoothed = gaussian_filter(q_sliced, sigma=1)
            theta_smoothed = gaussian_filter(theta_sliced, sigma=1)
            zeta_smoothed = gaussian_filter(zeta_sliced, sigma=1)

            # Create the figure
            fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

            # Plot the specific humidity, potential temperature, and wind barbs
            plt.contour(zeta_sliced['longitude'], zeta_sliced['latitude'], zeta_smoothed, colors='purple', levels=np.arange(10 * 10**-5, 30 * 10**-5, 10**-5), linewidths=0.5, label='$\\zeta$')
            plt.contour(theta_sliced['longitude'], theta_sliced['latitude'], theta_smoothed, colors='black', levels=np.arange(220, 340, 1), linewidths=0.5, label='$\\theta$')
            cf = plt.contourf(q_sliced['longitude'], q_sliced['latitude'], q_smoothed, cmap=cmap, levels=levels, norm=norm, extend='max')
            plt.colorbar(cf, orientation='vertical', label='Specific Humidity (g kg$^{-1}$)', fraction=0.046, pad=0.04)

            # Plot the 850-hPa wind barbs
            step = 10
            ax.barbs(u_sliced['longitude'][::step], u_sliced['latitude'][::step], u_sliced[::step, ::step], v_sliced[::step, ::step], length=6, color='black')

            # Adding custom legend entries (hardcoded)
            zeta_line = plt.Line2D([0], [0], color='purple', linewidth=1, label=r'$\zeta$ Relative Vorticity (s$^{-1}$)')
            theta_line = plt.Line2D([0], [0], color='black', linewidth=1, label=r'$\theta$ Potential Temperature (K)')

            # Creating the legend with the custom entries
            ax.legend(handles=[zeta_line, theta_line], loc='upper right')

            # Add the title, set the map extent, and add map features
            plt.title(f'ERA5 Reanalysis 850-hPa Specific Humidity, $\\theta$, $\\zeta$, and Wind Barbs | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
            ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-5])
            ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, color='white')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='#fbf5e9')

            # Add gridlines and format longitude/latitude labels
            gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
            gls.top_labels = False
            gls.right_labels = False
            ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
            ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter()
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)

            plt.tight_layout()
            formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
            plt.savefig(f'{path}/specific_humidity_{formatted_datetime}.png')
            plt.close()

def plot_pressure_pert(ds_sfc, directions, path):
    # Loop over the reanlysis time steps
    for i in range(0, len(ds_sfc.time)):
        # Slice the dataset to get the data for the current time step
        ds_sfc_sliced = ds_sfc.isel(time=i)
        
        # Slice the dataset to get the data for the region of interest
        ds_sfc_sliced = ds_sfc_sliced.sel(latitude=slice(directions['North']+5, directions['South']-5), longitude=slice(directions['West']-5, directions['East']+5))

        # Sice the dataset to get the MSL
        mslp = ds_sfc_sliced['MSL'] / 100 # units: hPa

        # Calculate pressure perturbation
        p_pert = mslp - mslp.mean()

            # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_sfc_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot 
        cf = plt.contourf(p_pert['longitude'], p_pert['latitude'], p_pert, cmap='PiYG', levels=np.arange(-20, 20, 1), extend='both')
        plt.colorbar(cf, orientation='vertical', label='hPa', fraction=0.046, pad=0.04)

        isobars = plt.contour(mslp['longitude'], mslp['latitude'], gaussian_filter(mslp, sigma=1), colors='black', levels=np.arange(960, 1080, 4), linewidths=1)
        try:
            plt.clabel(isobars, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label for isobars.")

        # Add the title, set the map extent, and add map features
        plt.title(f'ERA5 Reanalysis MSLP Perturbation and Isobars (hPa) | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='white')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')

        # Add gridlines and format longitude/latitude labels
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/p_{formatted_datetime}.png')
        plt.close()

def plot_250_isotachs_ageo(ds_pl, directions, g, path):
    # Loop over the reanalysis time steps
    for i in range(0, len(ds_pl.time.values)):
        # Slice the dataset to get the data for the current time step
        ds_pl_sliced = ds_pl.isel(time=i)

        # Slice the dataset to get the data for the region of interest
        ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
        
        # Slice the dataset to get the data for the pressure level at 250 hPa
        u_250 = ds_pl_sliced['U'].sel(level=250) # units: m/s
        v_250 = ds_pl_sliced['V'].sel(level=250) # units: m/s
        z_250 = ds_pl_sliced['Z'].sel(level=250) / g * units.meters # units: m
        pressure_levels = u_250.level * 100 # units: Pa

        # Calculate the ageostrophic wind
        u_ageo, v_ageo = mpcalc.ageostrophic_wind(z_250, u_250, v_250)

        # Calculate the wind speed at 250 hPa
        wind_speed_250 = np.sqrt(u_250**2 + v_250**2)
        ageo_wind_speed = mpcalc.wind_speed(u_ageo, v_ageo)
        #ageo_wind_speed = np.sqrt(u_ageo**2 + v_ageo**2)
        # Calculate the divergence
        div = mpcalc.divergence(u_250, v_250) * 1e5

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_pl_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Smooth the mslp and wind speed
        z_smoothed = gaussian_filter(z_250, sigma=3)
        wnd_smoothed = gaussian_filter(wind_speed_250, sigma=3)

        # Define the color levels and colors for the isotachs
        levels = np.arange(20, 95, 5)
        colors = ['#daedfb', '#b7dcf6', '#91bae4', '#7099ce', '#6a999d', '#72ad63', '#77c14a', '#cad955', '#f8cf4f', '#f7953c', '#ef5f28', '#e13e26', '#cd1e28', '#b1181e', '#901617']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Create the plot 
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the geopotential heights and isotachs
        isohypses = plt.contour(z_250['longitude'], z_250['latitude'], z_smoothed, colors='black', levels=np.arange(8700, 11820, 60), linewidths=1)
        try:
            ax.clabel(isohypses, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label.")
        #div_contours = plt.contour(ds_pl_sliced['longitude'], ds_pl_sliced['latitude'], div, colors='black', levels=np.arange(-24, 24, 1), linestyles='dashed')
        pos_div = plt.contour(ds_pl_sliced['longitude'], ds_pl_sliced['latitude'], div, colors='magenta', levels=np.arange(8, 24, 4), linestyles='solid')
        try:
            ax.clabel(pos_div, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label.")
        cf = plt.contourf(u_250['longitude'], u_250['latitude'], wnd_smoothed, cmap=cmap, norm=norm, levels=levels, extend='max')
        plt.colorbar(cf, orientation='vertical', label='Wind Speed (ms$^{-1}$)', fraction=0.046, pad=0.04)

        mask = ageo_wind_speed >= 2.5 * units.meters / units.second

        # Create DataArrays for u and v wind components with the appropriate dimensions and coordinates
        u_ageo_da = xr.DataArray(u_ageo, dims=['latitude', 'longitude'], coords={'latitude': u_250['latitude'], 'longitude': u_250['longitude']})
        v_ageo_da = xr.DataArray(v_ageo, dims=['latitude', 'longitude'], coords={'latitude': v_250['latitude'], 'longitude': v_250['longitude']})

        # Filter the DataArrays using the mask
        u_ageo_filtered = u_ageo_da.where(mask, drop=True)
        v_ageo_filtered = v_ageo_da.where(mask, drop=True)

        step = 10
        ax.barbs(u_ageo_filtered['longitude'][::step], u_ageo_filtered['latitude'][::step], u_ageo_filtered[::step, ::step], v_ageo_filtered[::step, ::step], length=6, color='black')
        
        # Plot the IVT vectors
        #step = 5 
        #plt.quiver(u_ivt_filtered['longitude'][::step], u_ivt_filtered['latitude'][::step], u_ivt_filtered[::step, ::step], v_ivt_filtered[::step, ::step], scale=500,scale_units='xy', color='black')

        # Adding custom legend entries (hardcoded)
        isohypses_line = plt.Line2D([0], [0], color='black', linewidth=1, label='Geopotential Height (m)')
        div_line = plt.Line2D([0], [0], color='magenta', linewidth=1, linestyle='solid', label='Divergence (10$^{-5}$ s$^{-1}$)')


        # Creating the legend with the custom entries
        ax.legend(handles=[isohypses_line, div_line], loc='upper right')

        # Add the title, set the map extent, and add map features
        plt.title(f'250-hPa Isoatachs, Ageostrophic Wind (m/s), and Divergence | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North'] - 5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.savefig(f'{path}/ageo_{formatted_datetime}.png')
        plt.close()

def plot_250_isotachs_ageo_stream(ds_pl, directions, g, path):
    # Loop over the reanalysis time steps
    for i in range(0, len(ds_pl.time.values)):
        # Slice the dataset to get the data for the current time step
        ds_pl_sliced = ds_pl.isel(time=i)

        # Slice the dataset to get the data for the region of interest
        ds_pl_sliced = ds_pl_sliced.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
        
        # Slice the dataset to get the data for the pressure level at 250 hPa
        u_250 = ds_pl_sliced['U'].sel(level=250) # units: m/s
        v_250 = ds_pl_sliced['V'].sel(level=250) # units: m/s
        z_250 = ds_pl_sliced['Z'].sel(level=250) / g * units.meters # units: m
        pressure_levels = u_250.level * 100 # units: Pa

        # Calculate the ageostrophic wind
        u_ageo, v_ageo = mpcalc.ageostrophic_wind(z_250, u_250, v_250)

        # Calculate the wind speed at 250 hPa
        wind_speed_250 = np.sqrt(u_250**2 + v_250**2)
        ageo_wind_speed = mpcalc.wind_speed(u_ageo, v_ageo)
        #ageo_wind_speed = np.sqrt(u_ageo**2 + v_ageo**2)
        # Calculate the divergence
        div = mpcalc.divergence(u_250, v_250) * 1e5

        # Get the time of the current time step and create a pandas DatetimeIndex
        time = ds_pl_sliced.time.values
        int_datetime_index = pd.DatetimeIndex([time])

        # Smooth the mslp and wind speed
        z_smoothed = gaussian_filter(z_250, sigma=3)
        wnd_smoothed = gaussian_filter(wind_speed_250, sigma=3)

        # Define the color levels and colors for the isotachs
        levels = np.arange(20, 95, 5)
        colors = ['#daedfb', '#b7dcf6', '#91bae4', '#7099ce', '#6a999d', '#72ad63', '#77c14a', '#cad955', '#f8cf4f', '#f7953c', '#ef5f28', '#e13e26', '#cd1e28', '#b1181e', '#901617']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Create the plot 
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the geopotential heights and isotachs
        isohypses = plt.contour(z_250['longitude'], z_250['latitude'], z_smoothed, colors='black', levels=np.arange(8700, 11820, 60), linewidths=1)
        try:
            ax.clabel(isohypses, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label.")
        #div_contours = plt.contour(ds_pl_sliced['longitude'], ds_pl_sliced['latitude'], div, colors='black', levels=np.arange(-24, 24, 1), linestyles='dashed')
        pos_div = plt.contour(ds_pl_sliced['longitude'], ds_pl_sliced['latitude'], div, colors='magenta', levels=np.arange(8, 24, 4), linestyles='solid')
        try:
            ax.clabel(pos_div, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label.")
        cf = plt.contourf(u_250['longitude'], u_250['latitude'], wnd_smoothed, cmap=cmap, norm=norm, levels=levels, extend='max')
        plt.colorbar(cf, orientation='vertical', label='Wind Speed (ms$^{-1}$)', fraction=0.046, pad=0.04)

        mask = ageo_wind_speed >= 2.5 * units.meters / units.second

        # Create DataArrays for u and v wind components with the appropriate dimensions and coordinates
        u_ageo_da = xr.DataArray(u_ageo, dims=['latitude', 'longitude'], coords={'latitude': u_250['latitude'], 'longitude': u_250['longitude']})
        v_ageo_da = xr.DataArray(v_ageo, dims=['latitude', 'longitude'], coords={'latitude': v_250['latitude'], 'longitude': v_250['longitude']})

        # Filter the DataArrays using the mask
        u_ageo_filtered = u_ageo_da.where(mask, drop=True)
        v_ageo_filtered = v_ageo_da.where(mask, drop=True)

        lon = u_ageo_filtered['longitude'].values
        lat = u_ageo_filtered['latitude'].values

        # Ensure that the grid dimensions are correct
        u_values = u_ageo_filtered.values  # u wind component
        v_values = v_ageo_filtered.values  # v wind component

        # Create a meshgrid for streamplot
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Create the figure and axis
        fig, ax = plt.subplots()

        # Define the step for reducing the density of the streamlines
        step = 10

        # Create streamlines using the filtered wind components
        strm = ax.streamplot(
            lon_grid[::step, ::step], 
            lat_grid[::step, ::step],
            u_values[::step, ::step], 
            v_values[::step, ::step], 
            color='black', 
            linewidth=1, 
            density=2,  # Adjust this for more or less density of streamlines
        )
        
        # Plot the IVT vectors
        #step = 5 
        #plt.quiver(u_ivt_filtered['longitude'][::step], u_ivt_filtered['latitude'][::step], u_ivt_filtered[::step, ::step], v_ivt_filtered[::step, ::step], scale=500,scale_units='xy', color='black')

        # Adding custom legend entries (hardcoded)
        isohypses_line = plt.Line2D([0], [0], color='black', linewidth=1, label='Geopotential Height (m)')
        div_line = plt.Line2D([0], [0], color='magenta', linewidth=1, linestyle='solid', label='Divergence (10$^{-5}$ s$^{-1}$)')


        # Creating the legend with the custom entries
        ax.legend(handles=[isohypses_line, div_line], loc='upper right')

        # Add the title, set the map extent, and add map features
        plt.title(f'250-hPa Isoatachs, Ageostrophic Wind (m/s), and Divergence | {int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC")}', fontsize=14, weight='bold')
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North'] - 5])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.tight_layout()
        formatted_datetime = int_datetime_index[0].strftime("%Y-%m-%d %H00 UTC").replace(':', '-')
        plt.show()
        #plt.savefig(f'{path}/ageo_{formatted_datetime}.png')
        #plt.close()