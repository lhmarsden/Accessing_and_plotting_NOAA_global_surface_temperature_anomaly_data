import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.animation as animation
import numpy as np

url = 'https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202212_c20230108T133308.nc'
xrds = xr.open_dataset(url)

# Select one year of data and then find the mean annual temperature anomalies
year = 2019
data_for_desired_year = xrds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
mean_annual_anom = data_for_desired_year['anom'].mean(dim='time')

fig = plt.figure(figsize=(16, 8))

projection = ccrs.Mollweide()
transform = ccrs.PlateCarree()
ax = plt.axes(projection=projection)

vmin = -3
vmax = 3
pcm = mean_annual_anom.plot(ax=ax, transform=transform, vmin=vmin, vmax=vmax, cmap='seismic', add_colorbar=False)
ax.coastlines()

# Get the extent of the data
xmin, xmax, ymin, ymax = ax.get_extent(transform)

# Create a patch to fill the areas outside the data extent with grey
ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        facecolor='grey', transform=transform, zorder=-1))

ax.coastlines()
ax.set_global()

cbar = plt.colorbar(pcm, ax=ax, orientation='vertical')
cbar.set_label('Anomaly (°C)')
ax.set_title(f'{year}')

plt.savefig('global_surface_temperature_anomalies_{year}.png', transparent=True)
plt.show()