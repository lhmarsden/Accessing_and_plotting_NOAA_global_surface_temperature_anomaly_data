import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

url = 'https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202212_c20230108T133308.nc'
xrds = xr.open_dataset(url)

desired_date = '2022-11-01'
projection = ccrs.Mollweide()
transform = ccrs.PlateCarree()

fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=projection)

# vmin = xrds['anom'].attrs['valid_min']
# vmax = xrds['anom'].attrs['valid_max'] # Use the metadata to help if the data includes invalid values
vmin = xrds['anom'].min()
vmax = xrds['anom'].max()
abs_max = max(abs(vmin), vmax)

data_for_desired_date = xrds.sel(time=desired_date, method='nearest')
data_for_desired_date['anom'].plot(ax=ax, transform=transform, vmin=-abs_max, vmax=abs_max, cmap='seismic')

# Get the extent of the data
xmin, xmax, ymin, ymax = ax.get_extent(transform)

# Create a patch to fill the areas outside the data extent with grey
ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        facecolor='grey', transform=transform, zorder=-1))

ax.coastlines()
ax.set_global()

plt.savefig(f'global_surface_temperature_anomalies_{desired_date}.png', transparent=True)
plt.show()