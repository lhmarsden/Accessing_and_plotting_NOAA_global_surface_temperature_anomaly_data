import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.animation as animation
import numpy as np

url = 'https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202212_c20230108T133308.nc'
xrds = xr.open_dataset(url)
#xrds = xrds.sel(time=slice('2018-01-01', None))

projection = ccrs.Mollweide()
transform = ccrs.PlateCarree()

vmin = -3
vmax = 3

dates = xrds['time'].values
start_date = xrds['time'].min().values
end_date = xrds['time'].max().values
years = np.arange(start_date, end_date, dtype='datetime64[Y]')

# Append 10 frames to the end to show the last year for 10 frames
for _ in range(10):
    years = np.append(years,years[-1])

def plot_axis(data_for_desired_year):
    ax = plt.axes(projection=projection)

    mean_annual_anom = data_for_desired_year['anom'].mean(dim='time')
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

    return ax

# Function to update the plot for each frame of the animation
def update(frame):
    fig.clear()
    year = years[frame]
    data_for_desired_year = xrds.sel(time=slice(f'{year-2}-01-01', f'{year+2}-12-31'))

    ax = plot_axis(data_for_desired_year)
    ax.set_title(f'{year}')

    return ax

fig = plt.figure(figsize=(16, 8))
year = years[0]
data_for_desired_year = xrds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))

ax = plot_axis(data_for_desired_year)
ax.set_title(f'{year}')

# Save the animation as an MP4 video
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(years))
writer = animation.FFMpegWriter(fps=2, metadata=dict(artist='Luke Marsden'), bitrate=1800)
ani.save('global_surface_temperature_annual_anomalies_animation.mp4', writer=writer)

plt.show()