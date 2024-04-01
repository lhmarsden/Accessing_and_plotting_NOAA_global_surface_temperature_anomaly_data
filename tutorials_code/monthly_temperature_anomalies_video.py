import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.animation as animation
import numpy as np

url = 'https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202212_c20230108T133308.nc'
xrds = xr.open_dataset(url)

# Filter only dates since 1970-01-01
xrds = xrds.sel(time=slice('1970-01-01', None))

projection = ccrs.Mollweide()
transform = ccrs.PlateCarree()

# vmin = xrds['anom'].attrs['valid_min']
# vmax = xrds['anom'].attrs['valid_max'] # Use the metadata to help if the data includes invalid values
vmin = xrds['anom'].min()
vmax = xrds['anom'].max()
abs_max = max(abs(vmin), vmax)

dates = xrds['time'].values
# Select only dates in February
dates = [date for date in dates if np.datetime64(date, 'M').astype(int) % 12 + 1 == 2]

def plot_axis(data_for_desired_date):
    ax = plt.axes(projection=projection)

    data_for_desired_date['anom'].plot(ax=ax, transform=transform, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    ax.coastlines()

    # Get the extent of the data
    xmin, xmax, ymin, ymax = ax.get_extent(transform)

    # Create a patch to fill the areas outside the data extent with grey
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            facecolor='grey', transform=transform, zorder=-1))

    ax.coastlines()
    ax.set_global()

    return ax

# Function to update the plot for each frame of the animation
def update(frame):
    fig.clear()
    desired_date = dates[frame]
    data_for_desired_date = xrds.sel(time=desired_date)

    ax = plot_axis(data_for_desired_date)

    return ax

fig = plt.figure(figsize=(16, 8))
desired_date = dates[0]
data_for_desired_date = xrds.sel(time=desired_date)

ax = plot_axis(data_for_desired_date)

# Save the animation as an MP4 video
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(dates))
writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Luke Marsden'), bitrate=1800)
ani.save('global_surface_temperature_anomalies_animation.mp4', writer=writer)

plt.show()