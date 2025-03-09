import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.animation as animation
import numpy as np
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from siphon.catalog import TDSCatalog
from datetime import datetime

annotation_text = 'Temperature anomalies relative\nto 1970–2000 climatology'

catalog_url = 'https://www.ncei.noaa.gov/thredds/catalog/noaa-global-temp-v6/latest.xml'
catalog = TDSCatalog(catalog_url)
dataset = list(catalog.datasets.values())[0]
opendap_url = dataset.access_urls['OPENDAP']
print(opendap_url)

xrds = xr.open_dataset(opendap_url)
print('Data loaded')

projection = ccrs.NorthPolarStereo()
transform = ccrs.PlateCarree()

background_colour = 'black' #'#1B041F'
title_color = '#F2BB05'
line_color = 'red'#'#5ce1e6'
text_color = '#FFFBFC'
ax2_color = '#5ce1e6'#'#84A98C'#"#F95738"

dates = xrds['time'].values
start_date = '1980-01-01'
#start_date = '2020-01-01'
min_lat = 65

end_date = xrds['time'].max().values

xrds = xrds.sel(lat=slice(min_lat, None), time=slice(start_date, end_date))
vmax = xrds['anom'].max().values
vmin = -vmax
# Extract latitudes
latitudes = xrds['lat']
# Compute weights (cosine of latitude)
weights = np.cos(np.deg2rad(latitudes))
weights /= weights.sum()  # normalize

years = np.arange(start_date, end_date, dtype='datetime64[Y]')

# # Append 10 frames to the beginning
# for _ in range(10):
#     years = np.append([years[0]], years)

# Append extra frames for the last year
for _ in range(10):
    years = np.append(years, years[-1])

def add_annotation(fig):
    fig.text(
        0.05, 0.88, annotation_text,
        fontsize=32, color=text_color,
        ha='left', va='top',
        bbox=dict(facecolor=background_colour, edgecolor='none', alpha=0.8)
    )

def interpolate_data(ds, method):
    min_lat = ds.lat.min().item()  # Ensure min_lat is defined

    ds_90_to_270 = ds.sel(lon=slice(87.5, 272.5))
    ds_90_to_270_interp = ds_90_to_270.interp(
        lat=np.arange(min_lat, 90.5, 0.5),
        lon=np.arange(87.5, 272.5, 0.5),
        method=method,
        kwargs={"fill_value": "extrapolate"}  # Allow extrapolation
    )
    ds_90_to_270_interp = ds_90_to_270_interp.sel(lon=slice(90, 270))
    ds_90_to_neg90 = ds_90_to_270_interp.assign_coords(lon=(ds_90_to_270_interp.lon + 180) % 360 - 180)

    # Combine the interpolated parts
    ds_0_to_90 = ds.sel(lon=slice(0, 92.5))
    ds_270_to_360 = ds.sel(lon=slice(267.5, 360))
    ds_combined = xr.concat([ds_0_to_90, ds_270_to_360], dim='lon')
    ds_neg90_to_90 = ds_combined.assign_coords(lon=(ds_combined.lon + 180) % 360 - 180)
    ds_neg90_to_90_interp = ds_neg90_to_90.interp(
        lat=np.arange(min_lat, 90.5, 0.5),
        lon=np.arange(-92.5, 92.5, 0.5),
        method=method,
        kwargs={"fill_value": "extrapolate"}  # Allow extrapolation
    )
    ds_neg90_to_90_interp = ds_neg90_to_90_interp.sel(lon=slice(-90, 90))

    interpolated_ds = xr.concat([ds_neg90_to_90_interp, ds_90_to_neg90], dim='lon')
    interpolated_ds = interpolated_ds.sortby('lon')

    return interpolated_ds

# Compute arctic mean anomaly
def compute_arctic_mean_anomaly(xrds, years):
    yearly_means = []
    for year in years:
        data_for_year = xrds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        annual_mean_anomaly = data_for_year['anom'].mean(dim='time')
        weighted_mean_anomaly = (annual_mean_anomaly * weights).sum(dim='lat')
        arctic_mean_anomaly = weighted_mean_anomaly.mean(dim='lon')
        yearly_means.append(arctic_mean_anomaly.values)
    return np.array(yearly_means)

arctic_mean_anomalies = compute_arctic_mean_anomaly(xrds, years)

def plot_data(ds, years_to_current, arctic_mean_anomalies_to_current, year):
    # Clear figure to avoid overlap in animation
    fig.clear()

    interpolated_ds_linear = interpolate_data(ds, 'linear')
    interpolated_ds_nearest = interpolate_data(ds, 'linear')
    data_for_desired_year = interpolated_ds_linear.combine_first(interpolated_ds_nearest)
    #data_for_desired_year = interpolated_ds_nearest
    # Adjust the layout: 2 rows (map + time series), 1 column
    gs = fig.add_gridspec(7, 15)  # 7 rows, 15 columns

    # Subplots
    ax1 = fig.add_subplot(gs[:, 6:], projection=projection)
    ax2 = fig.add_subplot(gs[2:5, :5])

    # Plot: mean annual anomaly
    mean_annual_anom = data_for_desired_year['anom'].mean(dim='time')
    pcm = mean_annual_anom.plot(
        ax=ax1,
        transform=transform,
        vmin=vmin,
        vmax=vmax,
        cmap='seismic',
        add_colorbar=False
    )

    # Coastlines & optional features for clarity
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle='--')

    # Add gridlines
    ax1.gridlines(draw_labels=True, color='grey', linestyle='--', linewidth=0.5)

    # Create a patch to fill the areas outside the data extent with grey
    ax1.set_facecolor('none')
    ax1.spines['geo'].set_edgecolor(background_colour)
    ax1.set_extent([-180, 180, min_lat, 90], crs=transform)

    # Colour bar
    cbar = plt.colorbar(pcm, ax=ax1, orientation='vertical', pad=0.05, fraction=0.05)
    cbar.set_label('°C', fontsize=30, color=text_color, rotation=0, labelpad=50)  # Set the colour of the colour bar label to white
    # Set tick marks and label colour to white on the colour bar
    cbar.ax.tick_params(labelsize=24, colors=text_color)  # Set colour bar tick mark labels to white

    # Main title
    ax1.set_title(f'Surface temperature anomalies: {year}', fontsize=40, pad=10, color=title_color, fontweight='bold')

    # Plot arctic mean anomaly time series
    ax2.plot(
        years_to_current,
        arctic_mean_anomalies_to_current,
        color=line_color,
        linestyle='-',
        marker='o',
        markersize=7,
        linewidth=3
    )

    ax2.set_xlim(years[0], years[-1])
    ax2.set_xlim(datetime.strptime(f'{start_date}', '%Y-%m-%d'), years[-1])
    ax2.set_ylim(arctic_mean_anomalies.min() - 0.2, arctic_mean_anomalies.max() + 0.2)

    ax2.set_xlabel('Year', fontsize=26)
    ax2.set_ylabel(f'(°C)', fontsize=30)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.5))  # Set y-axis grid at 0.5°C intervals
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Ensure gridlines are visible

    ax2.set_facecolor('none')

    ax2.spines['bottom'].set_color(text_color)
    ax2.spines['left'].set_color(text_color)
    ax2.spines['right'].set_color(ax2_color)
    ax2.xaxis.label.set_color(text_color)
    ax2.yaxis.label.set_color(text_color)
    ax2.tick_params(axis='x', colors=text_color, labelsize=24)
    ax2.tick_params(axis='y', colors=text_color, labelsize=24)

    ax2.yaxis.label.set_rotation(0)  # Set rotation to 0 (horizontal)
    ax2.yaxis.label.set_horizontalalignment('right')  # Align to the right to avoid overlap
    ax2.yaxis.label.set_verticalalignment('center')  # Keep it centered vertically

    ax2.set_title('Mean anomaly across Arctic\n(latitude > 65°)', fontsize=40, pad=10, color=title_color, fontweight='bold')

    # Avoid overlapping subplots
    fig.tight_layout()

    add_annotation(fig)

    return ax1, ax2

# Animation function
def update(frame):
    year = years[frame]
    print(year)
    # Grab a range around 'year' if you prefer, or just that year
    data_for_desired_year = xrds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))

    years_to_current = years[:frame+1]
    arctic_mean_anomalies_to_current = arctic_mean_anomalies[:frame+1]

    ax1, ax2 = plot_data(data_for_desired_year, years_to_current, arctic_mean_anomalies_to_current, year)
    return ax1, ax2

# --- Figure and initial plot setup ---
# Make the figure large and with high DPI for a crisp video (16:9 ratio is common for YouTube)
fig = plt.figure(figsize=(32, 18), dpi=100)  # Bigger figure with higher DPI
fig.patch.set_facecolor(background_colour)

# Initial frame
year0 = years[0]
data_for_desired_year0 = xrds.sel(time=slice(f'{year0}-01-01', f'{year0}-12-31'))
ax1, ax2 = plot_data(
    data_for_desired_year0,
    [year0],
    [arctic_mean_anomalies[0]],
    year0
)

fps = 2
interval = 1000 / fps

# Animation
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(years),
    interval=interval,  # milliseconds between frames (adjust as needed)
    blit=False
)

writer = animation.FFMpegWriter(
    fps=fps,
    metadata=dict(artist='Luke Marsden'),
    bitrate=12000,  # High bitrate for 1080p
    extra_args=['-vcodec', 'libx264']  # Better compression
)

ani.save('annual_arctic_anomalies_and_mean_subplot.mp4', writer=writer)