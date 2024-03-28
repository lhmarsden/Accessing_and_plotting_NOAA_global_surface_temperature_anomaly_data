# Accessing and plotting NOAA global surface temperature anomaly data

Python code to access and plot the NOAA Global Surface Temperature Dataset (NOAAGlobalTemp), Version 5.0.

## Contents

* temperature_anomaly_map.py - Plot the data on a map for one time slice
* temperature_anomalies_animation.py - Create a GIF animation of the entire time series of data

## Reference

I would like to thank the authors of the data for making them openly available and FAIR.

If you want to use the data publically, you should also give credit to the authors of the dataset by including the following recommended citation (change the access data at the end):

H.-M. Zhang, B. Huang, J. H. Lawrimore, M. J. Menne, and T. M. Smith (2019): NOAA Global Surface Temperature Dataset (NOAAGlobalTemp), Version 5.0 [indicate subset used]. NOAA National Centers for Environmental Information. doi:10.25921/9qth-2p70 [2024-03-28].

## Where are the data

Landing page for the data:
https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01585

OPeNDAP link to the data:
https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202212_c20230108T133308.nc

Append '.html' to the url above to access the OPeNDAP data access form.

To find the latest version of the data, go to:
https://www.ncei.noaa.gov/thredds/catalog/noaa-global-temp-v5/latest.html

