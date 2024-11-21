#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:02:20 2024

@author: dimitrisherrera
"""

import xarray as xr
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import cartopy
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import cartopy.feature as cf
import shapely.geometry as sgeom
import matplotlib.path as mpath
import datetime
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec


# =============================================================================
# Regrdding function
# =============================================================================
def regrid(ds, target_res):
    ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(21, 49.0, target_res)),
        "lon": (["lon"], np.arange(-127.0, -67.0, target_res)),})

    regridder = xe.Regridder(ds, ds_out, "nearest_s2d")
    ds_out = regridder(ds)
    
    return ds_out

# =============================================================================
# Cummulative density function plot function
# =============================================================================
def cdf(x):
   x_sorted = np.sort(x)[::1]
   bins = np.arange(len(x_sorted)) / ((len(x_sorted)))
   
   return x_sorted#, bins  
   
# =============================================================================
#  Quantile mapping bias-correction from Panosky function
# ============================================================================= 
def corrcdf(datasim,dataobs):
    Diff = (np.ma.sort(dataobs))-(np.ma.sort(datasim))
    Coef = np.ma.polyfit((np.ma.sort(datasim)),Diff,5)
    #COEF = np.polynomial.Polynomial.fit((np.ma.sort(datasim)),DIFF,5)

    #except ValueError:
   #     pass
    Bias_corrected = np.polyval(Coef,datasim) + datasim
    #SATCDF = np.polynomial.polynomial.polyval(COEF,datasim) + datasim
    return Bias_corrected

# =============================================================================
#  Plotting function NE
# =============================================================================

def plot_swath(data, lons, lats, title):

    plt.style.use('default')
   # gd = Geodesic()
   # src_crs = ccrs.PlateCarree()
    lcc = ccrs.Mercator() 
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111, projection=lcc)
    cax = ax.pcolor(lons, lats, data, transform=ccrs.PlateCarree(), vmin=0.0, vmax=200, cmap='RdYlGn_r')

    ax.set_extent([-83.0, -67.0, 36.0, 48.0])

    cbar = fig.colorbar(cax, ticks=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0], orientation='vertical', extendfrac='auto', fraction=0.06, pad=0.02)
    #cbar = fig.colorbar(cax, orientation='horizontal', extendfrac='auto', fraction=0.06, pad=0.02)
    #cbar = fig.colorbar(cax, ticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0], orientation='horizontal', extendfrac='auto', fraction=0.06, pad=0.02)

    cbar.ax.tick_params(labelsize=17)
    cbar.set_label('Precipitation', fontsize=20, labelpad=5)

    ax.coastlines(resolution='50m', color='black')
    ax.add_feature(cf.BORDERS, linewidth=1.5, color='black')
    ax.add_feature(cf.LAKES.with_scale('50m'), facecolor=cf.COLORS['water'])
    ax.add_feature(cf.STATES)

    plt.title(title,fontsize=23)

    #plt.savefig('/Users/dimitrisherrera/Downloads/Figure_1.png', dpi=600) 
    
    return(plt.show())

# =============================================================================
# Exercise
# =============================================================================
# Step 1: Load coarse and high-resolution (target) monthly data
# Coarse data: CRUts4.07
coarse = xr.open_dataset('/Users/dimitrisherrera/Downloads/cru_ts4.07.1901.2022.pre.dat.nc')

# Target data: CHIRPSv2.0
target = xr.open_dataset('/Users/dimitrisherrera/Documents/Climate_Data/CHIRPSv2/CHIRPS-monthly/chirps-v2.0.monthly.nc')

# Step 2: Select a target area, say, the CONUS:
coarse = coarse.where((coarse.lat > 21) & (coarse.lat < 49) & (coarse.lon > -127.00) & (coarse.lon < -67.00),drop=True)
target = target.where((target.latitude > 21) & (target.latitude < 49) & (target.longitude > -127.00) & (target.longitude < -67.00),drop=True)

# Step 3: Select a common time interval: Jan 1984 - Dec 2020
coarse = coarse.sel(time=slice(f'{1984}-01-01',f'{2020}-12-16',1))
target = target.sel(time=slice(f'{1984}-01-01',f'{2020}-12-01',1))
    
# Step 4: Regrid the target data to fit the coarse resolution using the regrid function previously defined.
coarser = regrid(target, 0.5)

# Step 5: Create a land-mask using the target data (CHIRPS)
target_p = target.precip

# Step 6: preprocess the input data before applying the bias-correction to the coarse data
coarser_p = coarser.precip
coarser_p = np.ma.masked_invalid(coarser_p)

# Step 7: Preprocess the data for QM bias-correction
    # Fill masked grids with a random number (e.g., 1.0) because NumPy.Polyfit has a bug to deal with masked arrays
coarser_p = np.ma.filled(coarser_p, fill_value=1.0)
coarse_p = coarse.pre
coarse_p = np.ma.masked_invalid(coarse_p)
coarse_p = np.ma.filled(coarse_p, 1.0)

# Step 8: apply the bias-correction downscaling
# Create an array filled with zeros
bias_corr = np.zeros_like(coarse.pre)   
scale = np.zeros_like(coarse.pre) 
for i in range(len(coarser.lat)):
    for j in range(len(coarser.lon)):
        bias_corr[:,i,j]  = corrcdf(coarse_p[:,i,j], coarser_p[:,i,j]) 
 
# Step 9: Calculate the scaling factors
scale_factors = bias_corr - coarser_p

# Step 10: Regrid scale_factors to the target resolution
    # Convert scale_factors from a numpy array to an Xarray array
scale_factorsxr = xr.DataArray(scale_factors, 
coords={'lat': coarse.lat,'lon': coarse.lon, 'time': coarse.time}, 
dims=["time", "lat", "lon"])
    # Regrid scaling factors to the target res. 0.05 deg.
scale_factors_target = regrid(scale_factorsxr, 0.05)

# Step 11: Finalize BCSD by aggregating the scaling factors to the target dataset
bcsd = np.array(scale_factors_target) + np.array(target_p)

# Step 12: make some plots
    # Create a concatenated file
concatenated = np.concatenate((bcsd[25,:,:], target_p[25,:,:]))
concatenated = np.reshape(concatenated, (2,560,1200))

# Plot your data
# Lat and lon
latd = target.latitude
lond = target.longitude

latc = coarse.lat
lonc = coarse.lon

# CRU data
#plt.imshow(coarse.pre[25,::-1,:], vmin=0, vmax=200, cmap='rainbow')
#plt.title("CRU")
plot_swath(coarser_p[25,70::,700::], lonc[700::], latc[70::], "CRU")

plot_swath(bcsd[25,70::,700::], lond[700::], latd[70::], "Downscaled CRU")
    # Original target and downscaled data
#titles = ['BCSD Downscaled', 'CHIRPS']
#fig, axarr = plt.subplots(2, figsize=(7, 7))
#for i in range(2):
#    axarr[i].imshow(concatenated[i,::-1,:], vmin=0, vmax=200, cmap='rainbow')
 #   axarr[i].title.set_text(titles[i])
#    plt.tight_layout()


