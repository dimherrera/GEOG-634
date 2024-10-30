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
        # CRU data
plt.imshow(coarse.pre[25,::-1,:], vmin=0, vmax=200, cmap='rainbow')
plt.title("CRU")

    # Original target and downscaled data
titles = ['BCSD Downscaled', 'CHIRPS']
fig, axarr = plt.subplots(2, figsize=(7, 7))
for i in range(2):
    axarr[i].imshow(concatenated[i,::-1,:], vmin=0, vmax=200, cmap='rainbow')
    axarr[i].title.set_text(titles[i])
    plt.tight_layout()
  

    