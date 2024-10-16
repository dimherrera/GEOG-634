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
def regrid(ds):
    ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(21, 49.0, 0.05)),
        "lon": (["lon"], np.arange(-127.0, -67.0, 0.05)),})

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
    
# Step 4: Regrid the coarse data to fit the target resolution using the regrid function previously defined.
finer = regrid(coarse)

# Step 5: Create a land-mask using the target data (CHIRPS)
mask = target.precip[0,:,:]
mask = np.ma.masked_invalid(mask)
mask = np.ma.where(mask!=0.0, 0.0, mask)

# Step 6: preprocess the input data before applying the bias-correction to the coarse data
finer_p = finer.pre
finer_p = np.ma.masked_invalid(finer_p)
finer_p = finer_p + mask
# Fill masked grids with a random number (e.g., 1.0) because NumPy.Polyfit has a bug to deal with masked arrays
finer_p = np.ma.filled(finer_p, fill_value=1.0)

target_p = target.precip
target_p = np.ma.masked_invalid(target_p)
target_p = np.ma.filled(target_p, fill_value=1.0)

# Step 7: apply the bias-correction downscaling
# Create an array filled with zeros
downscaled = np.zeros_like(target.precip)   

for i in range(len(target.latitude)):
    for j in range(len(target.longitude)):
        downscaled[:,i,j] = corrcdf(finer_p[:,i,j], target_p[:,i,j]) 
    
# Step 8: make some plots
concatenated = np.concatenate((finer_p[25,:,:], downscaled[25,:,:], target_p[25,:,:]))
concatenated = np.reshape(concatenated, (3,560,1200))

# Step 9: again, apply a land-mask to the downscaled and original data
concatenated = concatenated + mask


# Step 10: plot your data
# Maps
titles = ['CRU interpolated', 'QM Downscaled', 'CHIRPS']
fig, axarr = plt.subplots(3, figsize=(10, 10))
for i in range(3):
    axarr[i].imshow(concatenated[i,::-1,:], vmin=0, vmax=200, cmap='rainbow')
    axarr[i].title.set_text(titles[i])
    plt.tight_layout()
    
# CDFs:
plt.plot(cdf(finer_p[:,301,400]), color='red', label='Original')
plt.plot(cdf(downscaled[:,301,400]), color='orange', label='Downscaled')
plt.plot(cdf(target_p[:,301,400]), color='black', label='Target data')  
plt.legend()    
