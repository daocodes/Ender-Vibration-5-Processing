#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:31:04 2025

@author: daobui
"""




import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')


import numpy as np

import matplotlib.pyplot as plt


from scipy.stats import kurtosis
from scipy.stats import skew

#These are all of the functions for transforming the data and there is an fft version for each one

def amplitude(data):
    amplitudes = []
    for acceleration in data:
        # Compute amplitude as average of max absolute accel over all rows and columns 1, 2, 3
        max_ac0 = np.max(np.abs(acceleration[:, 1]))
        max_ac1 = np.max(np.abs(acceleration[:, 2]))
        max_ac2 = np.max(np.abs(acceleration[:, 3]))

        avg_amplitude = (max_ac0 + max_ac1 + max_ac2) / 3.0
        amplitudes.append(avg_amplitude)

    return {'Amplitude': np.array(amplitudes)}

def decay(data):
    decay_values = []
    for acceleration in data:
        x = acceleration[:, 0]  # time or frequency axis
        decay_factor = 1 / (x + 1e-6)  # decay per element
        
        # Apply decay to acceleration components (cols 1,2,3)
        decayed_vals = acceleration[:, 1:4] * decay_factor[:, np.newaxis]
        
        # Average across columns 1,2,3, then average across all rows for one scalar
        avg_decay_per_row = np.mean(decayed_vals, axis=1)
        avg_decay = np.mean(avg_decay_per_row)
        
        decay_values.append(avg_decay)

    return {'Decay': np.array(decay_values)}


def mean(data):
    mean_values = []
    
    for acceleration in data:
        # Compute mean of each acceleration column over all rows
        ac0_mean = np.mean(acceleration[:, 1])
        ac1_mean = np.mean(acceleration[:, 2])
        ac2_mean = np.mean(acceleration[:, 3])
        
        # Average the three means into one scalar
        avg_mean = (ac0_mean + ac1_mean + ac2_mean) / 3.0
        
        mean_values.append(avg_mean)
        
    return {'Mean': np.array(mean_values)}

def standard_deviation(data):
    std_values = []
    
    for acceleration in data:
        ac0_std = np.std(acceleration[:, 1])
        ac1_std = np.std(acceleration[:, 2])
        ac2_std = np.std(acceleration[:, 3])
        
        avg_std = (ac0_std + ac1_std + ac2_std) / 3.0
        std_values.append(avg_std)
        
    return {'Standard Deviation': np.array(std_values)}


def standard_deviation_fft(data):
    std_values = []
    
    for acceleration in data:
        ac0 = np.abs(acceleration[:, 1])
        ac1 = np.abs(acceleration[:, 2])
        ac2 = np.abs(acceleration[:, 3])
        
        ac0_std = np.std(ac0)
        ac1_std = np.std(ac1)
        ac2_std = np.std(ac2)
        
        avg_std = (ac0_std + ac1_std + ac2_std) / 3.0
        std_values.append(avg_std)
        
    return {'Standard Deviation FFT': np.array(std_values)}
            
    
    
def rms(data):
    rms_values = []
    
    for acceleration in data:
        ac0_rms = np.sqrt(np.mean(acceleration[:, 1] ** 2))
        ac1_rms = np.sqrt(np.mean(acceleration[:, 2] ** 2))
        ac2_rms = np.sqrt(np.mean(acceleration[:, 3] ** 2))
        
        avg_rms = (ac0_rms + ac1_rms + ac2_rms) / 3.0
        rms_values.append(avg_rms)
        
    return {'RMS': np.array(rms_values)}


def rms_fft(data):
    rms_values = []
    
    for acceleration in data:
        mag0 = np.abs(acceleration[:, 1])
        mag1 = np.abs(acceleration[:, 2])
        mag2 = np.abs(acceleration[:, 3])
        
        ac0_rms = np.sqrt(np.mean(mag0 ** 2))
        ac1_rms = np.sqrt(np.mean(mag1 ** 2))
        ac2_rms = np.sqrt(np.mean(mag2 ** 2))
        
        avg_rms = (ac0_rms + ac1_rms + ac2_rms) / 3.0
        rms_values.append(avg_rms)
        
    return {'RMS FFT': np.array(rms_values)}


def shape_factor(data):
    sf_values = []
    
    for acceleration in data:
        sf_ac0 = np.sqrt(np.mean(acceleration[:, 1] ** 2)) / (np.mean(np.abs(acceleration[:, 1])) + 1e-8)
        sf_ac1 = np.sqrt(np.mean(acceleration[:, 2] ** 2)) / (np.mean(np.abs(acceleration[:, 2])) + 1e-8)
        sf_ac2 = np.sqrt(np.mean(acceleration[:, 3] ** 2)) / (np.mean(np.abs(acceleration[:, 3])) + 1e-8)
        
        avg_sf = (sf_ac0 + sf_ac1 + sf_ac2) / 3.0
        sf_values.append(avg_sf)
        
    return {'Shape Factor': np.array(sf_values)}


def shape_factor_fft(data):
    sf_values = []
    
    for acceleration in data:
        mag0 = np.abs(acceleration[:, 1])
        mag1 = np.abs(acceleration[:, 2])
        mag2 = np.abs(acceleration[:, 3])
        
        sf_ac0 = np.sqrt(np.mean(mag0 ** 2)) / (np.mean(mag0) + 1e-8)
        sf_ac1 = np.sqrt(np.mean(mag1 ** 2)) / (np.mean(mag1) + 1e-8)
        sf_ac2 = np.sqrt(np.mean(mag2 ** 2)) / (np.mean(mag2) + 1e-8)
        
        avg_sf = (sf_ac0 + sf_ac1 + sf_ac2) / 3.0
        sf_values.append(avg_sf)
        
    return {'Shape Factor FFT': np.array(sf_values)}


def kurtosis_apply(data):
    kurt_values = []
    
    for acceleration in data:
        kurt_ac0 = kurtosis(acceleration[:, 1], fisher=False)
        kurt_ac1 = kurtosis(acceleration[:, 2], fisher=False)
        kurt_ac2 = kurtosis(acceleration[:, 3], fisher=False)
        
        avg_kurt = (kurt_ac0 + kurt_ac1 + kurt_ac2) / 3.0
        kurt_values.append(avg_kurt)
        
    return {'Kurtosis': np.array(kurt_values)}

def skewness(data):
    skew_values = []
    
    for acceleration in data:
        skew_ac0 = skew(acceleration[:, 1])
        skew_ac1 = skew(acceleration[:, 2])
        skew_ac2 = skew(acceleration[:, 3])
        
        avg_skew = (skew_ac0 + skew_ac1 + skew_ac2) / 3.0
        skew_values.append(avg_skew)
        
    return {'Skewness': np.array(skew_values)}


def skewness_fft(data):
    skew_values = []
    
    for acceleration in data:
        mag_ac0 = np.abs(acceleration[:, 1])
        mag_ac1 = np.abs(acceleration[:, 2])
        mag_ac2 = np.abs(acceleration[:, 3])
        
        skew_ac0 = skew(mag_ac0)
        skew_ac1 = skew(mag_ac1)
        skew_ac2 = skew(mag_ac2)
        
        avg_skew = (skew_ac0 + skew_ac1 + skew_ac2) / 3.0
        skew_values.append(avg_skew)
        
    return {'Skewness': np.array(skew_values)}
    


def impulse_factor(data):
    impulse_values = []
    
    for acceleration in data:
        imp_ac0 = np.max(np.abs(acceleration[:, 1])) / (np.mean(np.abs(acceleration[:, 1])) + 1e-8)
        imp_ac1 = np.max(np.abs(acceleration[:, 2])) / (np.mean(np.abs(acceleration[:, 2])) + 1e-8)
        imp_ac2 = np.max(np.abs(acceleration[:, 3])) / (np.mean(np.abs(acceleration[:, 3])) + 1e-8)

        avg_impulse = (imp_ac0 + imp_ac1 + imp_ac2) / 3.0
        impulse_values.append(avg_impulse)
    
    return {'Impulse Factor': np.array(impulse_values)}




def crest_factor(data):
    crest_values = []
    
    for acceleration in data:
        ac0_peak = np.max(np.abs(acceleration[:, 1]))
        ac0_rms = np.sqrt(np.mean(acceleration[:, 1] ** 2))
        cf_ac0 = ac0_peak / (ac0_rms + 1e-8)

        ac1_peak = np.max(np.abs(acceleration[:, 2]))
        ac1_rms = np.sqrt(np.mean(acceleration[:, 2] ** 2))
        cf_ac1 = ac1_peak / (ac1_rms + 1e-8)

        ac2_peak = np.max(np.abs(acceleration[:, 3]))
        ac2_rms = np.sqrt(np.mean(acceleration[:, 3] ** 2))
        cf_ac2 = ac2_peak / (ac2_rms + 1e-8)

        avg_cf = (cf_ac0 + cf_ac1 + cf_ac2) / 3.0
        crest_values.append(avg_cf)
    
    return {'Crest Factor': np.array(crest_values)}




plt.close('all')


sample_data = []   # For (X_Value, avg_acceleration) from Sample_ files
fft_data = []      # For (X_Value, avg_acceleration) from TestFFT_ files

# Process Sample_*.lvm files
for i in range(1, 20):
    D = np.loadtxt(f'Sample_{i}.lvm', skiprows=23, delimiter=',')
    # Keep all columns as is (time + 3 acceleration columns)
    sample_data.append(D)  

# Process TestFFT_*.lvm files
for i in range(1, 20):
    D = np.loadtxt(f'TestFFT_{i}.lvm', skiprows=23, delimiter=',')
    fft_data.append(D)
    


plt.figure(figsize=(12, 6))

for i in range(10):
    if i == 0:
        transform = amplitude(sample_data)
    elif i == 1:
        transform = decay(sample_data)
    elif i == 2:
        transform = mean(sample_data)
    elif i == 3:
        transform = standard_deviation(sample_data)
    elif i == 4:
        transform = rms(sample_data)
    elif i == 5:
        transform = shape_factor(sample_data)
    elif i == 6:
        transform = kurtosis_apply(sample_data)
    elif i == 7:
        transform = skewness(sample_data)
    elif i == 8: 
        transform = impulse_factor(sample_data)
    elif i == 9:
        transform = crest_factor(sample_data)
   
    x_sample = np.arange(len(sample_data))
    
    for label, y in transform.items():
        plt.plot(x_sample[:len(y)], y, label=label)

plt.title("Sample Data Transformations")
plt.xlabel("Data Sets")
plt.ylabel("Feature Values")
plt.legend()
plt.grid(True)
    
plt.savefig('Sample_Vibration_Data.png', dpi=300)
plt.savefig("Sample_Vibration_Data.pdf")


plt.figure(figsize=(12, 6))

for i in range(10):
    # Use FFT versions where available, otherwise skip or use time domain as fallback
    if i == 0:
        # amplitude FFT version not defined — skip or implement if needed
        transform = amplitude(fft_data)
    elif i == 1:
        # decay FFT version not defined — skip or implement if needed
        transform = decay(fft_data)
    elif i == 2:
        # mean FFT version not defined — skip or implement if needed
        transform = mean(fft_data)
    elif i == 3:
        transform = standard_deviation_fft(fft_data)
    elif i == 4:
        transform = rms_fft(fft_data)
    elif i == 5:
        transform = shape_factor_fft(fft_data)
    elif i == 6:
        # kurtosis FFT version not defined — skip or implement if needed
        transform = kurtosis_apply(fft_data)
    elif i == 7:
        transform = skewness_fft(fft_data)
    elif i == 8:
        # impulse_factor FFT version not defined — skip or implement if needed
        transform = impulse_factor(fft_data)
    elif i == 9:
        # crest_factor FFT version not defined — skip or implement if needed
        transform = crest_factor(fft_data)

    x_fft = np.arange(len(fft_data))
    
    for label, y in transform.items():
        plt.plot(x_fft[:len(y)], y, label=label)

plt.title("FFT Data Transformations")
plt.xlabel("FFT Data Sets")
plt.ylabel("Feature Values")
plt.legend()
plt.grid(True)

plt.savefig('FFT_Vibration_Data.png', dpi=300)
plt.savefig("FFT_Vibration_Data.pdf")

    
    

    
    

    
    
    
    
    
    