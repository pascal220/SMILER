#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:46:48 2021

@author: Filip Paszkiewicz
"""

import numpy as np
import matplotlib.pyplot as plt

from Base_Function import *
from scipy.signal import find_peaks
from windowing_SS_data import Windowing_SS_Data
import matplotlib.pyplot as plt

# First two files the StS is with added weight. The second two files StS was at body weight.
def Segmentation_SS_Speed_Force(PATH1,PATH2,count,FS=1000,Flag_Plotting=False):
    """ Setting parameteres """
    LOWCUT_MMG = 1                                     # Low cut-off frequency for MMG
    HIGHCUT_MMG = 100                                  # High cut-off frequency for MMG
    ORDER_MMG = 3                                      # Filter order for MMG
    
    FRECUT_Dyna = 15                                   # Low cut-off frequency for Dynamic data 
    FRECUT_Seg = 1                                     # Low cut-off frequency for segmetnation
    ORDER_Dyna = 2                                     # Filter order for Dynamic data
    
    """ Load the csv file into a Numpy array """
    raw_raw_normal = Open_file_to_array(PATH1)         # Read the CSV file and return a numpy 2D array
    raw_raw_slow = Open_file_to_array(PATH2)           # Read the CSV file and return a numpy 2D array
    
    """ Predefine array sizes """
    raw_acc_normal = raw_raw_normal[:,13:19]           # Acc signal raw data
    raw_mmg_normal = raw_raw_normal[:,19:35]           # MMG signal raw data
    
    raw_acc_slow = raw_raw_slow[:,13:19]               # Acc signal raw data
    raw_mmg_slow = raw_raw_slow[:,19:35]               # MMG signal raw data
    
    """ Convert raw accelerometer data into 16-bit singed integar """
    acc_16bit_normal = acc_raw_to_16bit(raw_acc_normal)  # Translate high and low bites of raw inertial signals (accelerometer)
    """ Convert raw mechanomyography data into a 10-bit unsinged integar """
    mmg_10bit_normal = mmg_raw_to_10bit(raw_mmg_normal)  # Translate high and low bites of raw mechanomayography signals 
    """ Convert raw accelerometer data into 16-bit singed integar """
    acc_16bit_slow = acc_raw_to_16bit(raw_acc_slow)      # Translate high and low bites of raw inertial signals (accelerometer)
    """ Convert raw mechanomyography data into a 10-bit unsinged integar """
    mmg_10bit_slow = mmg_raw_to_10bit(raw_mmg_slow)      # Translate high and low bites of raw mechanomayography signals 
    
    """ Designing desired filter """
    den_filter_seg, nom_filter_seg   =  butter_low(FRECUT_Seg, FS, ORDER_Dyna) # Calculating nom and den for a filter of specified order to filter for Segmentation
    den_filter_dyna, nom_filter_dyna   =  butter_low(FRECUT_Dyna, FS, ORDER_Dyna) # Calculating nom and den for a filter of specified order to filter Dynamic data
    den_filter_mmg, nom_filter_mmg     =  butter_bandpass(LOWCUT_MMG, HIGHCUT_MMG, FS, ORDER_MMG) # Calculating nom and den for a filter of specified order to filter MMG data

    """ Finding segmnetation points (peaks and troughs) """
    # Normal Speed
    filter_seg = butter_filter(acc_16bit_normal, den_filter_seg, nom_filter_seg)
    peaks_acc_normal, _ = find_peaks(-filter_seg[:,1],height=0.9,distance=2500)
    troughs_acc_normal, _ = find_peaks(filter_seg[500:,1],threshold=-0.3,distance=2500)
    
    # Slow Speed
    filter_seg = butter_filter(acc_16bit_slow, den_filter_seg, nom_filter_seg)
    peaks_acc_slow, _ = find_peaks(-filter_seg[:,1],height=0.9,distance=2500)
    troughs_acc_slow, _ = find_peaks(filter_seg[500:,1],threshold=-0.3,distance=5500)
    
    """ Filtering converted data"""
    filter_acc_normal = butter_filter(acc_16bit_normal, den_filter_dyna, nom_filter_dyna)    # Filter data with calculated filter (ACC) shank
    filter_mmg_normal = butter_filter(mmg_10bit_normal, den_filter_mmg, nom_filter_mmg)      # Filter data with calculated filter (MMG) thigh
    filter_acc_slow   = butter_filter(acc_16bit_slow, den_filter_dyna, nom_filter_dyna)      # Filter data with calculated filter (ACC) thigh
    filter_mmg_slow   = butter_filter(mmg_10bit_slow, den_filter_mmg, nom_filter_mmg)        # Filter data with calculated filter (MMG) thigh 
    
    """ Pull out the right sensors for both normal and slow MMG """
    for_processing_normal = np.stack((filter_mmg_normal[:,0],filter_mmg_normal[:,2],filter_mmg_normal[:,3],filter_mmg_normal[:,5],filter_mmg_normal[:,7]),axis=1)
    for_processing_slow   = np.stack((filter_mmg_slow[:,0],filter_mmg_slow[:,2],filter_mmg_slow[:,3],filter_mmg_slow[:,5],filter_mmg_slow[:,7]),axis=1)
    
    """ Calculate Signal Power """
    signal_power_normal = np.sqrt(np.mean(for_processing_normal[0:35,:]**2,axis=0))**2
    for i in range(35,len(for_processing_normal)-35,30):
        signal_power_normal = np.vstack((signal_power_normal,np.sqrt(np.mean(for_processing_normal[i:i+30,:]**2,axis=0))**2))
    signal_power_slow = np.sqrt(np.mean(for_processing_slow[0:35,:]**2,axis=0))**2
    for i in range(35,len(for_processing_normal)-35,30):
        signal_power_slow = np.vstack((signal_power_slow,np.sqrt(np.mean(for_processing_slow[i:i+30,:]**2,axis=0))**2))
    
    """ Calculating Sound Volume """
    sound_volume_normal = 20*np.log10(np.abs(for_processing_normal))
    sound_volume_slow   = 20*np.log10(np.abs(for_processing_slow))
    
    """ Plotting Original Signal """
    vline_min = 0.3
    vline_max = -0.3
    
    time_normal = np.linspace(0,len(for_processing_normal)/FS,len(for_processing_normal))
    time_slow   = np.linspace(0,len(for_processing_slow)/FS,len(for_processing_slow))
    plt.figure(count)
    plt.subplot(121)
    plt.title('Normal Speed')
    plt.plot(time_normal[troughs_acc_normal[2]:troughs_acc_normal[3]],np.mean(for_processing_normal[troughs_acc_normal[2]:troughs_acc_normal[3],0:3],axis=1),label='Quadriceps')
    plt.plot(time_normal[troughs_acc_normal[2]:troughs_acc_normal[3]],np.mean(for_processing_normal[troughs_acc_normal[2]:troughs_acc_normal[3],3:],axis=1),label='Hamstring')
    for trough in range(2,4):
        plt.vlines(troughs_acc_normal[trough]/FS,vline_min,vline_max,color='black',linestyle='--')
    for peak in range(3,4):
        plt.vlines(peaks_acc_normal[peak]/FS,vline_min,vline_max,color='red',linestyle='--')
    plt.ylabel('Volume (dB)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.subplot(122)
    plt.title('Slow Speed')
    plt.plot(time_slow[troughs_acc_slow[2]:troughs_acc_slow[3]],np.mean(for_processing_slow[troughs_acc_slow[2]:troughs_acc_slow[3],0:3],axis=1),label='Quadriceps')
    plt.plot(time_slow[troughs_acc_slow[2]:troughs_acc_slow[3]],np.mean(for_processing_slow[troughs_acc_slow[2]:troughs_acc_slow[3],3:],axis=1),label='Hamstring')
    for trough in range(2,4):
        plt.vlines(troughs_acc_slow[trough]/FS,vline_min,vline_max,color='black',linestyle='--')
    for peak in range(3,4):
        plt.vlines(peaks_acc_slow[peak]/FS,vline_min,vline_max,color='red',linestyle='--')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # """ Plotting Sound Volume """
    # vline_min = -85
    # vline_max = -10
    
    # time_normal = np.linspace(0,len(sound_volume_normal)/FS,len(sound_volume_normal))
    # time_slow   = np.linspace(0,len(sound_volume_slow)/FS,len(sound_volume_slow))
    # plt.figure(count)
    # plt.subplot(121)
    # plt.title('Normal Speed')
    # plt.plot(time_normal[troughs_acc_normal[2]:troughs_acc_normal[3]],np.mean(sound_volume_normal[troughs_acc_normal[2]:troughs_acc_normal[3],0:3],axis=1),label='Quadriceps')
    # plt.plot(time_normal[troughs_acc_normal[2]:troughs_acc_normal[3]],np.mean(sound_volume_normal[troughs_acc_normal[2]:troughs_acc_normal[3],3:],axis=1),label='Hamstring')
    # for trough in range(2,4):
    #     plt.vlines(troughs_acc_normal[trough]/FS,vline_min,vline_max,color='black',linestyle='--')
    # for peak in range(3,4):
    #     plt.vlines(peaks_acc_normal[peak]/FS,vline_min,vline_max,color='red',linestyle='--')
    # plt.ylabel('Volume (dB)')
    # plt.xlabel('Time (s)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # plt.subplot(122)
    # plt.title('Slow Speed')
    # plt.plot(time_slow[troughs_acc_slow[2]:troughs_acc_slow[3]],np.mean(sound_volume_slow[troughs_acc_slow[2]:troughs_acc_slow[3],0:3],axis=1),label='Quadriceps')
    # plt.plot(time_slow[troughs_acc_slow[2]:troughs_acc_slow[3]],np.mean(sound_volume_slow[troughs_acc_slow[2]:troughs_acc_slow[3],3:],axis=1),label='Hamstring')
    # for trough in range(2,4):
    #     plt.vlines(troughs_acc_slow[trough]/FS,vline_min,vline_max,color='black',linestyle='--')
    # for peak in range(3,4):
    #     plt.vlines(peaks_acc_slow[peak]/FS,vline_min,vline_max,color='red',linestyle='--')
    # plt.xlabel('Time (s)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    """ Plotting Signal Power """
    # plt.figure(count)
    # plt.subplot(121)
    # plt.title('Quadriceps - Body Weight')
    # plt.plot(np.mean(signal_power_normal[:,0:3],axis=1),label='Normal Speed')
    # plt.plot(np.mean(signal_power_slow[:,0:3],axis=1),label='Slow Speed')
    # plt.ylabel('Signal Power')
    # plt.xlabel('Samples')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    
    # plt.subplot(122)
    # plt.title('Hamstring - Body Weight')
    # plt.plot(np.mean(signal_power_normal[:,3:],axis=1),label='Normal Speed',color='r')
    # plt.plot(np.mean(signal_power_slow[:,3:],axis=1),label='Slow Speed',color='g')
    # plt.xlabel('Samples')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()





















