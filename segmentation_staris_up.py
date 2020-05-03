#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:58:22 2020

@author: Filip P. Paszkiewicz
"""

import numpy as np
import matplotlib.pyplot as plt

from Base_Function import *
from scipy.signal import find_peaks
from segmentation_HS import Segmentation_HS

def Segmentation_Staris_Up(PATH1,PATH2,sample_thigh,sample_shank,FS=1000,Flag_Save_Data=False,Flag_Plotting=False):
    """ Setting parameteres """
    LOWCUT_MMG = 1                                     # Low cut-off frequency for MMG
    HIGHCUT_MMG = 100                                  # High cut-off frequency for MMG
    ORDER_MMG = 3                                      # Filter order for MMG

    FRECUT_Dyna = 15                                   # Low cut-off freunecy for segmetnation
    FRECUT_Seg = 6                                     # Low cut-off frequency for Dynamic data
    ORDER_Dyna = 2                                     # Filter order for Dynamic data

    window_len = int(0.2*FS)+1                         # Size of moving sample window (set to 200ms)
    gait_cycle_duration = int(0.75*FS)+1               # Duration of an average gait cycle (varies based on gait cycle)  
    
    """ Load the csv file into a Numpy array """
    raw_raw_thigh = Open_file_to_array(PATH1)         # Read the CSV file and return a numpy 2D array
    raw_raw_shank = Open_file_to_array(PATH2)         # Read the CSV file and return a numpy 2D array
    
    """ Predefine array sizes """
    raw_gyro_thigh = raw_raw_thigh[:,1:7]             # GYRO signal raw data
    raw_acc_thigh = raw_raw_thigh[:,13:19]            # Acc signal raw data
    raw_mmg_thigh = raw_raw_thigh[:,19:35]            # MMG signal raw data
    
    raw_gyro_shank = raw_raw_shank[:,1:7]             # GYRO signal raw data 
    raw_acc_shank = raw_raw_shank[:,13:19]            # Acc signal raw data    

    """ Convert raw accelerometer data into 16-bit singed integar """
    acc_16bit_thigh = acc_raw_to_16bit(raw_acc_thigh)  # Translate high and low bites of raw inertial signals (accelerometer)
    """ Convert raw accelerometer data into 16-bit singed integar """
    gyro_16bit_thigh = gyro_raw_to_16bit(raw_gyro_thigh)
    """ Convert raw mechanomyography data into a 10-bit unsinged integar """
    mmg_10bit_thigh = mmg_raw_to_10bit(raw_mmg_thigh)  # Translate high and low bites of raw mechanomayography signals 
    """ Convert raw accelerometer data into 16-bit singed integar """
    acc_16bit_shank = acc_raw_to_16bit(raw_acc_shank)  # Translate high and low bites of raw inertial signals (accelerometer)
    """ Convert raw accelerometer data into 16-bit singed integar """
    gyro_16bit_shank = gyro_raw_to_16bit(raw_gyro_shank)


    """ Designing desired filter """
    den_filter_seg, nom_filter_seg   =  butter_low(FRECUT_Seg, FS, ORDER_Dyna) # Calculating nom and den for a filter of specified order to filter for Segmentation
    den_filter_dyna, nom_filter_dyna   =  butter_low(FRECUT_Dyna, FS, ORDER_Dyna) # Calculating nom and den for a filter of specified order to filter Dynamic data
    den_filter_mmg, nom_filter_mmg     =  butter_bandpass(LOWCUT_MMG, HIGHCUT_MMG, FS, ORDER_MMG) # Calculating nom and den for a filter of specified order to filter MMG data
    
    """ Finding segmnetation points """
    filter_seg = butter_filter(gyro_16bit_shank, den_filter_seg, nom_filter_seg)
    filter_seg = filter_seg - np.mean(filter_seg,axis=0)
    if PATH1.find('N010') !=-1:
        filter_seg = -filter_seg

    """ Identifying Heel Strikes in IMU data """
    peaks_gyro, _ = find_peaks(filter_seg[:,2])  
    
    HS = []
    for outer in range(1,len(peaks_gyro)):
        if filter_seg[peaks_gyro[outer-1],2]>250:
            HS = np.append(HS,peaks_gyro[outer])
    HS = HS.astype(int)
               
    """ Filtering converted data"""
    filter_acc_shank = butter_filter(acc_16bit_shank, den_filter_dyna, nom_filter_dyna)                     # Filter data with calculated filter (ACC) shank
    filter_gyro_shank = butter_filter(gyro_16bit_shank, den_filter_dyna, nom_filter_dyna)                   # Filter data with calculated filter (Gyro) shank 
    filter_acc_thigh = butter_filter(acc_16bit_thigh, den_filter_dyna, nom_filter_dyna)                     # Filter data with calculated filter (ACC) thigh
    filter_gyro_thigh = butter_filter(gyro_16bit_thigh, den_filter_dyna, nom_filter_dyna)                   # Filter data with calculated filter (Gyro) thigh
    filter_mmg_thigh = butter_filter(mmg_10bit_thigh, den_filter_mmg, nom_filter_mmg)                       # Filter data with calculated filter (MMG) thigh 
    
    filter_gyro_thigh = filter_gyro_thigh - np.mean(filter_gyro_thigh,axis=0)
    filter_gyro_shank = filter_gyro_shank - np.mean(filter_gyro_shank,axis=0)
    
    """ Find active MMG channels """
    active_channels = np.mean(mmg_10bit_thigh,axis=0)
    mean_channel = np.mean(active_channels)
    
    mmg_channels = [] 
    for idx in range(0, len(active_channels)) : 
        if active_channels[idx] > mean_channel: 
            mmg_channels.append(idx) 
    data_mmg = np.column_stack((filter_mmg_thigh[:,0],filter_mmg_thigh[:,2],filter_mmg_thigh[:,3],filter_mmg_thigh[:,5],filter_mmg_thigh[:,7]))
    
    """ Data to be segmented """
    data_thigh = np.concatenate((data_mmg, filter_gyro_thigh, filter_acc_thigh),axis=1)
    data_shank = np.concatenate((filter_gyro_shank, filter_acc_shank),axis=1)
            
    """ Segmentation of  data based on Gait Events (here Heel Strikes)"""
    seg_data_thigh_stairs_up, seg_data_shank_stairs_up = Segmentation_HS(data_thigh,data_shank,gait_cycle_duration,window_len,HS)
    
    """ Finding name for saving CSV files """
    if Flag_Save_Data == True:
        if PATH1.find('ME') !=-1:
            begin = PATH1.find('ME')
            name_thigh = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+2] + '_stairs_up_thigh_' 
            name_shank = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+2] + '_stairs_up_shank_'  
        else:
            begin = PATH1.find('N',22)
            name_thigh = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+4] + '_stairs_up_thigh_' 
            name_shank = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+4] + '_stairs_up_shank_' 
        
        """ Saving window samples as individual CSV files """  
        for i in range(seg_data_thigh_stairs_up.shape[2]):
            temp = name_thigh + str(sample_thigh) + '.csv'
            np.savetxt(temp, seg_data_thigh_stairs_up[:,:,i], delimiter=",")
            sample_thigh = sample_thigh + 1
        for i in range(seg_data_shank_stairs_up.shape[2]):
            temp = name_shank + str(sample_shank) + '.csv'
            np.savetxt(temp, seg_data_shank_stairs_up[:,:,i], delimiter=",")
            sample_shank = sample_shank + 1

    """ Some plotting for visualistion (and publications)  """
    if Flag_Plotting == True:
        Plot_Data(data_thigh,data_shank,FS,HS,sample_thigh,'Stair Ascent')
        
    return sample_thigh, sample_shank