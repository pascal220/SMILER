#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:05:57 2020

@author: Filip Paszkiewicz
"""
import numpy as np
import matplotlib.pyplot as plt

from Base_Function import *
from scipy.signal import find_peaks
from windowing_SS_data import Windowing_SS_Data

def Segmentation_SS(PATH1,PATH2,sample_sit_thigh,sample_stand_thigh,sample_sit_shank,sample_stand_shank,FS=1000,Flag_Save_Data=False,Flag_Plotting=False):
    """ Setting parameteres """
    LOWCUT_MMG = 1                                     # Low cut-off frequency for MMG
    HIGHCUT_MMG = 100                                  # High cut-off frequency for MMG
    ORDER_MMG = 3                                      # Filter order for MMG
    
    FRECUT_Dyna = 15                                   # Low cut-off frequency for Dynamic data 
    FRECUT_Seg = 1                                     # Low cut-off frequency for segmetnation
    ORDER_Dyna = 2                                     # Filter order for Dynamic data
    
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
    
    
    """ Finding segmnetation points (peaks and troughs) """
    filter_seg = butter_filter(acc_16bit_thigh, den_filter_seg, nom_filter_seg)
    if PATH1.find('N010') !=-1:
        peaks_acc, _ = find_peaks(-filter_seg[:,1],height=-0.1,distance=2500)
        troughs_acc, _ = find_peaks(filter_seg[:,1],height=0.1,distance=2500)
    else:
        peaks_acc, _ = find_peaks(filter_seg[:,1],height=0.2,distance=1500)
        troughs_acc, _ = find_peaks(-filter_seg[:,1],height=-0.4,distance=2000)
    
    """ Filtering converted data"""
    filter_acc_shank = butter_filter(acc_16bit_shank, den_filter_dyna, nom_filter_dyna)                     # Filter data with calculated filter (ACC) shank
    filter_gyro_shank = butter_filter(gyro_16bit_shank, den_filter_dyna, nom_filter_dyna)                   # Filter data with calculated filter (Gyro) shank 
    filter_acc_thigh = butter_filter(acc_16bit_thigh, den_filter_dyna, nom_filter_dyna)                     # Filter data with calculated filter (ACC) thigh
    filter_gyro_thigh = butter_filter(gyro_16bit_thigh, den_filter_dyna, nom_filter_dyna)                   # Filter data with calculated filter (Gyro) thigh
    filter_mmg_thigh = butter_filter(mmg_10bit_thigh, den_filter_mmg, nom_filter_mmg)                       # Filter data with calculated filter (MMG) thigh 
    
    """ Remove DC offset from Gyroscope signal """
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
    
    seg_sit_to_stand_thigh, seg_stand_to_sit_thigh = Windowing_SS_Data(data_thigh,peaks_acc,troughs_acc)
    seg_sit_to_stand_shank, seg_stand_to_sit_shank = Windowing_SS_Data(data_shank,peaks_acc,troughs_acc)
    
    """ Finding name for saving CSV files """
    if Flag_Save_Data == True:
        if PATH1.find('ME') !=-1:
            begin = PATH1.find('ME')
            name_thigh_sit = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+2] + '_sit_thigh_' 
            name_thigh_stand = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+2] + '_stand_thigh_' 
            name_shank_sit = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+2] + '_sit_shank_' 
            name_shank_stand = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+2] + '_stand_shank_' 
        else:
            begin = PATH1.find('N',22)
            name_thigh_sit = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+4] + '_sit_thigh_' 
            name_thigh_stand = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+4] + '_stand_thigh_' 
            name_shank_sit = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+4] + '_sit_shank_' 
            name_shank_stand = 'MMG_test/ML_Windowed_Data/' + PATH1[begin:begin+4] + '_stand_shank_' 
        
        """ Saving window samples as individual CSV files """
        for i in range(seg_sit_to_stand_thigh.shape[2]):
            temp = name_thigh_sit + str(sample_sit_thigh) + '.csv'
            np.savetxt(temp, seg_sit_to_stand_thigh[:,:,i], delimiter=",")
            sample_sit_thigh = sample_sit_thigh + 1        
        for i in range(seg_stand_to_sit_thigh.shape[2]):
            temp = name_thigh_stand + str(sample_stand_thigh) + '.csv'
            np.savetxt(temp, seg_stand_to_sit_thigh[:,:,i], delimiter=",")
            sample_stand_thigh = sample_stand_thigh + 1   
        for i in range(seg_sit_to_stand_shank.shape[2]):
            temp = name_shank_sit + str(sample_sit_shank) + '.csv'
            np.savetxt(temp, seg_sit_to_stand_shank[:,:,i], delimiter=",")
            sample_sit_shank = sample_sit_shank + 1
        for i in range(seg_stand_to_sit_shank.shape[2]):
            temp = name_shank_stand + str(sample_stand_shank) + '.csv'
            np.savetxt(temp, seg_stand_to_sit_shank[:,:,i], delimiter=",")
            sample_stand_shank = sample_stand_shank + 1
    
    """ Plotting SS """
    if Flag_Plotting == True:
        HS = [peaks_acc[1],troughs_acc[1],peaks_acc[2],troughs_acc[2],peaks_acc[3]]
        Plot_Data_SS(data_thigh,data_shank,sample_sit_thigh,FS,HS)                
    
    return sample_sit_thigh, sample_stand_thigh, sample_sit_shank, sample_stand_shank