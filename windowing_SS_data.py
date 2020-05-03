#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:34:32 2020

@author: Filip Paszkiewicz
"""

import numpy as np


def Windowing_SS_Data(data,peaks_acc,troughs_acc,FS=1000):
    """ Initialisation """
    seg_data_sit_to_stand = []
    seg_data_stand_to_sit = []
    
    window_length = int(0.2*FS)+1
    
    """ Using peaks and troughs in y-axis of the thigh accelrometer to segment MMG and IMU data into windows of window_lengthms.
    A sliding window with setp 50 is used. """
    if peaks_acc[0]<troughs_acc[0]: # If intial sitting was not recorded
        temp_sit_to_stand = data[0:peaks_acc[0],:]
        temp_stand_to_sit = data[peaks_acc[0]:troughs_acc[0],:]
        seg_data_sit_to_stand = temp_sit_to_stand[0:window_length,:]
        seg_data_stand_to_sit = temp_stand_to_sit[0:window_length,:]
        
        # Data starts from sitting. The following is the first transition from siting to standing, and from standing
        # to sitting.
        for i in range(50,temp_sit_to_stand.shape[0]-window_length,50):
            seg_data_sit_to_stand = np.dstack((seg_data_sit_to_stand,temp_sit_to_stand[i:i+window_length,:]))
        for i in range(50,temp_stand_to_sit.shape[0]-window_length,50):
            seg_data_stand_to_sit = np.dstack((seg_data_stand_to_sit,temp_stand_to_sit[i:i+window_length,:]))
        
        # Data of continous transitioning from sit to stand, and stand to sit.
        for i in range(1,len(peaks_acc)):
            temp_sit_to_stand = data[troughs_acc[i-1]:peaks_acc[i],:]
            for i in range(0,temp_sit_to_stand.shape[0]-window_length,50):
                seg_data_sit_to_stand = np.dstack((seg_data_sit_to_stand,temp_sit_to_stand[i:i+window_length,:]))
        for i in range(1,len(troughs_acc)):
            temp_stand_to_sit = data[peaks_acc[i]:troughs_acc[i],:]
            for i in range(0,temp_stand_to_sit.shape[0]-window_length,50):
                seg_data_stand_to_sit = np.dstack((seg_data_stand_to_sit,temp_stand_to_sit[i:i+window_length,:]))
        
        if peaks_acc[-1]>troughs_acc[-1]:
            # Last transitiong from standing to sitting, if last trough is lacking
            temp_stand_to_sit = data[peaks_acc[-1]:-1,:]
            for i in range(0,temp_stand_to_sit.shape[0]-window_length,50):
                    seg_data_stand_to_sit = np.dstack((seg_data_stand_to_sit,temp_stand_to_sit[i:i+window_length,:]))
                
    elif peaks_acc[0]>troughs_acc[0]:
        temp_sit_to_stand = data[troughs_acc[0]:peaks_acc[0],:]
        temp_stand_to_sit = data[peaks_acc[0]:troughs_acc[1],:]
        seg_data_sit_to_stand = temp_sit_to_stand[0:window_length,:]
        seg_data_stand_to_sit = temp_stand_to_sit[0:window_length,:]
        
        # Data starts from sitting. The following is the first transition from siting to standing, and from standing
        # to sitting.
        for i in range(50,temp_sit_to_stand.shape[0]-window_length,50):
            seg_data_sit_to_stand = np.dstack((seg_data_sit_to_stand,temp_sit_to_stand[i:i+window_length,:]))
        for i in range(50,temp_stand_to_sit.shape[0]-window_length,50):
            seg_data_stand_to_sit = np.dstack((seg_data_stand_to_sit,temp_stand_to_sit[i:i+window_length,:]))
            
        # Data of continous transitioning from sit to stand, and stand to sit.
        for i in range(1,len(peaks_acc)):
            temp_sit_to_stand = data[troughs_acc[i]:peaks_acc[i],:]
            for i in range(0,temp_sit_to_stand.shape[0]-window_length,50):
                seg_data_sit_to_stand = np.dstack((seg_data_sit_to_stand,temp_sit_to_stand[i:i+window_length,:]))
        for i in range(2,len(troughs_acc)):
            temp_stand_to_sit = data[peaks_acc[i-1]:troughs_acc[i],:]
            for i in range(0,temp_stand_to_sit.shape[0]-window_length,50):
                seg_data_stand_to_sit = np.dstack((seg_data_stand_to_sit,temp_stand_to_sit[i:i+window_length,:]))                   
        
        if peaks_acc[-1]>troughs_acc[-1]:
            # Last transitiong from standing to sitting, if last trough is lacking
            temp_stand_to_sit = data[peaks_acc[-1]:-1,:]
            for i in range(50,temp_stand_to_sit.shape[0]-window_length,50):
                    seg_data_stand_to_sit = np.dstack((seg_data_stand_to_sit,temp_stand_to_sit[i:i+window_length,:]))
                    
    return seg_data_sit_to_stand, seg_data_stand_to_sit