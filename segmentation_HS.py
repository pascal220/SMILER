#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:50:58 2020

@author: Filip Paszkiewicz
"""
import numpy as np

def Last_HS(seg_data,data,HS,gait_cycle_duration,window_len,index):
    if (data.shape[0]-HS[0])<gait_cycle_duration:
        temp = data[HS[index]-int(gait_cycle_duration*0.5):HS[index]+gait_cycle_duration,:]# Window centered arounf the HS - thigh
    elif (data.shape[0]-HS[0])<int(gait_cycle_duration*0.5):
        temp = data[HS[index]-int(gait_cycle_duration*0.5):HS[index]+int(gait_cycle_duration*0.5),:]# Window centered arounf the HS - thigh
    elif (data.shape[0]-HS[0])<window_len:
        temp = data[HS[index]-int(gait_cycle_duration*0.5):HS[index]+window_len,:]# Window centered arounf the HS - thigh 
    else:
        temp = data[HS[index]-int(gait_cycle_duration*0.5):HS[index],:]# Window centered arounf the HS - shank
        
    for i in range(0,temp.shape[0]-window_len,50): # Segement rest of the window into 200ms samples. Stacked as a 3D matrix 200ms x no_channels x no_number of sample windows
        seg_data = np.dstack((seg_data,temp[i:i+window_len,:]))
    
    return seg_data

def Segmentation_HS(data_thigh,data_shank,gait_cycle_duration,window_len,HS):
    # Initialise segmented walking data by taking the first part of the window centered arounf the HS
    seg_data_thigh = data_thigh[HS[0]-int(gait_cycle_duration/2):HS[0]-int(gait_cycle_duration/2-window_len),:]
    seg_data_shank = data_shank[HS[0]-int(gait_cycle_duration/2):HS[0]-int(gait_cycle_duration/2-window_len),:]
    temp_thigh = data_thigh[HS[0]-int(gait_cycle_duration/2-50):HS[0]+int(gait_cycle_duration/2),:]# Window centered arounf the HS - thigh
    temp_shank = data_shank[HS[0]-int(gait_cycle_duration/2-50):HS[0]+int(gait_cycle_duration/2),:]# Window centered arounf the HS - shank
    for i in range(0,temp_thigh.shape[0]-window_len,50): # Segement rest of the window into 200ms samples. Stacked as a 3D matrix 200ms x no_channels x no_number of sample windows
        seg_data_thigh = np.dstack((seg_data_thigh,temp_thigh[i:i+window_len,:]))
        seg_data_shank = np.dstack((seg_data_shank,temp_shank[i:i+window_len,:]))
    
    # Data gathered from IMU on Shank and from IMU on the thigh can have differnet lengths
    index = 0
    for i in range(len(HS)-1,0,-1):
        if HS[i]<data_thigh.shape[0]:
            index = i
            break
        
    for i in range(1,index-1):
        temp_thigh = data_thigh[HS[i]-int(gait_cycle_duration*0.5):HS[i]+int(gait_cycle_duration*0.5),:]# Window centered arounf the HS - thigh
        temp_shank = data_shank[HS[i]-int(gait_cycle_duration*0.5):HS[i]+int(gait_cycle_duration*0.5),:]# Window centered arounf the HS - shank
        for i in range(0,temp_thigh.shape[0]-window_len,50): # Segement rest of the window into 200ms samples. Stacked as a 3D matrix 200ms x no_channels x no_number of sample windows
            seg_data_thigh = np.dstack((seg_data_thigh,temp_thigh[i:i+window_len,:]))
            seg_data_shank = np.dstack((seg_data_shank,temp_shank[i:i+window_len,:]))  
    
    seg_data_thigh = Last_HS(seg_data_thigh,data_thigh,HS,gait_cycle_duration,window_len,index)  
    seg_data_shank = Last_HS(seg_data_shank,data_shank,HS,gait_cycle_duration,window_len,index)    
    
    return seg_data_thigh, seg_data_shank