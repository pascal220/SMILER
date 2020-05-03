#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:26:27 2020

@author: Filip Paszkiewicz
"""
import numpy as np

from features_ml import *
from Base_Function import *

def Call_all_Features(data_array,FS):
    data_array_features = Feature_RMS(data_array)
    data_array_features = np.append(data_array_features,Feature_IAV(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_MAV(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_SSI(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_VAR(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_Kurtosis(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_AVT3(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_AVT4(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_AVT5(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_WL(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_AAC(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_DAMV(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_DASDV(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_MMAV1(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_MMAV2(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_AR(data_array),axis=0)
    # data_array_features = np.append(data_array_features,Feature_LD(data_array),axis=0)
    data_array_features = np.append(data_array_features,Feature_Mean_F(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_Median_F(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_Peak_F(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_Mean_P(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_Total_P(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_SM1(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_SM2(data_array,FS),axis=0)
    data_array_features = np.append(data_array_features,Feature_SM3(data_array,FS),axis=0)
    
    return data_array_features

def Extract_Features(path,file_path,FS=1000):
    data_array = Open_file_to_array(path+file_path)
    if data_array.shape[1] == 11:
        data_mmg = data_array[:,:5]
        data_IMU = data_array[:,5:]
        
        features_MMG_back = Call_all_Features(data_mmg,FS)
        features_IMU_back = Call_all_Features(data_IMU,FS)
        return features_MMG_back, features_IMU_back
    else:
        features_to_back =  Call_all_Features(data_array,FS)
        return features_to_back
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    