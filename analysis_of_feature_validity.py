#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:16:44 2020

@author: Filip Paszkiewicz
"""
import numpy as np
# from scipy.stats import norm
# import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

def Analysis_of_Feature_Validity(thigh_mmg_ml,thigh_imu_ml,shank_ml):
    thresh_imu = 0.001
    
    # var_thigh_mmg = np.var(thigh_mmg_ml[:,1:],axis=0)
    # var_thigh_imu = np.var(thigh_imu_ml[:,1:], axis=0)
    # var_shank     = np.var(shank_ml[:,1:], axis=0)
    
    selector_mmg = VarianceThreshold(threshold = 0)
    selector_imu_thigh = VarianceThreshold(threshold = thresh_imu)   
    selector_imu_shank = VarianceThreshold(threshold = thresh_imu)   
     
    selector_mmg.fit(thigh_mmg_ml[:,1:])
    selector_imu_thigh.fit(thigh_imu_ml[:,1:])
    selector_imu_shank.fit(shank_ml[:,1:])
    
    red_mmg = selector_mmg.transform(thigh_mmg_ml[:,1:])
    red_thigh_imu = selector_imu_thigh.transform(thigh_imu_ml[:,1:])
    red_shank = selector_imu_shank.transform(shank_ml[:,1:])
    
    return_mmg = np.column_stack((thigh_mmg_ml[:,0],red_mmg))
    return_thigh_imu = np.column_stack((thigh_imu_ml[:,0],red_thigh_imu))
    return_shank = np.column_stack((shank_ml[:,0],red_shank))
    
    return return_mmg, return_thigh_imu, return_shank