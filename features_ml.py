#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:16:51 2020

@author: Filip Paszkiewicz
"""
import numpy as np

from scipy import signal
from statsmodels.tsa.ar_model import AR
from scipy.stats import kurtosis

def Feature_RMS(data):
    out = np.sqrt(np.mean(data**2,axis=0))
    return out

def Feature_IAV(data):# Integral Absolute Value (discrete)
    out = np.trapz(np.abs(data),axis=0)
    return out

def Feature_MAV(data):# Mean Absolute Value
    out = np.mean(np.abs(data),axis=0)
    return out

def Feature_SSI(data):# Simple Square Integral
    out = np.trapz(data**2,axis=0)
    return out

def Feature_VAR(data):
    out = np.var(data,axis=0)
    return out

def Feature_Kurtosis(data):# Kurtosis
    out = kurtosis(data,axis=0)
    return out

def Feature_AVT3(data):# Absolute Value of the 3rd Temporal Moment
    out = np.abs(np.mean(data**3,axis=0))
    return out

def Feature_AVT4(data):# Absolute Value of the 4th Temporal Moment
    out = np.abs(np.mean(data**4,axis=0))
    return out

def Feature_AVT5(data):# Absolute Value of the 5th Temporal Moment
    out = np.abs(np.mean(data**5,axis=0))
    return out

def Feature_WL(data):# Waveform length
    temp1 = data[1:data.shape[0],:]
    out = np.sum(temp1-data[0:-1,:],axis=0)
    return out

def Feature_AAC(data):# Average amplitude change
    temp1 = data[1:data.shape[0],:]
    out = np.mean(temp1-data[0:-1,:],axis=0)
    return out

def Feature_DAMV(data):# Difference Absolute Mean Value
    temp1 = data[1:data.shape[0],:]
    out = np.mean(np.abs(temp1-data[0:-1,:]),axis=0)
    return out

def Feature_DASDV(data):# Difference Absolute Standard Deviation Value
    temp1 = data[1:data.shape[0],:]
    out = np.sqrt(np.mean((temp1-data[0:-1,:])**2,axis=0))
    return out

def Feature_MMAV1(data):# Modified Mean Absolute Value 1
    long = data.shape[0]
    data[:int(long*0.25),:] = data[:int(long*0.25),:]*0.5
    data[int(long*0.75):,:] = data[int(long*0.75):,:]*0.5
    out = np.mean(np.abs(data),axis=0)
    return out

def Feature_MMAV2(data):# Modified Mean Absolute Value 2
    long = data.shape[0]    
    for i in range(long):
        if i < int(long*0.25):
            data[i,:] = data[i,:]*(4*i)/long
        elif i > int(long*0.75):
            data[i,:] = data[i,:]*(4*(long-i))/long
    out = np.mean(np.abs(data),axis=0)
    return out

def Feature_AR(data):# Auto-regressive model fit
    out = []    
    for i in range(data.shape[1]):
        model = AR(data[:,i])
        model_fit = model.fit(maxlag=4)
        out = np.concatenate((out,model_fit.params))
    return out    

def Feature_LD(data):# Log detector
    out = np.exp(np.mean(np.log(np.abs(data)),axis=0))
    return out

def Feature_Median_F(data,FS):# Medianb frequency
    _, P = signal.periodogram(data,FS,axis=0)
    out = np.sum(P,axis=0)/2
    return out

def Feature_Peak_F(data,FS):# Peak frequency
    _, P = signal.periodogram(data,FS,axis=0)
    out = np.max(P,axis=0)
    return out

def Feature_Mean_P(data,FS):# Mean Power
    _, P = signal.periodogram(data,FS,axis=0)
    out = np.mean(P,axis=0)
    return out

def Feature_Total_P(data,FS):# Total Power
    _, P = signal.periodogram(data,FS,axis=0)
    out = np.sum(P,axis=0)
    return out

def Feature_Mean_F(data,FS):# Mean frequency
    f, P = signal.periodogram(data,FS,axis=0)
    Pf = np.zeros(P.shape)
    for i in range(P.shape[1]):
        Pf[:,i] = P[:,i]*f
    out = np.sum((Pf),axis=0)/np.sum(P,axis=0)
    return out

def Feature_SM1(data,FS):# 1st Spectram Moment
    f, P = signal.periodogram(data,FS,axis=0)
    Pf = np.zeros(P.shape)
    for i in range(P.shape[1]):
        Pf[:,i] = P[:,i]*f
    out = np.sum((Pf),axis=0)
    return out

def Feature_SM2(data,FS):# 2nd Spectram Moment
    f, P = signal.periodogram(data,FS,axis=0)
    Pf = np.zeros(P.shape)
    f = f**2
    for i in range(P.shape[1]):
        Pf[:,i] = P[:,i]*f
    out = np.sum((Pf),axis=0)
    return out

def Feature_SM3(data,FS):# 3rd Spectram Moment
    f, P = signal.periodogram(data,FS,axis=0)
    Pf = np.zeros(P.shape)
    f = f**3
    for i in range(P.shape[1]):
        Pf[:,i] = P[:,i]*f
    out = np.sum((Pf),axis=0)
    return out








    