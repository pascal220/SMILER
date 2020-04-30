#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:58:22 2019

@author: Filip P. Paszkiewicz
"""

import math
import scipy.fftpack
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def Open_file_to_array(path_to_file): # Function takes in a file path, exctracts the file and sends it back as a numpy array
    data = pd.read_csv(path_to_file)
    raw_raw =  data.values # Translate list into a numpy array 

    return raw_raw 


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def Print_Error(raw_raw):
    ds = 0 # Number of dropped packages

    for i in range(1,raw_raw.shape[0]):
        temp = int(raw_raw[i,35]) - int(raw_raw[i-1,35])
        if temp < 0:
            ds = ds + (temp + 255)  # The clock is a single byte (or 8 bite)
        elif temp > 1:
            ds = ds + temp
            
    error = (ds/(len(raw_raw)+ds)) * 100    # Calculate the error
    print('Error rate: ',error,'\n')        # Display calculated error of missed packeges
            
            
            
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""           
# Filltering. Setting up filters and using them    
def butter_bandpass(lowcut, highcut, fs, order): # Define a band-pass filter for given low limit, high limit, samplling frequency and oder
    nyq = 0.5*fs # Find the Nyquist Frequency
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass') # Find the Polynomial for the butter filter band-pass

    return b, a # Returns the numertor and denominator of a desiered band-pass filter

def butter_low(cut_freq,fs,order):
    nyq = 0.5*fs # Find the Nyquist Frequency
    cut = cut_freq/nyq
    b, a = butter(order, cut, btype='low') # Find the Polynomial for the butter filter band-pass
    
    return b, a
    
def butter_filter(data, b, a): # Use pre designed filter. Pass in data to be fillterd, and the numertor and denominator
     y = filtfilt(b, a, data,axis=0)

     return y
 
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# For better understanding of the conversion visit https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def quaternion_to_euler_angle(w, x, y, z): # Translate incoming quaternion in the order W, X, Y, Z to Euler Angles in order X, Y , Z
    ysqr = y * y
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    
    return X, Y, Z

def quaternions_to_euler(raw_quat):
    euler_angle = np.zeros((raw_quat.shape[0],3),np.float) # Euler_angle converted from Quaternions 

    for i in range(raw_quat.shape[0]): # Calculate Euler angles from quaternions
        euler_angle[i,:] = quaternion_to_euler_angle(float(raw_quat[i,0]),float(raw_quat[i,1]),float(raw_quat[i,2]),float(raw_quat[i,3]))

    return euler_angle


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Conversion of High and Low bits into signed 16 or 10 bit numbers. Data converted Accelerometers, Gyroscope and MMG
def acc_raw_to_16bit(raw_data): # Translates a high and low bite to a 16 bit signed integerts 
    acc_16bit = np.zeros((int(raw_data.shape[0]),int(raw_data.shape[1]/2))) # Accelerometer signal after conversion from high-low bits

    k = 0
    for i in range(0,raw_data.shape[1],2): 
        acc_16bit[:,k] = raw_data[:,i] + raw_data[:,i+1]*256 # Transform a high and low bit to a 16-bit number
        k = k + 1
    
    for row in range(acc_16bit.shape[0]):
        for column in range(acc_16bit.shape[1]): 
            if acc_16bit[row,column] > ((2**15) - 1): # Power of two. Keeps numbers as signed 16 bit rather then unsigned 32 bit
                acc_16bit[row,column] = acc_16bit[row,column] - (2**16)

    return acc_16bit*(2/(2**15))  #Scale using range value from the IMU (defalut should be 2G)

def gyro_raw_to_16bit(raw_data_2): # Translates a high and low bite to a 16 bit signed integerts 
    gyro_16bit = np.zeros((int(raw_data_2.shape[0]),int(raw_data_2.shape[1]/2))) # Gyroscope signal after conversion from high-low bits

    k = 0
    for i in range(0,raw_data_2.shape[1],2): 
        gyro_16bit[:,k] = raw_data_2[:,i] + raw_data_2[:,i+1]*256 # Transform a high and low bit to a 16-bit number
        k = k + 1
    
    for row in range(gyro_16bit.shape[0]):
        for column in range(gyro_16bit.shape[1]): 
            if gyro_16bit[row,column] > ((2**15) - 1): # Power of two. Keeps numbers as signed 16 bit rather then unsigned 32 bit
                gyro_16bit[row,column] = gyro_16bit[row,column] - (2**16)

    return gyro_16bit*(2000/(2**15))  #Scale using range value from the IMU (defalut should be 2000 degree/s)

def mmg_raw_to_10bit(raw_data): # Translates a high and low bite to a 10 bit signed integerts 
    mmg_10bit = np.zeros((int(raw_data.shape[0]),int(raw_data.shape[1]/2))) # MMG signal after conversion from high-low bits
    
    k=0
    for i in range(0,raw_data.shape[1],2): 
        mmg_10bit[:,k] = raw_data[:,i]*256 + raw_data[:,i+1] # Transform a high and low bit to a 10-bit number
        k = k + 1
    
    return mmg_10bit*(3.3/(2**10)) #Scale by max value of recorded MMG (3.3V)  


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def find_fft_of_signal(filter_data):# Calculates the Fourier Transform channel wise
    fft_data = scipy.fftpack.fft2(filter_data,axes=0)

    return fft_data


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def Saving_Data_to_CSV(flag,sample_length,filter_mmg,filter_acc): # Takes the already filttered, de-noised data and divides it into 200 ms samples to save them in CSV files
    N = filter_mmg.shape[0]
    
    if flag == 'time':
        temp_mmg = np.zeros((sample_length,8),np.float)
        temp_acc = np.zeros((sample_length,3),np.float)
        to_save_time = np.zeros((sample_length,11),np.float)

        j = 1
        for i in range(0,N,sample_length): # Cycle through data 200ms sampels at a time and save them
            temp_mmg = filter_mmg[i:i+sample_length,:]
            temp_acc = filter_acc[i:i+sample_length,:]

            to_save_time = np.hstack((temp_mmg,temp_acc))

            # Names of data files CSV
            name_time_csv = 'saved\walking_time_' + str(j) + '.csv' 

            # Saving data in CSV (saving to JSON was attempted, but compelx numbers are not JSON serializable)
            np.savetxt(name_time_csv, to_save_time, delimiter=",") # Save time CSV
        
            j += 1
    elif flag == 'freq':
        temp_acc = np.zeros((sample_length,3),np.float)
        temp_mmg = np.zeros((sample_length,8),np.float)
        to_save_freq = np.zeros((sample_length,11),np.float)

        j = 1
        for i in range(0,N,sample_length): # Cycle through data 200ms sampels at a time and save them
            temp_mmg = filter_mmg[i:i+sample_length,:]
            temp_acc = filter_acc[i:i+sample_length,:]
            temp_fft_acc = scipy.fftpack.fft2(temp_acc)
            temp_fft_mmg = scipy.fftpack.fft2(temp_mmg)

            to_save_freq = np.hstack((temp_fft_mmg,temp_fft_acc))

            # Names of data files CSV
            name_freq_csv = 'saved/walking_freq_' + str(j) + '.csv'

            # Saving data in CSV (saving to JSON was attempted, but compelx numbers are not JSON serializable)
            np.savetxt(name_freq_csv, to_save_freq, delimiter=",") # Save frequency CSV
        
            j += 1
            
            
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def Plot_Data(data_thigh,data_shank,FS,HS,sample_thigh,gait_type):
    HS_start = 1
    HS_stop = HS_start + 3
    
    # Plotting muscle data=======================================================================================
    labels = ['Vastus Lateralis','Rectus Femoris','Vastus Medialis','Biceps Femoris','Semitendinosus']
    t = np.linspace(0,data_thigh.shape[0]/FS,data_thigh.shape[0])
    HS_lim_plus = np.max(np.max(data_thigh[HS[HS_start]:HS[HS_stop],:5],axis=0))
    HS_lim_minus = np.min(np.min(data_thigh[HS[HS_start]:HS[HS_stop],:5],axis=0))
    plt.figure(sample_thigh)
    plt.subplot(211)
    plt.title('Muscle activation Human Gait Cycle, Quadriceps - ' + gait_type)
    for i in range(len(labels)-2):  
        plt.plot(t[HS[HS_start]:HS[HS_stop]],data_thigh[HS[HS_start]:HS[HS_stop],i],label=labels[i])
    plt.legend()
    plt.grid(True)
    for i in range(HS_start,HS_stop+1):
        plt.vlines(HS[i]/FS,HS_lim_minus,HS_lim_plus,color='black',linestyle='-.')
    
    plt.subplot(212)
    plt.title('Muscle activation Human Gait Cycle, Hamstring - ' + gait_type)
    for i in range(3,len(labels)):  
        plt.plot(t[HS[HS_start]:HS[HS_stop]],data_thigh[HS[HS_start]:HS[HS_stop],i],label=labels[i])
    plt.legend()
    plt.grid(True)    
    for i in range(HS_start,HS_stop+1):
        plt.vlines(HS[i]/FS,HS_lim_minus,HS_lim_plus,color='black',linestyle='-.')
    
    # Plotting IMU data from Thigh =============================================================================
    labels = ['x-axis','y-axis','z-axis']
    temp = -data_thigh[HS[HS_start]:HS[HS_stop],5:8]
    HS_lim_plus = np.max(np.max(temp,axis=0))
    HS_lim_minus = np.min(np.min(temp,axis=0))
    plt.figure(sample_thigh+1)
    plt.subplot(211)
    plt.title('Gyroscope signals in Human Gait Cycle, Thigh - ' + gait_type)
    for i in range(len(labels)):
        plt.plot(t[HS[HS_start]:HS[HS_stop]],temp[:,i],label=labels[i])
    plt.legend()
    plt.grid(True)
    for i in range(HS_start,HS_stop+1):
        plt.vlines(HS[i]/FS,HS_lim_minus,HS_lim_plus,color='black',linestyle='-.')
    
    temp = -data_thigh[HS[HS_start]:HS[HS_stop],8:]
    HS_lim_plus = np.max(np.max(temp,axis=0))
    HS_lim_minus = np.min(np.min(temp,axis=0))
    plt.subplot(212)
    plt.title('Accelerometer signals in Human Gait Cycle, Thigh - ' + gait_type)
    for i in range(len(labels)):
        plt.plot(t[HS[HS_start]:HS[HS_stop]],temp[:,i],label=labels[i])
    plt.legend()
    plt.grid(True)
    for i in range(HS_start,HS_stop+1):
        plt.vlines(HS[i]/FS,HS_lim_minus,HS_lim_plus,color='black',linestyle='-.')
    
    # Plotting IMU data from Shank ===========================================================================
    labels = ['x-axis','y-axis','z-axis']
    temp = -data_shank[HS[HS_start]:HS[HS_stop],:3]
    HS_lim_plus = np.max(np.max(temp,axis=0))
    HS_lim_minus = np.min(np.min(temp,axis=0))
    plt.figure(sample_thigh+2)
    plt.subplot(211)
    plt.title('Gyroscope signals in Human Gait Cycle, Shank - ' + gait_type)
    for i in range(len(labels)):
        plt.plot(t[HS[HS_start]:HS[HS_stop]],temp[:,i],label=labels[i])
    plt.legend()
    plt.grid(True)
    for i in range(HS_start,HS_stop+1):
        plt.vlines(HS[i]/FS,HS_lim_minus,HS_lim_plus,color='black',linestyle='-.')
    
    temp = -data_shank[HS[HS_start]:HS[HS_stop],3:]
    HS_lim_plus = np.max(np.max(temp,axis=0))
    HS_lim_minus = np.min(np.min(temp,axis=0))
    plt.subplot(212)
    plt.title('Accelerometer signals in Human Gait Cycle, Shank - ' + gait_type)
    for i in range(len(labels)):
        plt.plot(t[HS[HS_start]:HS[HS_stop]],temp[:,i],label=labels[i])
    plt.legend()
    plt.grid(True)
    for i in range(HS_start,HS_stop+1):
        plt.vlines(HS[i]/FS,HS_lim_minus,HS_lim_plus,color='black',linestyle='-.')
    plt.show()