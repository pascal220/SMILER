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
def Plot_Accelerometer(Fs,filter_acc,fft_acc): # Plot some inertial data 
    N = filter_acc.shape[0] 
    Ts = 1/Fs
    ft_acc = np.linspace(0.0, 1.0/(2.0*Ts), N/2) # Show all frequencies from 0Hz
    time = np.linspace(0,N*Ts,N)                 # Length of recording converted into time using the sampling period
    filter_acc = filter_acc.astype(np.float)
    fft_acc = fft_acc.astype(np.complex)
    time = time.astype(np.float)

    plt.figure(1001) # Plot of the power spectrum (spectral analysis/frequency domain) accelerometer singal which was filtered with a lowpass filter 
    plt.suptitle('Accelerometer - Filtered response in Fourier')
    plt.subplot(311)
    plt.plot(ft_acc, 2.0/N * np.abs(fft_acc[:N//2,0],),'C0', label='x-axis') # Frequencies in  x-axis
    plt.xlim([-0.5,50])
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()
    plt.subplot(312)    
    plt.plot(ft_acc, 2.0/N * np.abs(fft_acc[:N//2,1],),'C1', label='y-axis') # Frequencies in  y-axis
    plt.xlim([-0.5,50])
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()
    plt.subplot(313)
    plt.plot(ft_acc, 2.0/N * np.abs(fft_acc[:N//2,2],),'C2', label='z-axis') # Frequencies in  z-axis
    plt.xlim([-0.5,50])
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()

    plt.figure(1000) # Plot of the Accelerometer singla, filtered by a lowpass filter, through time
    plt.suptitle('Accelerometer - Filtered response in Time')
    plt.subplot(311)
    plt.plot(time,filter_acc[:,0],'C0', label = 'x-axis')
    plt.legend()
    plt.grid(True)
    plt.subplot(312)
    plt.plot(time,filter_acc[:,1],'C1', label = 'y-axis')
    plt.legend()
    plt.grid(True)
    plt.subplot(313)
    plt.plot(time,filter_acc[:,2],'C2', label = 'z-axis')
    plt.xlabel('time/s')
    plt.grid(True)
    plt.legend()

    plt.show()

def Plot_Mechanomyography(Fs,filter_mmg,fft_mmg):  # Plot incoming mmg data (assumes 8 channels)
    N = filter_mmg.shape[0] 
    Ts = 1/Fs
    time = np.linspace(0,N*Ts,N)                 # Length of recording converted into time using the sampling period
    time = time.astype(np.float)
    ft_mmg = np.linspace(1.0, 1.0/(2.0*Ts), (N/2)-1) # Show frequencies from 1Hz
    time = np.linspace(0,N*Ts,N)                     # Length of recording converted into time using the sampling period
    filter_mmg = filter_mmg.astype(np.float)
    fft_mmg = fft_mmg.astype(np.complex)

    plt.figure(999) # Plot of the power spectrum (spectral analysis/frequency domain) MMG singal which was filtered with a bandpass filter 
    plt.suptitle('Mechanomayography - Filtered response in Fourier')
    plt.subplot(421)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,0]),'C0', label='MMG1') # Channel 1
    plt.xlim([1,100])
    plt.ylabel('MMG power amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(422)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,1]),'C1', label='MMG2') # Channel 2
    plt.xlim([1,100])
    plt.grid(True)
    plt.legend()
    plt.subplot(423)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,2]),'C2', label='MMG3') # Channel 3
    plt.xlim([1,100])
    plt.grid(True)
    plt.legend()
    plt.subplot(424)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,3]),'C3', label='MMG4') # Channel 4
    plt.xlim([1,100])
    plt.grid(True)
    plt.legend()
    plt.subplot(425)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,4]),'C4', label='MMG5') # Channel 5
    plt.xlim([1,100])
    plt.ylabel('MMG power amplitude')
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()
    plt.subplot(426)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,5]),'C5', label='MMG6') # Channel 6
    plt.xlim([1,100])
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()
    plt.subplot(427)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,6]),'C6', label='MMG7') # Channel 7
    plt.xlim([1,100])
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()
    plt.subplot(428)
    plt.plot(ft_mmg, 2.0/N * np.abs(fft_mmg[1:N//2,7]), 'C7', label='MMG8') # Channel 8
    plt.xlim([1,100])
    plt.xlabel('Hz')
    plt.grid(True)
    plt.legend()

    plt.figure(998) # Plot of the MMG singla, filtered by a bandpass filter, through time
    plt.suptitle('Mechanomayography - Filtered response in Time')
    plt.subplot(421)
    plt.plot(time,filter_mmg[:,0],'C0', label='MMG1') # Channel 1
    plt.ylabel('MMG amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(422)
    plt.plot(time,filter_mmg[:,1],'C1', label='MMG2') # Channel 2
    plt.grid(True)
    plt.legend()
    plt.subplot(423)
    plt.plot(time,filter_mmg[:,2],'C2', label='MMG3') # Channel 3
    plt.ylabel('MMG amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(424)
    plt.plot(time,filter_mmg[:,3],'C3', label='MMG4') # Channel 4
    plt.grid(True)
    plt.legend()
    plt.subplot(425)
    plt.plot(time,filter_mmg[:,4],'C4', label='MMG5') # Channel 5
    plt.ylabel('MMG amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(426)
    plt.plot(time,filter_mmg[:,5],'C5', label='MMG6') # Channel 6
    plt.grid(True)
    plt.legend()
    plt.subplot(427)
    plt.plot(time,filter_mmg[:,6],'C6', label='MMG7') # Channel 7
    plt.ylabel('MMG amplitude')
    plt.xlabel('time/s')
    plt.grid(True)
    plt.legend()
    plt.subplot(428)
    plt.plot(time,filter_mmg[:,7], 'C7', label='MMG8') # Channel 8
    plt.xlabel('time/s')
    plt.grid(True)
    plt.legend()

    plt.show()

def Plot_Differnece(Fs,acc_16bit,filter_acc,mmg_10bit,filter_mmg): # Plot the difference between filtered and raw MMG and ACC data
    N = filter_acc.shape[0]
    Ts = 1/Fs
    time = np.linspace(0,N*Ts,N) # Length of recording converted into time using the sampling period
    acc_16bit = acc_16bit.astype(np.float)
    mmg_10bit = mmg_10bit.astype(np.float)
    filter_acc = filter_acc.astype(np.float)  
    filter_mmg = filter_mmg.astype(np.float)
    time = time.astype(np.float)

    plt.figure(997)
    plt.suptitle('Mechanomayography - Comaprison between Filtered and Raw signal')
    plt.subplot(421)
    plt.plot(time,filter_mmg[:,0],'C0', label='MMG1 - Filtered') # Channel 1
    plt.plot(time,mmg_10bit[:,0],'C7', label='MMG1 - Raw') # Channel 1
    plt.ylabel('MMG amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(422)
    plt.plot(time,filter_mmg[:,1],'C1', label='MMG2 - Filtered') # Channel 2
    plt.plot(time,mmg_10bit[:,1],'C6', label='MMG2 - Raw') # Channel 2
    plt.grid(True)
    plt.legend()
    plt.subplot(423)
    plt.plot(time,filter_mmg[:,2],'C2', label='MMG3 - Filtered') # Channel 3
    plt.plot(time,mmg_10bit[:,2],'C5', label='MMG3 - Raw') # Channel 3
    plt.ylabel('MMG amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(424)
    plt.plot(time,filter_mmg[:,3],'C3', label='MMG4 - Filtered') # Channel 4
    plt.plot(time,mmg_10bit[:,3],'C4', label='MMG4 - Raw') # Channel 4
    plt.grid(True)
    plt.legend()
    plt.subplot(425)
    plt.plot(time,filter_mmg[:,4],'C4', label='MMG5 - Filtered') # Channel 5
    plt.plot(time,mmg_10bit[:,4],'C3', label='MMG5 - Raw') # Channel 5
    plt.ylabel('MMG amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(426)
    plt.plot(time,filter_mmg[:,5],'C5', label='MMG6 - Filtered') # Channel 6
    plt.plot(time,mmg_10bit[:,5],'C2', label='MMG6 - Raw') # Channel 6
    plt.grid(True)
    plt.legend()
    plt.subplot(427)
    plt.plot(time,filter_mmg[:,6],'C6', label='MMG7 - Filtered') # Channel 7
    plt.plot(time,mmg_10bit[:,6],'C1', label='MMG7 - Raw') # Channel 7
    plt.ylabel('MMG amplitude')
    plt.xlabel('time/s')
    plt.grid(True)
    plt.legend()
    plt.subplot(428)
    plt.plot(time,filter_mmg[:,7], 'C7', label='MMG8 - Filtered') # Channel 8
    plt.plot(time,mmg_10bit[:,7], 'C0', label='MMG8 - Raw') # Channel 8
    plt.xlabel('time/s')
    plt.grid(True)
    plt.legend()

    plt.figure(995)
    plt.suptitle('Accelerometer - Comaprison between Filtered and Raw signal')
    plt.subplot(311)
    plt.plot(time,filter_acc[:,0],'C0', label = 'x-axis - filtered')
    plt.plot(time,acc_16bit[:,0],'C3', label = 'x-axis - raw')
    plt.legend()
    plt.grid(True)
    plt.subplot(312)
    plt.plot(time,filter_acc[:,1],'C1', label = 'y-axis - filtered')
    plt.plot(time,acc_16bit[:,1],'C4', label = 'y-axis - raw')
    plt.legend()
    plt.grid(True)
    plt.subplot(313)
    plt.plot(time,filter_acc[:,2],'C2', label = 'z-axis - filtered')
    plt.plot(time,acc_16bit[:,2],'C5', label = 'z-axis - raw')
    plt.xlabel('time/s')
    plt.legend()
    plt.grid(True)

    plt.show()

