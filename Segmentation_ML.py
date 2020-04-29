#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:58:22 2019

@author: Filip P. Paszkiewicz
"""
# import argparse
import time
from os import listdir
from os.path import isfile, join


from segmentation_staris_up import Segmentation_Staris_Up
from segmentation_staris_down import Segmentation_Staris_Down
from segmentation_walking import Segmentation_Walking
# from segmentation_SS import Segmentation_SS

# parser = argparse.ArgumentParser(description='Segmenting MMG/IMU data for ML')
# parser.add_argument('-P', '--PATH', type=str, metavar='', required=True, help='Path of data files directory')
# parser.add_argument('-F', '--Frequency', type=float, metavar='', required=True, help='Sampling Frequency of the smaples')
# group = parser.add_mutually_exclusive_group()
# group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
# group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
# args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    PATH = 'MMG_test/Non_Amputee_New/'
    
    """ Working Directiory"""
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    files.sort()
    
    sample_1_thigh = 0
    sample_1_shank = 0
    sample_2_thigh = 0
    sample_2_shank = 0
    sample_3_thigh = 0
    sample_3_shank = 0
    # sample_4_thigh = 0
    # sample_5_thigh = 0
    # sample_4_shank = 0
    # sample_5_shank = 0
    
    for i in range(0,len(files),2):
        if files[i].find('up') !=-1: 
            PATH1 = PATH + files[i]
            PATH2 = PATH + files[i+1]
            sample_1_thigh = sample_1_thigh + 1
            sample_1_shank = sample_1_shank + 1
            sample_1_thigh, sample_1_shank = Segmentation_Staris_Up(PATH1,PATH2,sample_1_thigh,sample_1_shank)
        elif files[i].find('down') !=-1: 
            PATH1 = PATH + files[i]
            PATH2 = PATH + files[i+1]
            sample_2_thigh = sample_2_thigh + 1
            sample_2_shank = sample_2_shank + 1
            sample_2_thigh, sample_2_shank = Segmentation_Staris_Down(PATH1,PATH2,sample_2_thigh,sample_2_shank)
        elif files[i].find('walking') !=-1: 
            PATH1 = PATH + files[i]
            PATH2 = PATH + files[i+1]
            sample_3_thigh = sample_3_thigh + 1
            sample_3_shank = sample_3_shank + 1
            sample_3_thigh, sample_3_shank  = Segmentation_Walking(PATH1,PATH2,sample_3_thigh,sample_3_shank)
        # elif files[i].find('sit_to_stand') !=-1: 
        #     PATH1 = PATH + files[i]
        #     PATH2 = PATH + files[i+1]
        #     sample_4_thigh = sample_4_thigh + 1
        #     sample_5_thigh = sample_5_thigh + 1
        #     sample_4_shank = sample_4_shank + 1
        #     sample_5_shank = sample_5_shank + 1
        #     sample_4_thigh, sample_5_thigh, sample_4_shank, sample_5_shank = Segmentation_SS(PATH1,PATH2,sample_4_thigh,sample_5_thigh,sample_4_shank,sample_5_shank)
    
    print("Your program took %0.2f seconds to run." % (time.time() - start_time))
            
            
            
            
            
            
            
            
            
            
            
            