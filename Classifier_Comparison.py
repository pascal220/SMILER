#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:02:30 2020

@author: Filip Paszkiewicz
"""
# import argparse
import time

from classifier_SVM import Classifier_SVM

if __name__ == '__main__':
    start_time = time.time()
    PATH = 'MMG_test/ML_Windowed_Data/'
    Classifier = 'SVM'
    
    # if Classifier == 'SVM':
    Classifier_SVM(PATH)
    # elif Classifier = 'CNN':
        # Classifier_CNN()
    
    print("Your program took %0.2f seconds to run." % (time.time() - start_time))