#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:10:19 2020

@author: Filip Paszkiewicz
"""
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from extract_features import Extract_Features
from Base_Function import Open_file_to_array
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def Classifier_SVM(PATH,FS=1000,Flag_Load=True,flag_LDA_comparison=False):
    """ Load Features from saved files """
    if Flag_Load == True:
        thigh_mmg_ml = Open_file_to_array('MMG_test/ML_Features/thigh_mmg.csv')
        thigh_imu_ml = Open_file_to_array('MMG_test/ML_Features/thigh_imu.csv')
        shank_ml = Open_file_to_array('MMG_test/ML_Features/shank.csv')        
    else:
        """ Working Directiory"""
        files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
        files.sort()
        
        # Matrices have to be initialisated for vstack. Don't know how to go around any other way
        number_mmg = 28*5+1
        number_imu = 28*6+1
        number_shank = 28*6+1
        
        walking_thigh_mmg = np.zeros(number_mmg)
        walking_thigh_imu = np.zeros(number_imu)
        walking_shank = np.zeros(number_shank)
        
        stairs_up_thigh_mmg = np.zeros(number_mmg)
        stairs_up_thigh_imu = np.zeros(number_imu)
        stairs_up_shank = np.zeros(number_shank)
        
        stairs_down_thigh_mmg = np.zeros(number_mmg)
        stairs_down_thigh_imu = np.zeros(number_imu)
        stairs_down_shank = np.zeros(number_shank)
        
        sit_thigh_mmg = np.zeros(number_mmg)
        sit_thigh_imu = np.zeros(number_imu)
        sit_shank = np.zeros(number_shank)
        
        stand_thigh_mmg = np.zeros(number_mmg)
        stand_thigh_imu = np.zeros(number_imu)
        stand_shank = np.zeros(number_shank)
            
        for file in files:
            if file.find('walking') !=-1:
                if file.find('thigh') !=-1:
                    temp_mmg, temp_imu = Extract_Features(PATH,file)
                    temp_mmg = np.append([1],temp_mmg)
                    temp_imu = np.append([1],temp_imu)
                    walking_thigh_mmg = np.vstack((walking_thigh_mmg,temp_mmg)) 
                    walking_thigh_imu = np.vstack((walking_thigh_imu,temp_imu))
                elif file.find('shank') !=-1:
                    temp = np.append([1],Extract_Features(PATH,file))
                    walking_shank = np.vstack((walking_shank,temp))
            if file.find('up') !=-1:
                if file.find('thigh') !=-1:
                    temp_mmg, temp_imu = Extract_Features(PATH,file)
                    temp_mmg = np.append([2],temp_mmg)
                    temp_imu = np.append([2],temp_imu)
                    stairs_up_thigh_mmg = np.vstack((stairs_up_thigh_mmg,temp_mmg))
                    stairs_up_thigh_imu = np.vstack((stairs_up_thigh_imu,temp_imu))
                elif file.find('shank') !=-1:
                    temp = np.append([2],Extract_Features(PATH,file))
                    stairs_up_shank = np.vstack((stairs_up_shank,temp))
            if file.find('down') !=-1:
                if file.find('thigh') !=-1:
                    temp_mmg, temp_imu = Extract_Features(PATH,file)
                    temp_mmg = np.append([3],temp_mmg)
                    temp_imu = np.append([3],temp_imu)
                    stairs_down_thigh_mmg = np.vstack((stairs_down_thigh_mmg,temp_mmg))
                    stairs_down_thigh_imu = np.vstack((stairs_down_thigh_imu,temp_imu))
                elif file.find('shank') !=-1:
                    temp = np.append([3],Extract_Features(PATH,file))
                    stairs_down_shank = np.vstack((stairs_down_shank,temp))
            if file.find('sit') !=-1:
                if file.find('thigh') !=-1:
                    temp_mmg, temp_imu = Extract_Features(PATH,file)
                    temp_mmg = np.append([4],temp_mmg)
                    temp_imu = np.append([4],temp_imu)
                    sit_thigh_mmg = np.vstack((sit_thigh_mmg,temp_mmg))
                    sit_thigh_imu = np.vstack((sit_thigh_imu,temp_imu))
                elif file.find('shank') !=-1:
                    temp = np.append([4],Extract_Features(PATH,file))
                    sit_shank = np.vstack((sit_shank,temp))
            if file.find('stand') !=-1:
                if file.find('thigh') !=-1:
                    temp_mmg, temp_imu = Extract_Features(PATH,file)
                    temp_mmg = np.append([5],temp_mmg)
                    temp_imu = np.append([5],temp_imu)
                    stand_thigh_mmg = np.vstack((stand_thigh_mmg,temp_mmg))
                    stand_thigh_imu = np.vstack((stand_thigh_imu,temp_imu))
                elif file.find('shank') !=-1:
                    temp = np.append([5],Extract_Features(PATH,file))
                    stand_shank = np.vstack((stand_shank,temp))
        
        # This is not good code, but I didn't know how to get around initialisationin an other way 
        walking_thigh_mmg = walking_thigh_mmg[1:,:]
        walking_thigh_imu = walking_thigh_imu[1:,:]
        walking_shank = walking_shank[1:,:]
        stairs_up_thigh_mmg = stairs_up_thigh_mmg[1:,:]
        stairs_up_thigh_imu = stairs_up_thigh_imu[1:,:]
        stairs_up_shank = stairs_up_shank[1:,:]
        stairs_down_thigh_mmg = stairs_down_thigh_mmg[1:,:]
        stairs_down_thigh_imu = stairs_down_thigh_imu[1:,:]
        stairs_down_shank = stairs_down_shank[1:,:]
        sit_thigh_mmg = sit_thigh_mmg[1:,:]
        sit_thigh_imu = sit_thigh_imu[1:,:]
        sit_shank = sit_shank[1:,:]
        stand_thigh_mmg = stand_thigh_mmg[1:,:]
        stand_thigh_imu = stand_thigh_imu[1:,:]
        stand_shank = stand_shank[1:,:]
        
        thigh_mmg_ml = np.vstack((walking_thigh_mmg,stairs_up_thigh_mmg,stairs_down_thigh_mmg,sit_thigh_mmg,stand_thigh_mmg))
        thigh_imu_ml = np.vstack((walking_thigh_imu,stairs_up_thigh_imu,stairs_down_thigh_imu,sit_thigh_imu,stand_thigh_imu))
        shank_ml = np.vstack((walking_shank,stairs_up_shank,stairs_down_shank,sit_shank,stand_shank))
        
        np.savetxt('MMG_test/ML_Features/thigh_mmg.csv',thigh_mmg_ml,delimiter=",")
        np.savetxt('MMG_test/ML_Features/thigh_imu.csv',thigh_imu_ml,delimiter=",")
        np.savetxt('MMG_test/ML_Features/shank.csv',shank_ml,delimiter=",")
    
    # Thigh segregation
    y_mmg = thigh_mmg_ml[:,0]
    X_mmg = thigh_mmg_ml[:,1:]
    y_imu = thigh_imu_ml[:,0]
    X_imu = thigh_imu_ml[:,1:]
    
    
    # Shank segregetion 
    # y_shank_ml = shank_ml[:,0]
    # X_shank_ml = shank_ml[:,1:]
    
    Sensor_Fusion = True
    
    if Sensor_Fusion == True:    
        scaler = preprocessing.StandardScaler()# Initialise standarizer
        if np.array_equal(y_imu,y_mmg) == True:
            y_thigh_ml = y_mmg
        X_thigh_ml = np.column_stack((X_mmg,X_imu))
        
        """ Split data into a train and test sets """
        X_train, X_test, y_train, y_test = train_test_split(X_thigh_ml, y_thigh_ml, test_size=0.3, random_state=0)
        
        """ Standariza data before traning """
        scaler.fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)
        
        """ Dimensionality reduction using PCA """
        pca = PCA(n_components = 197)    
        pca.fit(X_train_stand)
        X_t_train = pca.transform(X_train_stand)
        X_t_test = pca.transform(X_test_stand)
        
        SVM_clf = SVC(decision_function_shape ='ovr',kernel ='rbf',gamma = 'scale')
        SVM_clf.fit(X_t_train, y_train)
        print('Calssification accuracy with SVM on test set: %0.3f' % SVM_clf.score(X_t_test, y_test))
        
        if flag_LDA_comparison == True:
            LDA_clf = LinearDiscriminantAnalysis(solver = 'svd')
            LDA_clf.fit(X_train_stand, y_train)
            print('Calssification accuracy with LDA on test set: %0.3f' % LDA_clf.score(X_test_stand, y_test))
        
        # Plot normalized confusion matrix
        labels = ['Walk','SA','SD','Sit','Stand']
        disp = plot_confusion_matrix(SVM_clf,X_t_test,y_test,cmap=plt.cm.Reds, normalize='true',display_labels=labels)
        disp.ax_.set_title("Normalized confusion matrix")
        plt.show()
    




















