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
        thigh = Open_file_to_array('MMG_test/ML_Features/thigh.csv')
        shank = Open_file_to_array('MMG_test/ML_Features/shank.csv')        
    else:
        """ Working Directiory"""
        files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
        files.sort()
        
        walking_thigh = np.zeros(320)
        walking_shank = np.zeros(175)
        stairs_up_thigh = np.zeros(320)
        stairs_up_shank = np.zeros(175)
        stairs_down_thigh = np.zeros(320)
        stairs_down_shank = np.zeros(175)
        sit_thigh = np.zeros(320)
        sit_shank = np.zeros(175)
        stand_thigh = np.zeros(320)
        stand_shank = np.zeros(175)
            
        for file in files:
            if file.find('walking') !=-1:
                if file.find('thigh') !=-1:
                    temp = np.append([1],Extract_Features(PATH,file))
                    walking_thigh = np.vstack((walking_thigh,temp))
                elif file.find('shank') !=-1:
                    temp = np.append([1],Extract_Features(PATH,file))
                    walking_shank = np.vstack((walking_shank,temp))
            if file.find('up') !=-1:
                if file.find('thigh') !=-1:
                    temp = np.append([2],Extract_Features(PATH,file))
                    stairs_up_thigh = np.vstack((stairs_up_thigh,temp))
                elif file.find('shank') !=-1:
                    temp = np.append([2],Extract_Features(PATH,file))
                    stairs_up_shank = np.vstack((stairs_up_shank,temp))
            if file.find('down') !=-1:
                if file.find('thigh') !=-1:
                    temp = np.append([3],Extract_Features(PATH,file))
                    stairs_down_thigh = np.vstack((stairs_down_thigh,temp))
                elif file.find('shank') !=-1:
                    temp = np.append([3],Extract_Features(PATH,file))
                    stairs_down_shank = np.vstack((stairs_down_shank,temp))
            if file.find('sit') !=-1:
                if file.find('thigh') !=-1:
                    temp = np.append([4],Extract_Features(PATH,file))
                    sit_thigh = np.vstack((sit_thigh,temp))
                elif file.find('shank') !=-1:
                    temp = np.append([4],Extract_Features(PATH,file))
                    sit_shank = np.vstack((sit_shank,temp))
            if file.find('stand') !=-1:
                if file.find('thigh') !=-1:
                    temp = np.append([5],Extract_Features(PATH,file))
                    stand_thigh = np.vstack((stand_thigh,temp))
                elif file.find('shank') !=-1:
                    temp = np.append([5],Extract_Features(PATH,file))
                    stand_shank = np.vstack((stand_shank,temp))
                    
        walking_thigh = walking_thigh[1:,:]
        walking_shank = walking_shank[1:,:]
        stairs_up_thigh = stairs_up_thigh[1:,:]
        stairs_up_shank = stairs_up_shank[1:,:]
        stairs_down_thigh = stairs_down_thigh[1:,:]
        stairs_down_shank = stairs_down_shank[1:,:]
        sit_thigh = sit_thigh[1:,:]
        sit_shank = sit_shank[1:,:]
        stand_thigh = stand_thigh[1:,:]
        stand_shank = stand_shank[1:,:]
        
        thigh = np.vstack((walking_thigh,stairs_up_thigh,stairs_down_thigh,sit_thigh,stand_thigh))
        shank = np.vstack((walking_shank,stairs_up_shank,stairs_down_shank,sit_shank,stand_shank))
        np.savetxt('MMG_test/ML_Features/thigh.csv',thigh,delimiter=",")
        np.savetxt('MMG_test/ML_Features/shank.csv',shank,delimiter=",")
        
    thigh_ml = np.concatenate((thigh[:,0:221],thigh[:,232:]),axis=1)
    shank_ml = np.concatenate((shank[:,0:121],shank[:,127:]),axis=1)
    
    y_thigh_ml = thigh_ml[:,0]
    y_shank_ml = shank_ml[:,0]
    X_thigh_ml = thigh_ml[:,1:]
    X_shank_ml = shank_ml[:,1:]
    
    scaler = preprocessing.StandardScaler()# Initialise standarizer
    
    """ Split data into a train and test sets """
    X_train, X_test, y_train, y_test = train_test_split(X_thigh_ml, y_thigh_ml, test_size=0.4, random_state=0)
    
    """ Standariza data before traning """
    scaler.fit(X_train)
    X_train_stand = scaler.transform(X_train)
    X_test_stand = scaler.transform(X_test)
    
    """ Dimensionality reduction using PCA """
    pca = PCA(n_components = 98)    
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
    disp = plot_confusion_matrix(SVM_clf, X_t_test, y_test, cmap=plt.cm.Reds, normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.show()
    




















