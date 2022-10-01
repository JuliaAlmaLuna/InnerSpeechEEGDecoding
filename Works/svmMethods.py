
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import dataLoader as dl
import tensorflow as tf
from tensorflow import keras
import util as ut

from copy import copy as dp
import itertools

#!pip3 install sklearn -q
from tabnanny import verbose
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report

from scipy.fft import rfft, ifft, fftshift, fftfreq
import seaborn as sn
import scipy as sc
from scipy import ndimage
from scipy.signal import hilbert 

import featureExtraction as fExtract

#from Inner_Speech_Dataset.Plotting.ERPs import 
from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Calculate_power_windowed
from Inner_Speech_Dataset.Python_Processing.Utilitys import picks_from_channels
from Inner_Speech_Dataset.Python_Processing.Data_processing import Average_in_frec

class SvmMets():
    
    def __init__(self):
        print("new SvmMets")

    def svmPipeline(self, data_train, data_test, labels_train,
        labels_test, kernel="linear", degree=3, gamma="auto", C = 1, coefs = None): 
        
        if coefs is None:
            coefs = np.zeros([1, data_train.shape[1] * data_train.shape[2]])
        anova_filter = SelectKBest(f_classif, k=10)
        
        clf = make_pipeline( StandardScaler() ,  SVC( #anova_filter/#,
            gamma=gamma, kernel=kernel, degree=degree, verbose=False, C=C, cache_size=1800))

        clf.fit(np.reshape(data_train, [data_train.shape[0], -1]), labels_train)
        predictions = clf.predict(np.reshape(data_test, [data_test.shape[0], -1]))
        
        
        if kernel == "linear":
            #print(classification_report(labels_test, predictions))
            coefs = coefs + clf[:-1].inverse_transform(clf[-1].coef_)
        #print(clf[:-1].inverse_transform(clf[-1].coef_).shape)

        correct = np.zeros(labels_test.shape)
        correctamount=0
        for nr, pred in enumerate(predictions,0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount +=1


        return correctamount/labels_test.shape[0], coefs

    def testSuite(self, data_train, data_test, labels_train, labels_test, name):
        
        bestLin = np.array([["None","dataSet"],[0, "accuracy"],["Linear", "Kernel"] ,[0,"C"]], dtype=object) 
        bestRBF = np.array([["None","dataSet"],[0, "accuracy"],["RBF", "Kernel"] ,[0,"C"]], dtype=object)
        bestSigm = np.array([["None","dataSet"],[0, "accuracy"],["Sigmoid", "Kernel"] ,[0,"C"]], dtype=object)
        allResults = []
        #testing using different kernels, C and degrees. 
        coefs = np.zeros([1, data_train.shape[1] *data_train.shape[2]]) #
        for kernel in ["linear", "rbf", "sigmoid"]:
            if kernel == "linear":
                for C in [0.5]:
                
                    for degree in range(1,2):
                        res, coefs = self.svmPipeline(data_train, data_test, labels_train, 
                        labels_test, degree=degree, kernel = kernel, C = C, coefs=coefs)
                        print("Result for degree {}, kernel {}, C = {}: {}".format(degree, kernel, (C*100//10)/10,  res))
                        allResults.append([name, res, kernel, C])
                        if res > float(bestLin[1,0]):
                            bestLin[:,0] = [name , res, kernel, C]
            else:
                for C in np.linspace(0.5,5,5):
                    # if C < 0.8:
                    #     continue
                    for gamma in ["auto"]:

                        res = self.svmPipeline(data_train, data_test, labels_train, 
                        labels_test, degree=degree, kernel = kernel, gamma=gamma, C = C)
                        print("Result for gamma {}, kernel {}, C = {}: {}".format(gamma, kernel, (C*100//10)/10, res[0]))
                        allResults.append([name, res[0], kernel, C])
                        if kernel == "rbf":
                            if res[0] > float(bestRBF[1,0]):
                                bestRBF[:,0] = [name , res[0], kernel, C]
                        if kernel == "sigmoid":
                            if res[0] > float(bestSigm[1,0]):
                                bestSigm[:,0] = [name , res[0], kernel, C]
                # bestRBF = None
                # bestSigm = None
        coefs = np.reshape(coefs, [128,-1])
        return bestLin, bestRBF, bestSigm, np.array(allResults, dtype=object)