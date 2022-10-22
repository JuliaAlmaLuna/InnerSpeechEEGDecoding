
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import dataLoader as dl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

import featureExtraction as fclass
import svmMethods as svmMet

#from Inner_Speech_Dataset.Plotting.ERPs import 
from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Calculate_power_windowed
from Inner_Speech_Dataset.Python_Processing.Utilitys import picks_from_channels
from Inner_Speech_Dataset.Python_Processing.Data_processing import Average_in_frec


#Set seed either randomly or manually. Needs to be same for each subject.
#fExtract = fclass.featureEClass()
fClassDict = dict()
fmetDict = dict()
for seed in np.arange(1818,5300):
    np.random.seed(seed)
    for sub in [1,2,3,4,5,6,7,8,9]: #1 ,2 as well!
    #for sub in [1]:
        print(f"seed:{seed}-sub:{sub}")
        fClassDict[f"{seed},{sub}"] = fclass.featureEClass() 
        fmetDict[f"{seed},{sub}"] = svmMet.SvmMets()

        #fExtract = fclass.featureEClass()

        
        specificSubject = sub
        mDataList = None
        mDataList = fClassDict[f"{seed},{sub}"].getFeatures(specificSubject, t_min = 2, t_max = 3 , sampling_rate= 256,
        twoDLabels=False)
    


        bestResultsSigm = None
        bestResultsLin = None
        bestResultsRBF = None
        bestResultsTot = None
        bestResultsPerDataSet = []
        allResultsPerDataSet = []
        first = True
        
        
        
        
        for data_train, data_test, labels_train, labels_test , name in mDataList:

            
            print(f"\n Running dataset: {name} \n")

            bestLin, bestRBF, bestSigm, allResults = fmetDict[f"{seed},{sub}"].testSuite(data_train, data_test, labels_train, labels_test, name)
            allResultsPerDataSet.append(allResults)
            if bestRBF is not None:

                bestResultsPerDataSet.append([bestLin, bestRBF, bestSigm])

                if first == True:
                    bestResultsSigm = bestSigm
                    bestResultsLin = bestLin
                    bestResultsRBF = bestRBF
                    bestResultsTot = bestLin
                    for results in [bestLin, bestRBF, bestSigm]:
                        if float(results[1,0]) > float(bestResultsTot[1,0]):
                            bestResultsTot = results
                    first = False

                if float(bestLin[1,0]) > float(bestResultsLin[1,0]):
                    bestResultsLin = bestLin
                if float(bestRBF[1,0]) > float(bestResultsRBF[1,0]):
                    bestResultsRBF = bestRBF
                if float(bestSigm[1,0]) > float(bestResultsSigm[1,0]):
                    bestResultsSigm = bestSigm
            else:
                bestResultsPerDataSet.append([bestLin])
                if first == True:
                    bestResultsLin = bestLin
                    first = False
                if float(bestLin[1,0]) > float(bestResultsLin[1,0]):
                    bestResultsLin = bestLin


        # for res in bestResultsPerDataSet:
        #     print("\nLinear")
        #     print(res[0][:,0])
        #     print(res[0][1,0])
            # print("\nRBF")
            # print(res[1][:,0])
            # print(res[1][1,0])
            # print("\nSigm")
            # print(res[2][:,0])
            # print(res[2][1,0])

        print("\n\n")
        print("Best Linear")
        print(bestResultsLin)
        print("Best RBF")
        print(bestResultsRBF)
        print("Best Gamma")
        print(bestResultsSigm)

        #Saving the results
        savearray = np.array([bestResultsPerDataSet, seed, specificSubject, allResultsPerDataSet], dtype=object)
        
        from datetime import datetime
        now = datetime.now()

        # Month abbreviation, day and year	
        now_string = now.strftime("D--%d-%m-%Y-T--%H-%M-%S")

        np.save(f"F:/PythonProjects/NietoExcercise-1/SavedResults/savedBestSeed-{seed}-Subject-{specificSubject}-Date-{now_string}",savearray)



        #Data no longer needed in this file. Moved

        # sampling_rate = 256
        # nr_of_datasets= 1
        # specificSubject = sub
        # data, labels = dl.load_multiple_datasets(nr_of_datasets=nr_of_datasets, 
        # sampling_rate=sampling_rate, t_min=2, t_max=3, specificSubject=specificSubject,
        # twoDLabels=False)
        # #Names of EEG channels 
        # ch_names = ut.get_channelNames()


        # # data_p =  ut.get_power_array(data[:,:128,:], sampling_rate, trialSplit=1).squeeze()
        # # print("Power band data shape: {}".format(data_p.shape))

        # # #Creating freqBandBuckets
        # # nr_of_buckets = 15
        # # buckets = ut.getFreqBuckets(data, nr_of_buckets=nr_of_buckets)


        # # #Getting Freq Data 
        # # data_f = ut.data_into_freq_buckets(data[:,:128,:], nr_of_buckets, buckets)
        # # print("Freq band bucket separated data shape: {}".format(data_f.shape))


        # #Make FFT data
        # fftdata = ut.fftData(dp(data))
        # print("FFT data shape: {}".format(fftdata.shape))

        # #Make covariance of FFT data
        # dataFFTCV = np.array(ut.fftCovariance(fftdata))
        # print(dataFFTCV.shape)

        # #Make Welch data
        # welchdata = ut.welchData(dp(data), fs=256, nperseg=256)
        # print("Welchdata data shape: {}".format(welchdata.shape))

        # #Make covariance of welch data
        # dataWCV = np.array(ut.fftCovariance(welchdata))
        # print(dataWCV.shape)

        # #Hilbert data
        # dataH = hilbert(dp(data), axis=2, N=256)
        # dataHR = dataH.real
        # dataHI = dataH.imag
        # print("Hilbert real data shape: {}".format(dataHR.shape))

        # #Make covariance of Hilber data
        # dataHRCV= np.array(ut.fftCovariance(dataHR))
        # print(dataHRCV.shape)
        # dataHICV= np.array(ut.fftCovariance(dataHI))
        # print(dataHICV.shape)

        # #Make covariance of non fft data
        # datagauss = ndimage.gaussian_filter1d(dp(data), 5, axis=2)
        # dataCV = np.array(ut.fftCovariance(datagauss))
        # print(dataCV.shape)


        # #Note try PSD 





        # plotHeatMaps(dataFFTCV[2])
        # plotHeatMaps(dataCV[2])
        # plotHeatMaps(dataWCV[2])
        # plotHeatMaps(dataHRCV[2])








        # #seed = np.random.random_integers(0,40)
        # seed = 4
        # mDataList = createListOfDataMixes([[np.copy(dataHRCV), "dataHRCV"], [np.copy(dataHICV), "dataHICV"], [np.copy(dataCV), "dataCV"], [np.copy(dataWCV), "dataWCV"]], labels, seed=seed)
        # #mDataList = createListOfDataMixes([[dataHRCV, "dataHRCV"]], labels)

        # #, [dp(dataFFTCV), "dataFFTCV"]
        # #mDataList = createListOfDataMixes([[dp(dataWCV), "dataWCV"]], dp(labels))
        # for x in mDataList:
        #     print(x[4])