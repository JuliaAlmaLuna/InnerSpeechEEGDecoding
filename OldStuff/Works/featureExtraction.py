
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

#from Inner_Speech_Dataset.Plotting.ERPs import 
from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Calculate_power_windowed
from Inner_Speech_Dataset.Python_Processing.Utilitys import picks_from_channels
from Inner_Speech_Dataset.Python_Processing.Data_processing import Average_in_frec


class featureEClass():
    
    def __init__(self):
        print("newFClass")
        
    #Plot heatmaps 
    def plotHeatMaps(plotData):
        plt.figure()
        plt.imshow(plotData, cmap="hot", interpolation="nearest")
        plt.show()

    def createListOfDataMixes(self, dataArray, labels, order):

    

        print("Mixing Data")
        dataList = []
        nameList = []
        labelsList = []
        dataNrs = np.arange(len(dataArray))
        #print(dataNrs)
        combos = []
        for L in range(1,len(dataNrs)+1):
            for subsetNr in itertools.combinations(dataNrs, L):
                combos.append(dp(subsetNr))

        print(f"Nr of combinations = {len(combos)}")
        #print(type(combos))
        combos = np.array(combos, dtype=object)
        #print(combos)
       #print(combos.shape)

        for comb in combos:
            
           # print(comb[0])
            nameRow = ""
            dataRo = np.copy(dataArray[comb[0]][0])
            labelsRo = np.copy(labels)
            nameRow = nameRow + "" + dataArray[comb[0]][1]  
            #print(dataArray[comb[0]][0].shape)

            for nr in comb[1:]:

               # print(nr)
                data = np.copy(dataArray[nr][0])
                dataRo = np.concatenate([dataRo, data], axis=1)
               # print(dataRo.shape)
                nameRow = nameRow + "" + dataArray[nr][1]

            dataList.append(dataRo)
            nameList.append(nameRow)
            labelsList.append(labelsRo)

        normShuffledDataList = []
        for x, dataR in enumerate(dataList):
            #featureEClass.plotHeatMaps(dataR[2])
            #print("helloJu")

            nData = np.copy(dataR)
            lData = np.copy(labelsList[x])
     
            #nDataRow = keras.utils.normalize(nData, axis=0, order=2)

            nDataRow = nData
            sDataRow = np.array(self.shuffleSplitData(nDataRow, lData, nameList[x], order=order), dtype=object)
            # sDataRow[0] = keras.utils.normalize(sDataRow[0], axis=0, order=2)
            # sDataRow[1] = keras.utils.normalize(sDataRow[1], axis=0, order=2)
            
            # avg = np.mean(sDataRow[0])
            # std = np.std(sDataRow[0])
            # sDataRow[0] = (sDataRow[0]- avg) / std
            # sDataRow[1] = (sDataRow[1]- avg) / std
            # print(np.mean(sDataRow[0]))
            
            min = np.min(sDataRow[0])
            max = np.max(sDataRow[0])
            sDataRow[0] = (sDataRow[0] - min) / (max-min)
            sDataRow[1] = (sDataRow[1] - min) / (max-min)
            sDataRow[0] = (sDataRow[0] - 0.5)*2
            sDataRow[1] = (sDataRow[1] - 0.5)*2
            
            # print(np.mean(sDataRow[0]))
            # print(np.mean(sDataRow[1]))
            # print(np.min(sDataRow[0]))
            # print(np.min(sDataRow[1]))
            # print(np.max(sDataRow[0]))
            # print(np.max(sDataRow[1]))
            # print(np.var(sDataRow[0]))
            normShuffledDataList.append(sDataRow)
        
        # npDataList = []

        
        # if seed is None:
        #     print("No seed set")


        # print(f"Seed = {seed}")
        
        # for row in dataList[1:]:
        #     npRow = None
        #     first = True
        #     nameRow = ""
            

        #     for column in row:
        #         why = False
        #         if first == True:
                    
        #             npRow = np.copy(column[0])
        #             nameRow = nameRow + "" + column[1]      
        #         else:
        #             npRow = np.concatenate([npRow, np.copy(column[0])], axis = 1 )
        #             nameRow = nameRow + " , " + column[1]
        #         first = False
                
        #     normRow = None
        #     normRow = keras.utils.normalize(npRow, axis=1, order=2)
        #     dataRow = None
        #     dRow = np.copy(normRow)
        #     lRow = np.copy(labels)
            
        #     #This one is okay
        #     dataRow = dp(self.shuffleSplitData(normRow, labels, nameRow, seed=seed))
        #     npDataList.append(dataRow)             
                        
    
        return normShuffledDataList#npDataList



    #Splitting into training and test data
    # print(f"\n\nTesting {name} ")
    #This split seems to work!
    def shuffleSplitData(self, data_t ,labels_t, name, order):

        # if seed != None:
        #         np.random.seed(seed)
                
        # order = np.arange(labels_t.shape[0])
        # np.random.shuffle(order)
        
        # data_train, data_test, labels_train, labels_test = ut.splitData(data_t, labels_t, 0.8)
        
        # print(labels_t.shape)
        # print(data_t.shape)
        # print(order.shape)
        data_s = np.copy(data_t)
        labels_s = np.copy(labels_t)

        data_train = data_s[order[0:int(labels_s.shape[0]*0.8)]]
        data_test = data_s[order[int(labels_s.shape[0]*0.8):]]
        labels_train = labels_s[order[0:int(labels_s.shape[0]*0.8)]]
        labels_test = labels_s[order[int(labels_s.shape[0]*0.8):]]
       
        # print(labels_train)
        # print(labels_train.shape)
        # print(data_train.shape)
        return data_train, data_test, labels_train, labels_test, name


    def getFeatures(self, subject, t_min =2 , t_max = 3, sampling_rate = 256
        , twoDLabels = False):

        #featurearray = [0,1,1,1,1] Not added yet
        """
        Takes in subject nr and array of features: 1 for include 0 for not, True for each 
        one that should be recieved in return array.
        Possible features arranged by order in array:
        FFTCV = Fourier Covariance,
        HRCV = Hilbert real part Covariance,
        HICV = Hilbert imaginary part Covariance 
        CV = Gaussian smootheed EEG covariance
        WCV = Welch Covariance
        TBadded Frequency bands, power bands
        """
      
        nr_of_datasets= 8
        specificSubject = 1
        data, labels = dl.load_multiple_datasets(nr_of_datasets=nr_of_datasets, 
        sampling_rate=sampling_rate, t_min=2, t_max=3, specificSubject=specificSubject,
        twoDLabels=twoDLabels)
        #Names of EEG channels 
        ch_names = ut.get_channelNames()


        # data_p =  ut.get_power_array(data[:,:128,:], sampling_rate, trialSplit=1).squeeze()
        # print("Power band data shape: {}".format(data_p.shape))

        # #Creating freqBandBuckets
        # nr_of_buckets = 15
        # buckets = ut.getFreqBuckets(data, nr_of_buckets=nr_of_buckets)


        # #Getting Freq Data 
        # data_f = ut.data_into_freq_buckets(data[:,:128,:], nr_of_buckets, buckets)
        # print("Freq band bucket separated data shape: {}".format(data_f.shape))


        #Make FFT data
        fftdata = ut.fftData(dp(data))
        print("FFT data shape: {}".format(fftdata.shape))

        #Make covariance of FFT data
        dataFFTCV = np.array(ut.fftCovariance(fftdata))
        #dataFFTCV  = keras.utils.normalize(dataFFTCV , axis=1, order=2)
        print(dataFFTCV.shape)

        #Make Welch data
        welchdata = ut.welchData(dp(data), fs=256, nperseg=256)
        print("Welchdata data shape: {}".format(welchdata.shape))

        #Make covariance of welch data
        dataWCV = np.array(ut.fftCovariance(welchdata))
        #dataWCV  = keras.utils.normalize(dataWCV , axis=1, order=2)

        print(dataWCV.shape)

        #Hilbert data
        dataH = hilbert(dp(data), axis=2, N=256)
        dataHR = dataH.real
        dataHI = dataH.imag
        print("Hilbert real data shape: {}".format(dataHR.shape))

        #Make covariance of Hilber data
        dataHRCV= np.array(ut.fftCovariance(dataHR))
        #dataHRCV = keras.utils.normalize(dataHRCV , axis=1, order=2)
        print(dataHRCV.shape)
        dataHICV= np.array(ut.fftCovariance(dataHI))
        #dataHICV = keras.utils.normalize(dataHICV , axis=1, order=2)

        print(dataHICV.shape)

        #Make covariance of non fft data
        #Try covariance with time allowing for different times. 
        #Maybe each channel becomes 5 channels. zero padded. Moved 1, 2 steps back or forward
        datagauss = ndimage.gaussian_filter1d(dp(data), 5, axis=2)
        dataCV = np.array(ut.fftCovariance(datagauss))
       #dataCV = keras.utils.normalize(dataCV , axis=1, order=2)

        print(dataCV.shape)
        
        datagauss2 = ndimage.gaussian_filter1d(dp(data), 10, axis=2)
        dataCV2 = np.array(ut.fftCovariance(datagauss2))
        #dataCV2 = keras.utils.normalize(dataCV2 , axis=1, order=2)

        print(dataCV2.shape)
        
        datagauss3= ndimage.gaussian_filter1d(dp(data), 2, axis=2)
        dataCV3 = np.array(ut.fftCovariance(datagauss3))
        #dataCV3 = keras.utils.normalize(dataCV3 , axis=1, order=2)
        
        print(dataCV2.shape)



        ### I THINK THIS PART WORKS!

        #Note try PSD 
        
        # featureEClass.plotHeatMaps(dataFFTCV[4])
        # featureEClass.plotHeatMaps(dataCV[4])
        # featureEClass.plotHeatMaps(dataWCV[4])
        # featureEClass.plotHeatMaps(dataHRCV[4])


        # #seed = np.random.random_integers(0,40)
        # for feat in featurearray:
        #     if feat == 1: To be done!


        #sendLabels = dp(labels)
        #SOMETHING IN THIS ONE IS BAD!
        #FOR SURE! Probably Subset.! REDO!
        #mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"], [dataHICV, "dataHICV"], [dataCV, "dataCV"], [dataWCV, "dataWCV"]], labels, seed=seed)
        
        
        
        #mDataList = createListOfDataMixes([[dataHRCV, "dataHRCV"]], labels)
        # for x in mDataList:
        #     print(x[4])
            

            #THis works!

        from copy import deepcopy as deep
        
        
        order = np.arange(labels.shape[0])
        np.random.shuffle(order)


        #mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"],[dataWCV, "dataWCV"]], labels, order=order) #        mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"], [dataHICV, "dataHICV"], [dataCV, "dataCV"], [dataWCV, "dataWCV"]], labels, order=order)
        mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"], [dataCV, "dataCV"], [dataWCV, "dataWCV"]], labels, order=order) #        mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"], [dataHICV, "dataHICV"], [dataCV, "dataCV"], [dataWCV, "dataWCV"]], labels, order=order)
        #mDataList = self.createListOfDataMixes([[dataWCV, "dataWCV"], ], labels, order=order) #        mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"], [dataHICV, "dataHICV"], [dataCV, "dataCV"], [dataWCV, "dataWCV"]], labels, order=order)

        #mDataList = self.createListOfDataMixes([[dataHRCV, "dataHRCV"], [dataWCV, "dataWCV"]], labels, order=order)

        #[dataCV2, "dataCV2"], [dataCV3, "dataCV3"]]

        # wcv1 = deep(dataWCV)
        # wcv2 = deep(dataWCV)
        # hrcv1 = deep(dataHRCV)
        # hrcv2 = deep(dataHRCV)
        # hicv1 = deep(dataHICV)
        # hicv2 = deep(dataHICV)
        # labels1 = deep(labels)
        # labels2 = deep(labels)
    
        # dataT = np.concatenate([wcv1, hrcv1, hicv1], axis =1) #, data
        # #dataT = np.concatenate([np.copy(dataWCV), np.copy(dataHRCV), np.copy(dataHICV)], axis =1) #, data
        # dataT = keras.utils.normalize(dataT, axis=1, order=2)
        # print("asdjulias")
       
        # data_train, data_test, labels_train, labels_test, name = self.shuffleSplitData(dataT, labels1, "hiya", order)
        # #data_train, data_test, labels_train, labels_test = ut.splitData(dataT, labels, 0.8)
        # mDataList = []
        # mDataList.append([data_train, data_test, labels_train, labels_test, "hello"])

        # dataT2 = np.concatenate([wcv2, hrcv2, hicv2], axis =1) #, data
        # #dataT2 = np.concatenate([np.copy(dataWCV), np.copy(dataHRCV), np.copy(dataHICV)], axis =1) #, data
        # dataT2 = keras.utils.normalize(dataT2, axis=1, order=2)
        # print("asdjulias")
        
        # #np.equal(dataT2, dataT)

        # data_train2, data_test2, labels_train2, labels_test2, name = self.shuffleSplitData(dataT2, labels2, "hiya2", order)
        # #data_train, data_test, labels_train, labels_test = ut.splitData(dataT, labels, 0.8)
        # mDataList.append([data_train2, data_test2, labels_train2, labels_test2, "hello2"])


        # dataT = np.concatenate([dataWCV, dataHRCV, dataHICV], axis =1 ) #, data
        # , [dp(dataFFTCV), "dataFFTCV"]
        # mDataList = createListOfDataMixes([[dp(dataWCV), "dataWCV"]], dp(labels))
        

        return mDataList