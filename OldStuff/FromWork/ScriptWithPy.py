
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import dataLoader as dl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import util2 as ut

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


#Plot heatmaps 
def plotHeatMaps(plotData):
    plt.figure()
    plt.imshow(plotData, cmap="hot", interpolation="nearest")
    plt.show()

def createListOfDataMixes(dataArray, labels, seed = None ):
    print("Mixing Data")
    dataList = []
    for L in range(len(dataArray)+1):
        for subset in itertools.combinations(dataArray, L):
            dataList.append(dp(subset))
    print(f"Nr of combinations = {len(dataList)-1}")
    
    npDataList = []

    
    if seed is None:
        print("No seed set")


    print(f"Seed = {seed}")
    
    for row in dataList[1:]:
        npRow = None
        first = True
        nameRow = ""
        

        for column in row:
            why = False
            if first == True:
                
                npRow = np.copy(column[0])
                nameRow = nameRow + "" + column[1]      
            else:
                npRow = np.concatenate([npRow, np.copy(column[0])], axis = 1 )
                nameRow = nameRow + " , " + column[1]
            first = False
            
        # normRow = None
        # normRow = np.copy(keras.utils.normalize(np.copy(npRow), axis=1, order=2))
        dataRow = None
        dRow = np.copy(npRow)
        lRow = np.copy(labels)
        dataRow = dp(shuffleSplitData(dp(dRow), dp(lRow), nameRow, seed=seed))
        npDataList.append(dataRow)             
                    
   
    return npDataList

def svmPipeline(data_train, data_test, labels_train,
    labels_test, kernel="linear", degree=3, gamma="auto", C = 1, coefs = None): 
    
    if coefs is None:
        coefs = np.zeros([1, data_train.shape[1] * data_train.shape[2]])
    anova_filter = SelectKBest(f_classif, k=10)
    
    clf = make_pipeline( StandardScaler() ,  SVC( #anova_filter/#,
        gamma=gamma, kernel=kernel, degree=degree, verbose=False, C=C))

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

def testSuite(data_train, data_test, labels_train, labels_test, name):
    
    bestLin = np.array([[0, "accuracy"],["None","dataSet"],[0,"C"]], dtype=object) 
    bestRBF = np.array([[0, "accuracy"],["None","dataSet"],[0,"C"]], dtype=object)
    bestSigm = np.array([[0, "accuracy"],["None","dataSet"],[0,"C"]], dtype=object)

    #testing using different kernels, C and degrees. 
    coefs = np.zeros([1, data_train.shape[1] *data_train.shape[2]]) #
    for kernel in ["linear", "rbf", "sigmoid"]:
        if kernel == "linear":
            for C in [0.5]:
            
                for degree in range(1,2):
                    res, coefs = svmPipeline(data_train, data_test, labels_train, 
                    labels_test, degree=degree, kernel = kernel, C = C, coefs=coefs)
                    print("Result for degree {}, kernel {}, C = {}: {}".format(degree, kernel, (C*100//10)/10,  res))
                    if res > float(bestLin[0,0]):
                        bestLin[:,0] = [res, name , C]
        else:
            for C in np.linspace(0.5,3,4):
                for gamma in ["auto"]:

                    res = svmPipeline(data_train, data_test, labels_train, 
                    labels_test, degree=degree, kernel = kernel, gamma=gamma, C = C)
                    print("Result for gamma {}, kernel {}, C = {}: {}".format(gamma, kernel, (C*100//10)/10, res[0]))
                    if kernel == "rbf":
                        if res[0] > float(bestRBF[0,0]):
                            bestRBF[:,0] = [res[0], name , C]
                    if kernel == "sigmoid":
                        if res[0] > float(bestSigm[0,0]):
                            bestSigm[:,0] = [res[0], name , C]
            # bestRBF = None
            # bestSigm = None
    coefs = np.reshape(coefs, [128,-1])
    return bestLin, bestRBF, bestSigm


#Splitting into training and test data
# print(f"\n\nTesting {name} ")
#This split seems to work!
def shuffleSplitData(data_t ,labels_t, name, seed = None):

    if seed != None:
            np.random.seed(seed)
               
    order = np.arange(labels_t.shape[0])
    np.random.shuffle(order)

    data_train = data_t[order[0:int(labels_t.shape[0]*0.8)]]
    data_test = data_t[order[int(labels_t.shape[0]*0.8):]]
    labels_train = labels_t[order[0:int(labels_t.shape[0]*0.8)]]
    labels_test = labels_t[order[int(labels_t.shape[0]*0.8):]]

    return data_train, data_test, labels_train, labels_test, name

for sub in [1,2,4,5,6,7,8]:

    sampling_rate = 256
    nr_of_datasets= 1
    specificSubject = sub
    data, labels = dl.load_multiple_datasets(nr_of_datasets=nr_of_datasets, 
    sampling_rate=sampling_rate, t_min=2, t_max=3, specificSubject=specificSubject,
    twoDLabels=False)
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
    print(dataFFTCV.shape)

    #Make Welch data
    welchdata = ut.welchData(dp(data), fs=256, nperseg=256)
    print("Welchdata data shape: {}".format(welchdata.shape))

    #Make covariance of welch data
    dataWCV = np.array(ut.fftCovariance(welchdata))
    print(dataWCV.shape)

    #Hilbert data
    dataH = hilbert(dp(data), axis=2, N=256)
    dataHR = dataH.real
    dataHI = dataH.imag
    print("Hilbert real data shape: {}".format(dataHR.shape))

    #Make covariance of Hilber data
    dataHRCV= np.array(ut.fftCovariance(dataHR))
    print(dataHRCV.shape)
    dataHICV= np.array(ut.fftCovariance(dataHI))
    print(dataHICV.shape)

    #Make covariance of non fft data
    datagauss = ndimage.gaussian_filter1d(dp(data), 5, axis=2)
    dataCV = np.array(ut.fftCovariance(datagauss))
    print(dataCV.shape)


    #Note try PSD 





    plotHeatMaps(dataFFTCV[2])
    plotHeatMaps(dataCV[2])
    plotHeatMaps(dataWCV[2])
    plotHeatMaps(dataHRCV[2])








    #seed = np.random.random_integers(0,40)
    seed = 4
    mDataList = createListOfDataMixes([[np.copy(dataHRCV), "dataHRCV"], [np.copy(dataHICV), "dataHICV"], [np.copy(dataCV), "dataCV"], [np.copy(dataWCV), "dataWCV"]], labels, seed=seed)
    #mDataList = createListOfDataMixes([[dataHRCV, "dataHRCV"]], labels)

    #, [dp(dataFFTCV), "dataFFTCV"]
    #mDataList = createListOfDataMixes([[dp(dataWCV), "dataWCV"]], dp(labels))
    for x in mDataList:
        print(x[4])


    bestResultsSigm = None
    bestResultsLin = None
    bestResultsRBF = None
    bestResultsTot = None
    bestResultsPerDataSet = []
    first = True
    for data_train, data_test, labels_train, labels_test , name in mDataList:


        print(f"\n Running dataset: {name} \n")

        bestLin, bestRBF, bestSigm = testSuite(data_train, data_test, labels_train, labels_test, name)
        if bestRBF is not None:
            bestResultsPerDataSet.append([bestLin, bestRBF, bestSigm])
            if first == True:
                bestResultsSigm = bestSigm
                bestResultsLin = bestLin
                bestResultsRBF = bestRBF
                bestResultsTot = bestLin
                for results in [bestLin, bestRBF, bestSigm]:
                    if float(results[0,0]) > float(bestResultsTot[0,0]):
                        bestResultsTot = results
                first = False

            if float(bestLin[0,0]) > float(bestResultsLin[0,0]):
                bestResultsLin = bestLin
            if float(bestRBF[0,0]) > float(bestResultsRBF[0,0]):
                bestResultsRBF = bestRBF
            if float(bestSigm[0,0]) > float(bestResultsSigm[0,0]):
                bestResultsSigm = bestSigm
        else:
            bestResultsPerDataSet.append([bestLin])
            if first == True:
                bestResultsLin = bestLin
                first = False
            if float(bestLin[0,0]) > float(bestResultsLin[0,0]):
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


    savearray = np.array([bestResultsPerDataSet, seed, specificSubject], dtype=object)
    from datetime import datetime
    now = datetime.now()
    # Month abbreviation, day and year	
    now_string = now.strftime("D--%d-%m-%Y-T--%H-%M-%S")

    np.save(f"F:/PythonProjects/NietoExcercise-1/SavedResults/savedBestSeed-{seed}-Date-{now_string}",savearray)

    #np.save(f"F:/PythonProjects/NietoExcercise-1/SavedResults/savedBest-{now_string}",savearray)


