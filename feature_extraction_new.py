import itertools
from copy import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dp
from scipy import ndimage
from scipy.signal import hilbert
import dataLoader as dl
import util as ut


# pylint: disable=C0103
class featureEClass:
    def __init__(self, subject):
        """
        File that holds class handling feature extraction
        and creation of test/train data

        Returns:
            test train data
        """
        self.labels = None
        self.createdFeatureList = []
        # self.goodData = None
        self.subject = subject

        print(f"Feature class for subject {self.subject} created")

    def plotHeatMaps(plotData):
        # Plots heatmaps, used for covariance features.
        # This function does not need to be in this class
        plt.figure()
        plt.imshow(plotData, cmap="hot", interpolation="nearest")
        plt.show()

    def normalizeData(self, trainData, testData, verbose=False):
        # Normalizes train and test data based on trainData max/min
        # This function does not need to be in this class
        min = np.min(trainData)
        max = np.max(trainData)
        trainData = (trainData - min) / (max - min)
        testData = (testData - min) / (max - min)
        trainData = (trainData - 0.5) * 2
        testData = (testData - 0.5) * 2

        if verbose:
            print(np.mean(trainData))
            print(np.mean(testData))
            print(np.min(trainData))
            print(np.min(testData))
            print(np.max(trainData))
            print(np.max(testData))
            print(np.var(trainData))

        return trainData, testData

    def standardizeData(trainData, testData):
        # Standardizes train and test data based on trainData avg/std
        # This function does not need to be in this class
        avg = np.mean(trainData)
        std = np.std(trainData)
        trainData = (trainData - avg) / std
        testData = (testData - avg) / std

        return trainData, testData

    def flattenAllExceptTrial(self, unflatdata):
        flatData = np.reshape(unflatdata, [unflatdata.shape[0], -1])
        return flatData

    def createListOfDataMixes(
        self, featureList, labels, order, maxCombinationAmount, goodDataList
    ):  #
        """
        Mixes the features that are sent in into combinations
        then shuffles and splits them before sending back. Combines the names for each as well

        Args:
            featureList (list): List of features that are to be mixed, split, shuffle and sent back
            labels (np.array): labels for the data
            order (_type_): Order prerandomized that the data is shuffled according to before splitting into train/test

        Returns:
            list normShufflesDataList: List of all mixes of features that have
            been normalized, shuffled and split into training test datasets
        """

        print("Mixing Data")
        dataList = []
        gddataList = []
        nameList = []
        labelsList = []
        dataNrs = np.arange(len(featureList))
        combos = []

        if maxCombinationAmount > len(dataNrs):
            maxCombinationAmount = len(dataNrs)
        for L in range(1, maxCombinationAmount + 1):
            for subsetNr in itertools.combinations(dataNrs, L):
                combos.append(cp(subsetNr))

        print(f"Nr of combinations = {len(combos)}")
        combos = np.array(combos, dtype=object)

        for comb in combos:

            nameRow = ""
            dataRo = self.flattenAllExceptTrial(np.copy(featureList[comb[0]][0]))
            gddataRo = goodDataList[comb[0]]
            labelsRo = np.copy(labels)
            nameRow = nameRow + "-" + featureList[comb[0]][1]

            for nr in comb[1:]:

                data = self.flattenAllExceptTrial(np.copy(featureList[nr][0]))
                gddata = goodDataList[nr]
                dataRo = np.concatenate([dataRo, data], axis=1)
                gddataRo = np.concatenate([gddataRo, gddata], axis=0)
                nameRow = featureList[nr][1] + "-" + nameRow

            dataList.append(dataRo)
            gddataList.append(gddataRo)
            nameList.append(nameRow)
            labelsList.append(labelsRo)

        normShuffledDataList = []
        for x, dataR in enumerate(dataList):

            # Copying to be sure no information is kept between rows, subjects, seeds when running pipeline
            # Probably not needed
            nData = np.copy(dataR)
            lData = np.copy(labelsList[x])

            #  Shuffle the data according to order randomized earlier, and then split it.
            nDataRow = nData
            sDataRow = np.array(
                self.shuffleSplitData(
                    nDataRow, lData, nameList[x], order=order, gdData=gddataList[x]
                ),
                dtype=object,
            )

            # Normalizing data, if standardization is wanted. Use other function called standardizeData
            sDataRow[0], sDataRow[1] = self.normalizeData(
                trainData=sDataRow[0], testData=sDataRow[1]
            )

            normShuffledDataList.append(sDataRow)

        return normShuffledDataList

    def shuffleSplitData(self, data_t, labels_t, name, order, gdData):

        data_s = np.copy(data_t)
        labels_s = np.copy(labels_t)

        data_train = data_s[order[0 : int(labels_s.shape[0] * 0.8)]]
        data_test = data_s[order[int(labels_s.shape[0] * 0.8) :]]
        labels_train = labels_s[order[0 : int(labels_s.shape[0] * 0.8)]]
        labels_test = labels_s[order[int(labels_s.shape[0] * 0.8) :]]

        return data_train, data_test, labels_train, labels_test, name, gdData

    def getFeatures(
        self,
        subject,
        t_min=2,
        t_max=3,
        sampling_rate=256,
        twoDLabels=False,
        maxCombinationAmount=1,
        featureList=[
            False,  # FFT
            False,  # Welch
            False,  # Hilbert
            False,  # Powerbands
            False,  # FFT frequency buckets
            False,  # FFT Covariance
            True,  # Welch Covariance
            True,  # Hilbert Covariance
            False,  # Covariance on smoothed Data
            False,  # Covariance on smoothed Data 2
            # More to be added
        ],
        verbose=True,
    ):

        # featurearray = [0,1,1,1,1] Not added yet
        """
        Takes in subject nr and array of features: 1 for include 0 for not,
        True for each one that should be recieved in return array.
        maxCombinationAmount = Maximum amount of separate features ( for example WCV, FFT) that are combined to form one
        dataset to be trained/tested on. If set to Default = 1, the features are not combined at all.


        Possible features arranged by order in array:
        FFTCV = Fourier Covariance,
        HRCV = Hilbert real part Covariance,
        HICV = Hilbert imaginary part Covariance
        CV = Gaussian smootheed EEG covariance
        WCV = Welch Covariance
        TBadded Frequency bands, power bands
        """

        # Load the Nieto datasets if those are the ones tested. If not, data and labels loaded
        # from some other dataset needs to be
        # In the same shape they are for the rest of the function.
        nr_of_datasets = 1
        specificSubject = subject
        data, self.labels = dl.load_multiple_datasets(
            nr_of_datasets=nr_of_datasets,
            sampling_rate=sampling_rate,
            t_min=t_min,
            t_max=t_max,
            specificSubject=specificSubject,
            twoDLabels=twoDLabels,
        )

        tempData = np.copy(data)
        for fNr, useFeature in enumerate(featureList, 1):

            del tempData
            tempData = np.copy(
                data
            )  # To make sure every feature is created from original data
            if useFeature:
                if fNr == 1:
                    createdFeature = [ut.fftData(tempData), "fftData"]  # fftData

                if fNr == 2:
                    createdFeature = [
                        ut.welchData(tempData, fs=256, nperseg=256),
                        "welchData",
                    ]  # welchData

                if fNr == 3:
                    dataH = hilbert(tempData, axis=2, N=128)
                    createdFeature = [dataH.real, "dataHR"]  # dataHR
                    # dataHI = dataH.imag

                if fNr == 4:
                    # data_p =  ut.get_power_array(data[:,:128,:], sampling_rate,
                    # trialSplit=1).squeeze()
                    # print("Power band data shape: {}".format(data_p.shape))

                    print("Powerbands")

                if fNr == 5:
                    print("Frequency buckets")
                    # #Creating freqBandBuckets
                    # nr_of_buckets = 15
                    # buckets = ut.getFreqBuckets(data, nr_of_buckets=nr_of_buckets)

                if fNr == 6:
                    fftdata = ut.fftData(tempData)
                    createdFeature = [
                        np.array(ut.fftCovariance(fftdata)),
                        "dataFFTCV",
                    ]  # dataFFTCV

                if fNr == 7:
                    welchdata = ut.welchData(tempData, fs=256, nperseg=256)
                    createdFeature = [
                        np.array(ut.fftCovariance(welchdata)),
                        "dataWCV",
                    ]  # dataWCV

                if fNr == 8:
                    dataH = hilbert(tempData, axis=2, N=256)  # dataH
                    dataHR = dataH.real
                    createdFeature = [
                        np.array(ut.fftCovariance(dataHR)),
                        "dataHRCV",
                    ]  # dataHRCV
                    # dataHICV = np.array(ut.fftCovariance(dataHI))

                if fNr == 9:
                    datagauss = ndimage.gaussian_filter1d(tempData, 5, axis=2)
                    createdFeature = [
                        np.array(ut.fftCovariance(datagauss)),
                        "dataCV",
                    ]  # dataCV

                if fNr == 10:
                    datagauss2 = ndimage.gaussian_filter1d(tempData, 10, axis=2)
                    createdFeature = [
                        np.array(ut.fftCovariance(datagauss2)),
                        "dataCV2",
                    ]  # dataCV2

                if verbose:
                    print(f"Data feature nr {fNr} has shape: {createdFeature[0].shape}")

                self.createdFeatureList.append(createdFeature)

        return self.createdFeatureList, self.labels

    def getFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)
        return tempFeatureList

    def getLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels

    # def setgoodData(self, goodData):
    #     self.goodData = goodData

    # def getgoodData(self):
    #     tempgoodData = dp(self.goodData)
    #     return tempgoodData

    # def multiLabels(labels):
    #     mlabels = np.zeros([labels.shape[0], 2])
    #     for ind, label in enumerate(labels):
    #         if label > 3 and label < 8:
    #             mlabels[ind, 1] = 1
    #             mlabels[ind, 0] = label - 4
    #         if label < 4:
    #             mlabels[ind, 1] = 0
    #         if label > 7:
    #             mlabels[ind, 1] = 2
    #             mlabels[ind, 0] = label - 8
    #     labels = mlabels
    #     return labels
    #     # Getting Freq Data
    #     # data_f = ut.data_into_freq_buckets(data[:,:128,:],
    #     # nr_of_buckets, buckets)
    #     # print("Freq band bucket separated data shape: \
    #     # {}".format(data_f.shape))
