"""
File that holds class handling feature extraction
and creation of test/train data

Returns:
    test train data
"""

import itertools
from copy import copy as dp
import numpy as np
import matplotlib.pyplot as plt
# !pip3 install sklearn -q
from scipy import ndimage
from scipy.signal import hilbert
import dataLoader as dl
import util as ut


# pylint: disable=C0103
class featureEClass():

    def __init__(self):
        print("newFClass")

    # Plot heatmaps

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
        combos = []
        for L in range(1, len(dataNrs) + 1):
            for subsetNr in itertools.combinations(dataNrs, L):
                combos.append(dp(subsetNr))

        print(f"Nr of combinations = {len(combos)}")
        combos = np.array(combos, dtype=object)

        for comb in combos:

            nameRow = ""
            dataRo = np.copy(dataArray[comb[0]][0])
            labelsRo = np.copy(labels)
            nameRow = nameRow + "" + dataArray[comb[0]][1]

            for nr in comb[1:]:

                data = np.copy(dataArray[nr][0])
                dataRo = np.concatenate([dataRo, data], axis=1)
                nameRow = nameRow + "" + dataArray[nr][1]

            dataList.append(dataRo)
            nameList.append(nameRow)
            labelsList.append(labelsRo)

        normShuffledDataList = []
        for x, dataR in enumerate(dataList):

            nData = np.copy(dataR)
            lData = np.copy(labelsList[x])

            # nDataRow = keras.utils.normalize(nData, axis=0, order=2)

            nDataRow = nData
            sDataRow = np.array(self.shuffleSplitData(
                nDataRow, lData, nameList[x], order=order), dtype=object)
            # sDataRow[0] = keras.utils.normalize(sDataRow[0], axis=0, order=2)
            # sDataRow[1] = keras.utils.normalize(sDataRow[1], axis=0, order=2)

            # avg = np.mean(sDataRow[0])
            # std = np.std(sDataRow[0])
            # sDataRow[0] = (sDataRow[0]- avg) / std
            # sDataRow[1] = (sDataRow[1]- avg) / std
            # print(np.mean(sDataRow[0]))

            min = np.min(sDataRow[0])
            max = np.max(sDataRow[0])
            sDataRow[0] = (sDataRow[0] - min) / (max - min)
            sDataRow[1] = (sDataRow[1] - min) / (max - min)
            sDataRow[0] = (sDataRow[0] - 0.5) * 2
            sDataRow[1] = (sDataRow[1] - 0.5) * 2

            # print(np.mean(sDataRow[0]))
            # print(np.mean(sDataRow[1]))
            # print(np.min(sDataRow[0]))
            # print(np.min(sDataRow[1]))
            # print(np.max(sDataRow[0]))
            # print(np.max(sDataRow[1]))
            # print(np.var(sDataRow[0]))
            normShuffledDataList.append(sDataRow)
        """

        Returns:
            list normShufflesDataList: List of all mixes of features that have
            been normalized, shuffled and split into training test datasets
        """
        return normShuffledDataList  # npDataList

    # Splitting into training and test data
    # print(f"\n\nTesting {name} ")
    # This split seems to work!

    def shuffleSplitData(self, data_t, labels_t, name, order):

        data_s = np.copy(data_t)
        labels_s = np.copy(labels_t)

        data_train = data_s[order[0:int(labels_s.shape[0] * 0.8)]]
        data_test = data_s[order[int(labels_s.shape[0] * 0.8):]]
        labels_train = labels_s[order[0:int(labels_s.shape[0] * 0.8)]]
        labels_test = labels_s[order[int(labels_s.shape[0] * 0.8):]]

        return data_train, data_test, labels_train, labels_test, name

    def getFeatures(self, subject, t_min=2, t_max=3,
                    sampling_rate=256, twoDLabels=False):

        # featurearray = [0,1,1,1,1] Not added yet
        """
        Takes in subject nr and array of features: 1 for include 0 for not,
        True for each one that should be recieved in return array.
        Possible features arranged by order in array:
        FFTCV = Fourier Covariance,
        HRCV = Hilbert real part Covariance,
        HICV = Hilbert imaginary part Covariance
        CV = Gaussian smootheed EEG covariance
        WCV = Welch Covariance
        TBadded Frequency bands, power bands
        """

        nr_of_datasets = 1
        specificSubject = subject
        data, labels = dl.load_multiple_datasets(
            nr_of_datasets=nr_of_datasets, sampling_rate=sampling_rate,
            t_min=2, t_max=3, specificSubject=specificSubject,
            twoDLabels=twoDLabels)

        # Names of EEG channels
        # ch_names = ut.get_channelNames()

        # data_p =  ut.get_power_array(data[:,:128,:], sampling_rate,
        # trialSplit=1).squeeze()
        # print("Power band data shape: {}".format(data_p.shape))

        # #Creating freqBandBuckets
        # nr_of_buckets = 15
        # buckets = ut.getFreqBuckets(data, nr_of_buckets=nr_of_buckets)

        # #Getting Freq Data
        # data_f = ut.data_into_freq_buckets(data[:,:128,:],
        # nr_of_buckets, buckets)
        # print("Freq band bucket separated data shape: \
        # {}".format(data_f.shape))

        # print(labels)
        # labels[np.where(labels==2)] = 0
        # labels[np.where(labels==3)] = 1
        # print(labels)
        # Make FFT data'

        fftdata = ut.fftData(dp(data))
        print("FFT data shape: {}".format(fftdata.shape))

        # Make covariance of FFT data
        dataFFTCV = np.array(ut.fftCovariance(fftdata))
        # dataFFTCV  = keras.utils.normalize(dataFFTCV , axis=1, order=2)
        print(dataFFTCV.shape)

        # Make Welch data
        welchdata = ut.welchData(dp(data), fs=256, nperseg=256)
        print("Welchdata data shape: {}".format(welchdata.shape))

        # Make covariance of welch data
        dataWCV = np.array(ut.fftCovariance(dp(welchdata)))
        # dataWCV2 = np.array(ut.fftCorrelation(dp(welchdata)))
        # dataWCV  = keras.utils.normalize(dataWCV , axis=1, order=2)

        print(dataWCV.shape)

        # Hilbert data
        dataH = hilbert(dp(data), axis=2, N=256)
        dataHR = dataH.real
        dataHI = dataH.imag
        print("Hilbert real data shape: {}".format(dataHR.shape))

        # Make covariance of Hilber data
        dataHRCV = np.array(ut.fftCovariance(dataHR))
        # dataHRCV = keras.utils.normalize(dataHRCV , axis=1, order=2)
        print(dataHRCV.shape)
        dataHICV = np.array(ut.fftCovariance(dataHI))
        # dataHICV = keras.utils.normalize(dataHICV , axis=1, order=2)

        print(dataHICV.shape)

        # Make covariance of non fft data
        # Try covariance with time allowing for different times.
        # Maybe each channel becomes 5 channels. zero padded. Moved 1, 2 steps
        # back or forward
        datagauss = ndimage.gaussian_filter1d(dp(data), 5, axis=2)
        # dataCVRoll5= ut.fftCovarianceRoll(datagauss, 5)
        dataCV = np.array(ut.fftCovariance(datagauss))
        # dataCV = keras.utils.normalize(dataCV , axis=1, order=2)

        print(dataCV.shape)

        datagauss2 = ndimage.gaussian_filter1d(dp(data), 10, axis=2)
        dataCV2 = np.array(ut.fftCovariance(datagauss2))
        # dataCV2 = keras.utils.normalize(dataCV2 , axis=1, order=2)

        print(dataCV2.shape)

        datagauss3 = ndimage.gaussian_filter1d(dp(data), 2, axis=2)
        dataCV3 = np.array(ut.fftCovariance(datagauss3))
        # dataCV3 = keras.utils.normalize(dataCV3 , axis=1, order=2)

        print(dataCV3.shape)

        order = np.arange(labels.shape[0])
        np.random.shuffle(order)

        mDataList = self.createListOfDataMixes(
            [[dataWCV, "dataWCV"], [dataCV2, "dataCV2"],
             [dataHRCV, "dataHRCV"]], labels,
            order=order)

        return mDataList

    """
    Class that handles extracting features from data

    Returns:
        features usually : for use in learning/classification
    """
