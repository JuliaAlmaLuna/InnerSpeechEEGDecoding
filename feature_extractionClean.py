import itertools
from copy import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dp
from scipy import ndimage
from scipy.signal import hilbert
import dataLoader as dl
import util as ut
import glob
import os
from sklearn.model_selection import StratifiedShuffleSplit

# import re
# pylint: disable=C0103


class featureEClass:
    def __init__(
        self,
        subject,
        paradigmName,
        globalSignificance,
        chunkAmount,
        signAll=True,
        signSolo=False,
        uniqueThresh=0.8,
        featureFolder="SavedFeaturesNew",
        chunk=False,
        onlyUniqueFeatures=False,
        useSepSubjFS=False,
    ):
        """
        File that holds class handling feature extraction
        and creation of test/train data

        Returns:
            test train data
        """
        self.paradigmName = paradigmName
        self.labels = None
        self.labelsAux = None
        self.data = None
        self.orderList = None
        self.testNr = None
        self.createdFeatureList = []
        self.maskedFeatureList = []
        self.globalGoodFeatureMask = None
        self.globalSignificance = globalSignificance
        self.subject = subject
        self.featureFolder = featureFolder
        self.chunk = chunk
        self.chunkAmount = chunkAmount
        self.onlyUniqueFeatures = onlyUniqueFeatures
        self.uniqueThresh = uniqueThresh
        self.signAll = signAll
        self.signSolo = signSolo
        self.useSepSubjFS = useSepSubjFS

        if self.signAll or self.signSolo:
            self.onlySign = True
        else:
            self.onlySign = False

        print(f"Feature class for subject {self.subject} created")

    # This class is weirdly written to say the least
    def createMaskedFeatureList(self):
        featureList = self.getFeatureList()
        goodDataMaskList = self.getGlobalGoodFeaturesMask()
        maskedFeatureList = dp(featureList)

        cleanMaskedFeatureList = []
        for feature, mask, maskedFeature in zip(
            featureList, goodDataMaskList, maskedFeatureList
        ):

            # If not onlySign is false, send back all features
            if self.onlySign:
                maskedFeature[0] = self.onlySignData(
                    feature=feature[0], goodData=mask)
                if maskedFeature[0] is not None:
                    cleanMaskedFeatureList.append(maskedFeature)
            else:
                maskedFeature[0] = self.flattenAllExceptTrial(feature[0])
                cleanMaskedFeatureList.append(maskedFeature)

        self.maskedFeatureList = cleanMaskedFeatureList

    def getMaskedFeatureList(self):
        tempMaskedFeatureList = dp(self.maskedFeatureList)
        return tempMaskedFeatureList

    # Also flattens the data. What if i dont use this?
    def onlySignData(self, feature, goodData=None, goodData2=None):

        # One feature at a time. Only feature part.
        # Really should flatten feature way earlier.
        flatFdata = self.flattenAllExceptTrial(feature)

        if self.signAll and self.signSolo:
            if flatFdata[:, [goodData != 0][0] + [goodData2 != 0][0]].shape[1] < 2:
                return None
            onlySignificantFeatures = flatFdata[
                :, [goodData != 0][0] + [goodData2 != 0][0]
            ]

        elif self.signAll:
            if flatFdata[:, np.where(goodData != 0)[0]].shape[1] < 2:
                return None
            onlySignificantFeatures = flatFdata[:, np.where(goodData != 0)[0]]

        elif self.signSolo:
            if flatFdata[:, np.where(goodData2 != 0)[0]].shape[1] < 3:
                return None
            onlySignificantFeatures = flatFdata[:, np.where(goodData2 != 0)[0]]
            # ndata_test = ndata_test[:, np.where(goodData2 != 0)[0]]

        return onlySignificantFeatures

    def saveFeatures(self, name, array):

        saveDir = f"{os.getcwd()}/{self.featureFolder}/sub-{self.subject}-par-{self.paradigmName}"
        if os.path.exists(saveDir) is not True:
            os.makedirs(saveDir)

        np.save(
            f"{saveDir}/{name}",
            array,
        )

    def loadFeatures(self, name):
        svpath = f"{os.getcwd()}/{self.featureFolder}/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(svpath + f"/{name}.npy")
        if len(path) > 0:
            savedFeatures = np.load(path[0], allow_pickle=True)
            return savedFeatures
        else:
            return None

    def loadAnovaMask(self, featurename, maskname):
        name = f"{featurename}{maskname}"
        if self.onlyUniqueFeatures:
            name = f"{name}u{self.uniqueThresh}"

        saveDir = f"{os.getcwd()}/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(saveDir + f"/{name}.npy")
        # print(saveDir)
        # print(name)
        # print(path)
        if len(path) > 0:
            savedAnovaMask = np.load(path[0], allow_pickle=True)
            return savedAnovaMask
        else:
            return None

    def saveAnovaMask(self, featurename, maskname, array):
        name = f"{featurename}{maskname}"

        if self.onlyUniqueFeatures:
            name = f"{name}u{self.uniqueThresh}"

        saveDir = f"{os.getcwd()}/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
        if os.path.exists(saveDir) is not True:
            os.makedirs(saveDir)

        np.save(
            f"{saveDir}/{name}",
            array,
        )

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

    # Usually use Standarscaler instead
    def standardizeData(trainData, testData):
        # Standardizes train and test data based on trainData avg/std
        # This function does not need to be in this class
        avg = np.mean(trainData)
        std = np.std(trainData)
        trainData = (trainData - avg) / std
        testData = (testData - avg) / std

        return trainData, testData

    # Flattens the features, leaves trials undone
    def flattenAllExceptTrial(self, unflatdata):
        flatData = np.reshape(unflatdata, [unflatdata.shape[0], -1])
        return flatData

    def createListOfDataMixes(
        self,
        featureList,
        labels,
        maxCombinationAmount,
        bestFeatures,
        useBestFeaturesTest,
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
        nameList = []
        labelsList = []
        dataNrs = np.arange(len(featureList))
        combos = []

        namesAndIndex = np.array([len(featureList), 2], dtype=object)
        namesAndIndexBestFeatures = np.zeros(
            np.array(bestFeatures, dtype=object).shape)
        bestFeatures = np.array(bestFeatures, dtype=object)
        # print(bestFeatures.shape)
        for index, feature in enumerate(featureList, 0):
            # print(feature[1])
            # print(np.where(bestFeatures == feature[1]))
            namesAndIndex[0] = feature[1]
            namesAndIndex[1] = index
            if np.where(bestFeatures == feature[1])[0].shape[0] > 0:
                row = np.where(bestFeatures == feature[1])[0]
                if maxCombinationAmount > 2:
                    column = np.where(bestFeatures == feature[1])[1]
                    namesAndIndexBestFeatures[row, column] = int(index)
                else:
                    namesAndIndexBestFeatures[row] = int(index)

        # create All combinations of bestFeatures, dvs bara dem
        # Sen ta all combinations, of them and all other values

        if useBestFeaturesTest:
            maxCombinationAmount = maxCombinationAmount - bestFeatures.shape[1]

        if maxCombinationAmount < 1:
            for row in namesAndIndexBestFeatures:
                combos.append(np.array(row))
        else:
            if maxCombinationAmount > len(dataNrs):
                maxCombinationAmount = len(dataNrs)
            for L in range(1, maxCombinationAmount + 1):
                for subsetNr in itertools.combinations(dataNrs, L):
                    if useBestFeaturesTest:
                        for row in namesAndIndexBestFeatures:
                            if all(nr not in row for nr in subsetNr):
                                # if subsetNr not in row:
                                combos.append(
                                    np.array(
                                        np.concatenate(
                                            [np.array(row), cp(subsetNr)], axis=0
                                        ),
                                        dtype=int,
                                    )
                                )
                    else:
                        combos.append(cp(subsetNr))

        print(f"Nr of combinations = {len(combos)}")

        if maxCombinationAmount < 1:
            combos = np.array(combos, dtype=int)

        combos = np.array(combos, dtype=object)

        for comb in combos:  # 10000

            nameRow = ""
            dataRo = np.copy(featureList[comb[0]][0])
            labelsRo = np.copy(labels)
            nameRow = nameRow + featureList[comb[0]][1]

            for nr in comb[1:]:

                data = np.copy(featureList[nr][0])

                dataRo = np.concatenate([dataRo, data], axis=1)

                nameRow = featureList[nr][1] + "_&_" + nameRow

            dataList.append(dataRo)

            nameList.append(nameRow)
            labelsList.append(labelsRo)

        normShuffledDataList = []
        for x, dataR in enumerate(dataList):  # Should be zip

            # Copying to be sure no information is kept between rows, subjects, seeds when running pipeline
            # Probably not needed
            nData = np.copy(dataR)
            lData = np.copy(labelsList[x])

            #  Shuffle the data according to order randomized earlier, and then split it.
            nDataRow = nData
            sDataRow = np.array(
                self.shuffleSplitData(
                    # gdData=gddataList[x]
                    nDataRow,
                    lData,
                    nameList[x],
                ),
                dtype=object,
            )

            # # Normalizing data, if standardization is wanted. Use other function called standardizeData
            # # THIS might be unnecessary since Standardscaler is used later
            # # It does help it seems. Even tho it takes time
            sDataRow[0], sDataRow[1] = self.normalizeData(
                trainData=sDataRow[0], testData=sDataRow[1]
            )

            normShuffledDataList.append(sDataRow)

        return normShuffledDataList

    def shuffleSplitData(self, data_t, labels_t, name):

        data_s = np.copy(data_t)
        labels_s = np.copy(labels_t)

        data_train = data_s[self.getOrder()[0]]
        labels_train = labels_s[self.getOrder()[0]]
        data_test = data_s[self.getOrder()[1]]
        labels_test = labels_s[self.getOrder()[1]]

        del data_s
        del labels_s
        return data_train, data_test, labels_train, labels_test, name  # , gdData

    # TODO Add a loader for bad data, to use in testing
    def loadData(self, t_min, t_max, sampling_rate, twoDLabels, paradigms):
        # Load the Nieto datasets if those are the ones tested. If not, data and labels loaded
        # from some other dataset needs to be
        # In the same shape they are for the rest of the function.
        nr_of_datasets = 1
        specificSubject = self.subject
        self.data, self.labels, self.labelsAux = dl.load_multiple_datasets(
            nr_of_datasets=nr_of_datasets,
            sampling_rate=sampling_rate,
            t_min=t_min,
            t_max=t_max,
            specificSubject=specificSubject,
            twoDLabels=twoDLabels,
            paradigms=paradigms,
        )

        # Makes the size of the trial a size that is divisible with no remainder by chunkAmount
        while True:
            if self.data.shape[2] % self.chunkAmount != 0:
                self.data = self.data[:, :, :-1]
            else:
                break

        return self.data, self.labels

    def chunkCovariance(self, loadedCorrectFeature, featureNameSaved):
        loadedCorrectFeature[0] = np.reshape(
            loadedCorrectFeature[0],
            [
                loadedCorrectFeature[0].shape[0],
                loadedCorrectFeature[0].shape[1] * self.chunkAmount,
                -1,
            ],
        )
        from math import ceil

        covSplitList = []
        onlyFeature = loadedCorrectFeature[0]
        for splitNr in range(int(ceil(self.chunkAmount / 2))):
            splitFeature = onlyFeature[
                :, splitNr:: (int(ceil(self.chunkAmount / 2))), :
            ]  # Split Channels
            covSplitList.append(ut.fftCovariance(splitFeature))
            # covSplits.append(splitFeature)

        rejoinedSplits = np.concatenate(covSplitList, axis=1)
        createdFeature = [
            np.array(rejoinedSplits),
            featureNameSaved,
        ]  # dataFFTCV
        return createdFeature

    def averageChannels(self, preAveragedData):
        groupNames = ["OL", "OZ", "OR", "FL", "FZ",
                      "FR", "CL", "CZ", "CR", "PZ", "OPZ"]
        averagedChannelsData = np.zeros(
            [preAveragedData.shape[0], len(
                groupNames), preAveragedData.shape[2]]
        )
        for avgNr, groupName in enumerate(groupNames):
            cnNrs = ut.channelNumbersFromGroupName(groupName)
            chosenChannels = preAveragedData[:, cnNrs, :]
            averagedChannelsData[:, avgNr, :] = np.mean(chosenChannels, axis=1)
        return averagedChannelsData

    # This covariance works best for features with time
    # If they do not have time. Then this is not necessary
    # I think. Or if they dont have time. Just dont
    def newCovariance(self, preCovFeature, splits=24):
        splits = 24
        postCovFeature = []
        if splits > 1:
            averagedData = self.averageChannels(preCovFeature)
            while True:
                if averagedData.shape[2] % splits == 0:
                    break
                else:
                    averagedData = averagedData[:, :, :-1]
            preCovFeature = np.reshape(
                averagedData,
                [averagedData.shape[0], averagedData.shape[1] * splits, -1],
            )
            print(averagedData.shape[1] * splits)
        for trial in preCovFeature:
            postCovFeature.append(np.cov(trial))
        postCovFeature = np.array(postCovFeature)

        return postCovFeature

    # TODO: Add derivative feature. First, second
    # Then derivate CV
    # Then either BC after first or second step.
    def createFeature(self, featureName, tempData):
        noReshape = False
        createdFeature = None
        if self.chunk:
            if f"cn{self.chunkAmount}" in featureName:
                featureNameSaved = featureName
            else:
                featureNameSaved = f"{featureName}cn{self.chunkAmount}"
        else:
            featureNameSaved = featureName
        splitNr = 24
        if featureName[-2:] == "CV":
            print(featureName)
            print(featureName[:-3])
            preCVFeature = self.loadFeatures(featureName[:-3])[0]
            createdFeature = [
                np.array(self.newCovariance(preCVFeature, splits=splitNr)),
                featureNameSaved,
            ]
        else:
            if featureName == "fftData":
                absFFT, angleFFT = ut.fftData2(tempData)
                createdFeature = [absFFT, featureNameSaved]
                self.saveFeatures(
                    f"angle{featureNameSaved}",
                    [
                        angleFFT,
                        f"angle{featureNameSaved}",
                    ],
                )

            if featureName == "fftData_BC_ifft":
                if self.chunk:
                    # noReshape = True
                    loadedFFTBC = self.loadFeatures(
                        f"fftDatacn{self.chunkAmount}BC")
                    loadedFFTAngle = self.loadFeatures(
                        f"anglefftDatacn{self.chunkAmount}"
                    )

                else:
                    loadedFFTBC = self.loadFeatures("fftData_BC")
                    loadedFFTAngle = self.loadFeatures("anglefftData")
                if loadedFFTBC is not None and loadedFFTAngle is not None:

                    fftdata = loadedFFTBC[0]
                    fftangledata = loadedFFTAngle[0]
                    createdFeature = [
                        np.array(ut.ifftData(fftdata, fftangledata)),
                        featureNameSaved,
                    ]
                else:
                    return None

            if featureName == "welchData":
                if self.chunk:
                    createdFeature = [
                        ut.welchData(
                            tempData,
                            fs=int(256 / self.chunkAmount),
                            nperseg=int(256 / self.chunkAmount),
                        ),
                        featureNameSaved,
                    ]
                else:
                    createdFeature = [
                        ut.welchData(tempData, fs=256, nperseg=256),
                        featureNameSaved,
                    ]
            if featureName == "stftData":
                import scipy.signal as signal
                import math
                wantedShape = splitNr
                arLength = tempData.shape[-1]
                nperseg = math.floor(arLength / (wantedShape - 2))

                stftFeature = abs(signal.stft(tempData, fs=256, window="blackman", boundary="zeros",
                                              padded=True, noverlap=0, axis=-1, nperseg=nperseg)[2])[:, :, :, :splitNr]
                print(stftFeature.shape)
                stftFeatureReshaped = np.reshape(
                    stftFeature, [stftFeature.shape[0], stftFeature.shape[1], -1])
                createdFeature = [stftFeatureReshaped, featureNameSaved]

            if featureName == "hilbertData":
                dataH = hilbert(tempData, axis=2, N=128)
                # print(dataH.real)
                createdFeature = [dataH.real, featureNameSaved]  # dataHR

            if featureName == "Powerbands":
                # data_p =  ut.get_power_array(data[:,:128,:], sampling_rate,
                # trialSplit=1).squeeze()
                # print("Power band data shape: {}".format(data_p.shape))
                pass
            if featureName == "Frequency buckets":
                # #Creating freqBandBuckets
                # nr_of_buckets = 15
                # buckets = ut.getFreqBuckets(data, nr_of_buckets=nr_of_buckets)
                pass
            if featureName == "dataFFTCV":
                fftdata = ut.fftData(tempData)
                createdFeature = [
                    np.array(ut.fftCovariance(fftdata)),
                    featureNameSaved,
                ]
            if featureName == "dataWCV":
                # welchdata = ut.welchData(tempData, fs=256, nperseg=256)
                if self.chunk:
                    welchdata = ut.welchData(
                        tempData,
                        fs=int(256 / self.chunkAmount),
                        nperseg=int(256 / self.chunkAmount),
                    )
                else:
                    welchdata = ut.welchData(tempData, fs=256, nperseg=256)
                createdFeature = [
                    np.array(ut.fftCovariance(welchdata)),
                    featureNameSaved,
                ]
            if featureName == "dataHRCV":
                dataH = hilbert(tempData, axis=2, N=256)  # dataH
                dataHR = dataH.real
                createdFeature = [
                    np.array(ut.fftCovariance(dataHR)),
                    featureNameSaved,
                ]  # dataHRCV

            if featureName == "dataGCV":
                datagauss = ndimage.gaussian_filter1d(tempData, 5, axis=2)
                createdFeature = [
                    np.array(ut.fftCovariance(datagauss)),
                    featureNameSaved,
                ]

            if featureName == "dataGCV2":
                datagauss2 = ndimage.gaussian_filter1d(tempData, 5, axis=2)

                if self.chunk:
                    datagauss2 = np.reshape(
                        datagauss2,
                        [
                            datagauss2.shape[0],
                            -1,
                            int(datagauss2.shape[2] / self.chunkAmount),
                        ],
                    )

                createdFeature = [
                    np.array(ut.fftCovariance(datagauss2)),
                    featureNameSaved,
                ]

            if featureName == "dataCorr1d":
                weights = np.zeros(shape=[20])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[17:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=2, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "1dataCorr2ax1d":
                weights = np.zeros(shape=[20])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[17:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "05dataCorr2ax1d":
                channelHop = 7
                weights = np.zeros(shape=[3 + channelHop + 3])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[channelHop + 3:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "2dataCorr2ax1d":
                channelHop = 30
                weights = np.zeros(shape=[3 + channelHop + 3])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[channelHop + 3:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "3dataCorr2ax1d":
                channelHop = 50
                weights = np.zeros(shape=[3 + channelHop + 3])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[channelHop + 3:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "4dataCorr2ax1d":
                channelHop = 70
                weights = np.zeros(shape=[3 + channelHop + 3])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[channelHop + 3:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]
            if featureName == "5dataCorr2ax1d":
                channelHop = 90
                weights = np.zeros(shape=[3 + channelHop + 3])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[channelHop + 3:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]
            if featureName == "6dataCorr2ax1d":
                channelHop = 110
                weights = np.zeros(shape=[3 + channelHop + 3])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[channelHop + 3:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "dataCorr2ax1dthree":
                weights = np.zeros(shape=[58])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[55:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=1, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "dataCorr1d01s":
                weights = np.zeros(shape=[32])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[29:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=2, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "dataCorr1d02s":
                weights = np.zeros(shape=[58])
                weights[:3] = [0.25, 0.5, 0.25]
                weights[55:] = [0.25, 0.5, 0.25]
                createdFeature = [
                    ndimage.correlate1d(
                        tempData, weights=weights, axis=2, mode="wrap"),
                    featureNameSaved,
                ]

            if featureName == "iFFTdataCorr1d01s-BC":
                loadedCorrectFeature = self.loadFeatures("inverseFFT-BC")

                if loadedCorrectFeature is not None:
                    onlyFeature = loadedCorrectFeature[0]
                    weights = np.zeros(shape=[32])
                    weights[:3] = [0.25, 0.5, 0.25]
                    weights[29:] = [0.25, 0.5, 0.25]
                    createdFeature = [
                        ndimage.correlate1d(
                            onlyFeature, weights=weights, axis=2, mode="wrap"
                        ),
                        featureNameSaved,
                    ]
                else:
                    return None

            if featureName == "iFFTdataCorr1d02s-BC":
                loadedCorrectFeature = self.loadFeatures("inverseFFT-BC")

                if loadedCorrectFeature is not None:
                    onlyFeature = loadedCorrectFeature[0]
                    weights = np.zeros(shape=[58])
                    weights[:3] = [0.25, 0.5, 0.25]
                    weights[55:] = [0.25, 0.5, 0.25]
                    createdFeature = [
                        ndimage.correlate1d(
                            onlyFeature, weights=weights, axis=2, mode="wrap"
                        ),
                        featureNameSaved,
                    ]
                else:
                    return None

            if featureName == "iFFTdataCorr1d005s-BC":
                loadedCorrectFeature = self.loadFeatures("inverseFFT-BC")

                if loadedCorrectFeature is not None:
                    onlyFeature = loadedCorrectFeature[0]
                    weights = np.zeros(shape=[20])
                    weights[:3] = [0.25, 0.5, 0.25]
                    weights[17:] = [0.25, 0.5, 0.25]
                    createdFeature = [
                        ndimage.correlate1d(
                            onlyFeature, weights=weights, axis=2, mode="wrap"
                        ),
                        featureNameSaved,
                    ]
                else:
                    return None

            if featureName == "iFFTdataCorr2ax1d005s-BC":
                loadedCorrectFeature = self.loadFeatures("inverseFFT-BC")

                if loadedCorrectFeature is not None:
                    onlyFeature = loadedCorrectFeature[0]
                    weights = np.zeros(shape=[20])
                    weights[:3] = [0.25, 0.5, 0.25]
                    weights[17:] = [0.25, 0.5, 0.25]
                    createdFeature = [
                        ndimage.correlate1d(
                            onlyFeature, weights=weights, axis=1, mode="wrap"
                        ),
                        featureNameSaved,
                    ]
                else:
                    return None

            if featureName == "dataFFTCV-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"fftDatacn{self.chunkAmount}BC"
                    )

                else:
                    loadedCorrectFeature = self.loadFeatures("fftDataBC")
                if loadedCorrectFeature is not None:
                    if self.chunk:
                        createdFeature = self.chunkCovariance(
                            loadedCorrectFeature, featureNameSaved
                        )
                    else:

                        fftdata = loadedCorrectFeature[0]
                        createdFeature = [
                            np.array(ut.fftCovariance(fftdata)),
                            featureNameSaved,
                        ]  # dataFFTCV
                else:
                    return None

            if featureName == "dataFFTCV2-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"fftDatacn{self.chunkAmount}BC"
                    )

                    if loadedCorrectFeature is not None:
                        # Part below could be a function for all chunked covariance
                        # Shape it so that each split has their own channels
                        # Send in loadedFeature and get back covariance but done with every 3rd/6th channel only
                        createdFeature = self.chunkCovariance(
                            loadedCorrectFeature, featureNameSaved
                        )
                    else:
                        return None
                else:
                    return None

            if featureName == "dataWCV-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"welchDatacn{self.chunkAmount}BC"
                    )
                else:
                    loadedCorrectFeature = self.loadFeatures("welchDataBC")
                if loadedCorrectFeature is not None:

                    if self.chunk:
                        createdFeature = self.chunkCovariance(
                            loadedCorrectFeature, featureNameSaved
                        )
                    else:
                        welchdata = loadedCorrectFeature[0]
                        createdFeature = [
                            np.array(ut.fftCovariance(welchdata)),
                            featureNameSaved,
                        ]  # dataWCV
                else:
                    return None
            if featureName == "dataHRCV-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"dataHRcn{self.chunkAmount}BC"
                    )
                else:
                    loadedCorrectFeature = self.loadFeatures("dataHRBC")
                if loadedCorrectFeature is not None:
                    if self.chunk:
                        createdFeature = self.chunkCovariance(
                            loadedCorrectFeature, featureNameSaved
                        )
                    else:
                        dataHR = loadedCorrectFeature[0]
                        # dataHR = dataH.real
                        createdFeature = [
                            np.array(ut.fftCovariance(dataHR)),
                            featureNameSaved,
                        ]  # dataHRCV
                else:
                    return None

            if featureName == "dataGCV-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"gaussianDatacn{self.chunkAmount}BC"
                    )
                else:
                    loadedCorrectFeature = self.loadFeatures("gaussianDataBC")
                if loadedCorrectFeature is not None:
                    if self.chunk:
                        createdFeature = self.chunkCovariance(
                            loadedCorrectFeature, featureNameSaved
                        )
                    else:
                        gaussiandata = loadedCorrectFeature[0]
                        createdFeature = [
                            np.array(ut.fftCovariance(gaussiandata)),
                            featureNameSaved,
                        ]  # dataFFTCV
                else:
                    return None

            if featureName == "inverseFFTCV-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"inverseFFT{self.chunkAmount}-BC"
                    )
                else:
                    loadedCorrectFeature = self.loadFeatures("inverseFFT-BC")
                if loadedCorrectFeature is not None:
                    if self.chunk:
                        createdFeature = self.chunkCovariance(
                            loadedCorrectFeature, featureNameSaved
                        )
                    else:
                        ifftdata = loadedCorrectFeature[0]
                        createdFeature = [
                            np.array(ut.fftCovariance(ifftdata)),
                            featureNameSaved,
                        ]  # dataFFTCV
                else:
                    return None
            if featureName == "dataGCV2-BC":
                if self.chunk:
                    noReshape = True
                    loadedCorrectFeature = self.loadFeatures(
                        f"gaussianDatacn{self.chunkAmount}BC"
                    )

                    if loadedCorrectFeature is not None:
                        loadedCorrectFeature[0] = np.reshape(
                            loadedCorrectFeature[0],
                            [
                                loadedCorrectFeature[0].shape[0],
                                -1,
                                int(
                                    loadedCorrectFeature[0].shape[2] /
                                    self.chunkAmount
                                ),
                            ],
                        )
                else:
                    loadedCorrectFeature = self.loadFeatures("gaussianDataBC")

                if loadedCorrectFeature is not None:

                    gaussiandata = loadedCorrectFeature[0]
                    createdFeature = [
                        np.array(ut.fftCovariance(gaussiandata)),
                        featureNameSaved,
                    ]  # dataFFTCV
                else:
                    return None

            if featureName == "gausData":
                createdFeature = [
                    ndimage.gaussian_filter1d(tempData, 5, axis=2),
                    featureNameSaved,
                ]

        if self.chunk:
            if noReshape is False:
                # Reshape chunks into more time
                createdFeature[0] = np.reshape(
                    createdFeature[0],
                    [
                        int(createdFeature[0].shape[0] / self.chunkAmount),
                        createdFeature[0].shape[1],
                        -1,
                    ],
                )

        self.saveFeatures(featureNameSaved, createdFeature)
        return createdFeature

    def insert_cn(self, string, index=-2):
        return string[:index] + f"cn{self.chunkAmount}" + string[index:]

    def getFeatures(
        self,
        featureList,
        verbose,
    ):

        if self.data is None:  # Really should load this separately
            raise Exception("Data should not be None")

        self.createdFeatureList = []
        tempData = np.copy(self.data)
        correctedExists = True

        for fNr, useFeature in enumerate(featureList, 1):

            del tempData
            tempData = np.copy(self.data)

            # Splitting the data in time, into chunkAmount of smaller bits, each creating
            # A feature separately
            if self.chunk:
                tempData = np.reshape(
                    tempData,
                    (tempData.shape[0] * self.chunkAmount,
                     tempData.shape[1], -1),
                )

            if useFeature:
                featureName = None
                if fNr == 1:
                    featureName = "fftData"

                if fNr == 2:
                    featureName = "welchData"

                if fNr == 3:
                    featureName = "hilbertData"

                if fNr == 4:
                    print("Powerbands")
                    continue

                if fNr == 5:
                    print("Frequency buckets")
                    continue

                if fNr == 6:
                    featureName = "fftData_CV"

                if fNr == 7:
                    featureName = "welchData_CV"

                if fNr == 8:
                    featureName = "hilbertData_CV"

                if fNr == 9:
                    featureName = "gausData"

                if fNr == 10:
                    continue
                # featureName = "dataGCV2"  # Skip - remove it.

                if fNr == 11:
                    featureName = "normDatacor2x1"

                if fNr == 12:
                    featureName = "fftData_BC"

                if fNr == 13:
                    featureName = "welchData_BC"

                if fNr == 14:
                    featureName = "hilbertData_BC"

                if fNr == 15:
                    featureName = "fftData_BC_CV"

                if fNr == 16:
                    featureName = "welchData_BC_CV"

                if fNr == 17:
                    featureName = "hilbertData_BC_CV"

                if fNr == 18:
                    featureName = "gausData_CV"

                if fNr == 19:
                    featureName = "gausData_CV_BC"

                if fNr == 20:
                    featureName = "gausData_BC"

                if fNr == 21:
                    featureName = "gausData_BC_CV"

                # TODO: ADD BOTH KINDS OF BC. It exists. Just use it!
                # if fNr == 22:
                #     featureName = "dataFFTCV2-BC"

                # if fNr == 23:
                #     featureName = "dataGCV2-BC"

                if fNr == 24:
                    featureName = "normDatacor2x1_BC"

                if fNr == 25:
                    featureName = "fftData_BC_ifft"

                if fNr == 26:
                    featureName = "normDatacor2x2"
                if fNr == 27:
                    featureName = "normDatacor2x3"
                if fNr == 28:
                    featureName = "fftData_BC_ifft_cor2x1"
                if fNr == 29:
                    featureName = "fftData_BC_ifft_cor2x2"
                if fNr == 30:
                    featureName = "fftData_BC_ifft_cor2x3"
                if fNr == 31:
                    featureName = "normDatacor2x2_BC"
                if fNr == 32:
                    featureName = "normDatacor2x3_BC"
                if fNr == 33:
                    featureName = "normDatacor1x1"
                if fNr == 34:
                    featureName = "fftData_BC_ifft_cor1x1"
                if fNr == 35:
                    featureName = "normDatacor1x1_BC"
                if fNr == 36:
                    featureName = "fftData_BC_ifft_CV"
                if fNr == 37:
                    featureName = "anglefftData"
                if fNr == 38:
                    featureName = "anglefftDataBC"
                if fNr == 39:
                    featureName = "normDatacor2x2"
                if fNr == 40:
                    featureName = "normDatacor2x2_BC"
                if fNr == 41:
                    featureName = "normDatacor2x3"
                if fNr == 42:
                    featureName = "normDatacor2x3_BC"
                if fNr == 43:
                    featureName = "normDatacor2x4"
                if fNr == 44:
                    featureName = "normDatacor2x4_BC"
                if fNr == 45:
                    featureName = "normDatacor2x5"
                if fNr == 46:
                    featureName = "normDatacor2x5_BC"
                # if fNr == 47:
                #     featureName = "normDatacor2x6"
                # if fNr == 48:
                #     featureName = "normDatacor2x6_BC"
                if fNr == 51:
                    featureName = "stftData"
                if fNr == 52:
                    featureName = "stftData_BC"
                if fNr == 53:
                    featureName = "stftData_CV"
                if fNr == 54:
                    featureName = "stftData_BC_CV"
                if fNr == 55:
                    featureName = "fftData_CV_BC"
                if fNr == 56:
                    featureName = "welchData_CV_BC"
                if fNr == 57:
                    featureName = "hilbertData_CV_BC"
                if fNr == 58:
                    featureName = "stftData_CV_BC"

                if "baseline" in self.paradigmName or "split" in self.paradigmName:
                    if "BC" in featureName:
                        continue

                if self.chunk:
                    #  if load(featureName) is not None:
                    #       just use loadedFeature
                    #  else:
                    #       if featureName[-2:] == "BC"
                    #           continue/skip
                    #       if featureName[-2:] == "CV"
                    #           load featureName[:-3]
                    #           do CV on feature
                    #           save(featureName)

                    # if "BC" in featureName and "-BC" not in featureName:
                    if featureName[-2:] == "BC":
                        loadedFeature = self.loadFeatures(
                            f"{featureName[0:-2]}cn{self.chunkAmount}BC"
                        )

                    else:
                        loadedFeature = self.loadFeatures(
                            f"{featureName}cn{self.chunkAmount}"
                        )
                else:
                    loadedFeature = self.loadFeatures(featureName)

                if loadedFeature is not None:
                    createdFeature = loadedFeature
                else:
                    if featureName[-2:] == "BC":
                        if self.chunk:
                            if "split" not in self.paradigmName:
                                print(
                                    f"{featureName[0:-2]}cn{self.chunkAmount}BC not exists"
                                )
                        else:
                            if "split" not in self.paradigmName:
                                print(
                                    f"FeatureNameCorrectedNotExist {featureName}")
                        correctedExists = False
                        continue
                    else:

                        createdFeature = self.createFeature(
                            featureName, tempData=tempData
                        )

                if createdFeature is not None:
                    if verbose:
                        if self.chunk:
                            print(
                                f"CHUNKED Data feature nr {fNr} has shape: {createdFeature[0].shape}"
                            )
                        else:
                            # print(featureName)
                            print(
                                f"Data feature nr {fNr} has shape: {createdFeature[0].shape}"
                            )
                    self.createdFeatureList.append(createdFeature)

        return self.createdFeatureList, self.labels, correctedExists

    def getOrigData(self):
        tempData = dp(self.data)
        return tempData

    def getFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)
        return tempFeatureList

    def getFeatureListFlat(self):

        tempFeatureList = dp(self.createdFeatureList)
        for feature in tempFeatureList:
            feature[0] = self.flattenAllExceptTrial(feature[0])

        return tempFeatureList

    def extendFeatureList(self, additionList):
        self.createdFeatureList.extend(additionList)
        # tempFeatureList = dp(self.createdFeatureList)
        # return tempFeatureList

    def getLabelsAux(self):
        tempLabelsAux = dp(self.labelsAux)
        return tempLabelsAux

    # def getTrainFeatureList(self):
    #     tempFeatureList = dp(self.createdFeatureList)

    #     for f in tempFeatureList:
    #         f[0] = f[0][self.order[0 : int(self.labels.shape[0] * 0.8)]]

    #     return tempFeatureList

    # def getTestFeatureList(self):
    #     tempFeatureList = dp(self.createdFeatureList)

    #     for f in tempFeatureList:
    #         f[0] = f[0][self.order[int(self.labels.shape[0] * 0.8) :]]

    #     return tempFeatureList

    # def getTrainLabels(self):
    #     tempLabels = dp(self.labels)
    #     return tempLabels[self.order[0 : int(self.labels.shape[0] * 0.8)]]

    # def getTestLabels(self):
    #     tempLabels = dp(self.labels)
    #     return tempLabels[self.order[int(self.labels.shape[0] * 0.8) :]]

    def getLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels

    def setTestNr(self, testNr):
        self.testNr = testNr

    def setOrder(self, seed, testSize):
        # Set the random order of shuffling for the subject/seed test
        self.orderList = []
        sss = StratifiedShuffleSplit(
            testSize, train_size=0.8, test_size=0.2, random_state=seed
        )

        for train_index, test_index in sss.split(
            X=np.zeros(self.labels.shape[0]), y=self.labels
        ):

            self.orderList.append((train_index, test_index))

    def setGlobalGoodFeaturesMask(self, goodFeatures):
        # Needs to loop through feature mask and save them, using their name which is [1] in the list/tuple

        for feature, mask in zip(self.getFeatureList(), goodFeatures):

            self.saveAnovaMask(
                feature[1], f"sign{self.globalSignificance}", mask)

        self.globalGoodFeatureMask = goodFeatures

    def extendGlobalGoodFeaturesMaskList(self, additionList):
        self.globalGoodFeatureMask.extend(additionList)
        print("Extended global good features mask list")

    def getGlobalGoodFeaturesMask(self):
        # Needs to loop through feature mask and get them, using their name which is [0][1] in the list/tuple
        goodFeatures = []

        oldparadigmName = self.paradigmName
        if self.useSepSubjFS:
            self.paradigmName = f"{self.paradigmName}usingSoloFSubs"

        if self.globalGoodFeatureMask is None:
            print("InGlobalGoodFeatures")
            # print("hola")
            for feature in self.getFeatureList():
                if (
                    self.loadAnovaMask(
                        feature[1], f"sign{self.globalSignificance}")
                    is None
                ):
                    print(feature[1])
                    if self.useSepSubjFS:
                        self.paradigmName = oldparadigmName

                    return None

                goodFeatures.append(
                    self.loadAnovaMask(
                        feature[1], f"sign{self.globalSignificance}")
                )

            self.globalGoodFeatureMask = goodFeatures

        tempFeatureMask = dp(self.globalGoodFeatureMask)
        if self.useSepSubjFS:
            self.paradigmName = oldparadigmName

        return tempFeatureMask

    def getOrder(self):
        return self.orderList[self.testNr]
        # return self.order

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
