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

# Something might have happened at import
# pylint: disable=C0103


class featureEClass:
    def __init__(
        self,
        subject,
        paradigmName,
        globalSignificance,
        chunkAmount,
        featureFolder="SavedFeatures",
        chunk=False,

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
        self.order = None
        self.createdFeatureList = []
        self.globalGoodFeatureMask = None
        self.globalSignificance = globalSignificance
        self.subject = subject
        self.featureFolder = featureFolder
        self.chunk = chunk
        self.chunkAmount = chunkAmount

        print(f"Feature class for subject {self.subject} created")

    def saveFeatures(self, name, array):

        import os

        saveDir = f"F:/PythonProjects/NietoExcercise-1/{self.featureFolder}/sub-{self.subject}-par-{self.paradigmName}"
        if os.path.exists(saveDir) is not True:
            os.makedirs(saveDir)

        np.save(
            f"{saveDir}/{name}",
            array,
        )

    def loadFeatures(self, name):
        svpath = f"F:/PythonProjects/NietoExcercise-1/{self.featureFolder}/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(svpath + f"/{name}.npy")
        if len(path) > 0:
            savedFeatures = np.load(path[0], allow_pickle=True)
            return savedFeatures
        else:
            return None

    def loadAnovaMask(self, featurename, maskname):
        name = f"{featurename}{maskname}"
        curSavePath = f"F:/PythonProjects/NietoExcercise-1/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(curSavePath + f"/{name}.npy")
        if len(path) > 0:
            savedAnovaMask = np.load(path[0], allow_pickle=True)
            return savedAnovaMask
        else:
            return None

    def saveAnovaMask(self, featurename, maskname, array):
        name = f"{featurename}{maskname}"
        import os

        saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
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
        self, featureList, labels, order, maxCombinationAmount
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
            dataRo = self.flattenAllExceptTrial(
                np.copy(featureList[comb[0]][0]))
            if self.globalGoodFeatureMask is not None:
                gddataRo = self.globalGoodFeatureMask[comb[0]]
            labelsRo = np.copy(labels)
            nameRow = nameRow + "-" + featureList[comb[0]][1]

            for nr in comb[1:]:

                data = self.flattenAllExceptTrial(np.copy(featureList[nr][0]))
                if self.globalGoodFeatureMask is not None:
                    gddata = self.globalGoodFeatureMask[nr]
                dataRo = np.concatenate([dataRo, data], axis=1)
                if self.globalGoodFeatureMask is not None:
                    gddataRo = np.concatenate([gddataRo, gddata], axis=0)
                nameRow = featureList[nr][1] + "-" + nameRow

            dataList.append(dataRo)
            if self.globalGoodFeatureMask is not None:
                gddataList.append(gddataRo)
            else:
                gddataList.append(None)
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
                    nDataRow, lData, nameList[x], order=order, gdData=gddataList[x]
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

    def shuffleSplitData(self, data_t, labels_t, name, order, gdData):

        data_s = np.copy(data_t)
        labels_s = np.copy(labels_t)

        data_train = data_s[order[0: int(labels_s.shape[0] * 0.8)]]
        data_test = data_s[order[int(labels_s.shape[0] * 0.8):]]
        labels_train = labels_s[order[0: int(labels_s.shape[0] * 0.8)]]
        labels_test = labels_s[order[int(labels_s.shape[0] * 0.8):]]

        return data_train, data_test, labels_train, labels_test, name, gdData

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
        # print(labelsAux)
        # print(labelsAux[:, 1])  # Class
        # print(labelsAux[:, 2])  # Cond
        # print(labelsAux[:, 3])  # Session

        # Make data divisible by chunkAmount

        while True:
            if self.data.shape[2] % self.chunkAmount != 0:
                self.data = self.data[:, :, :-1]
            else:
                break

        return self.data, self.labels

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

        if featureName == "fftData":
            createdFeature = [ut.fftData(tempData), featureNameSaved]

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
        if featureName == "dataHR":
            dataH = hilbert(tempData, axis=2, N=128)
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
        if featureName == "dataCV":
            datagauss = ndimage.gaussian_filter1d(tempData, 5, axis=2)
            createdFeature = [
                np.array(ut.fftCovariance(datagauss)),
                featureNameSaved,
            ]
        if featureName == "dataCV2":
            datagauss2 = ndimage.gaussian_filter1d(tempData, 10, axis=2)
            createdFeature = [
                np.array(ut.fftCovariance(datagauss2)),
                featureNameSaved,
            ]
        if featureName == "dataCorr1d":
            weights = np.zeros(shape=[20])
            weights[:3] = 1
            weights[16:] = 1
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=2),
                featureNameSaved,
            ]
        if featureName == "dataFFTCV-BC":
            if self.chunk:
                noReshape = True
                loadedCorrectFeature = self.loadFeatures(
                    f"fftDatacn{self.chunkAmount}BC"
                )
            else:
                loadedCorrectFeature = self.loadFeatures("fftDataBC")
            if loadedCorrectFeature is not None:
                print("Hey")
                fftdata = loadedCorrectFeature[0]
                createdFeature = [
                    np.array(ut.fftCovariance(fftdata)),
                    featureNameSaved,
                ]  # dataFFTCV
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
                dataHR = loadedCorrectFeature[0]
                # dataHR = dataH.real
                createdFeature = [
                    np.array(ut.fftCovariance(dataHR)),
                    featureNameSaved,
                ]  # dataHRCV
            else:
                return None

        if self.chunk:
            if noReshape is False:
                # print(f"CHUNKED Data feature has shape: {createdFeature[0].shape}")
                # print("SHAPING Back chunked data")
                print("ho")
                createdFeature[0] = np.reshape(
                    createdFeature[0],
                    [
                        int(createdFeature[0].shape[0] / self.chunkAmount),
                        createdFeature[0].shape[1],
                        -1,
                    ],
                )
            # print(f"CHUNKED Data feature has shape: {createdFeature[0].shape}")
            # featureName = f"{featureName}cn{self.chunkAmount}"

        self.saveFeatures(featureNameSaved, createdFeature)
        return createdFeature

    def insert_cn(self, string, index=-2):
        return string[:index] + f'cn{self.chunkAmount}' + string[index:]

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
            False,  # Correlate1d
            # More to be added
            # Add a feature that tells you if it is inner, pronounced or visualized
            # Make it shape(128,1) with maybe 10 ones at specific points if it is
            # each one of these
            # Maybe also make one similar with 10 for each subject
        ],
        verbose=True,
        paradigms=[
            [["Inner"], ["Inner"], ["Inner"], ["Inner"]],
            [
                ["Up"],
                ["Down"],
                ["Right"],
                ["Left"],
            ],
        ],
    ):

        if self.data is None:  # Really should load this separately
            print("Using load here")
            self.data, self.labels = self.loadData(
                t_min, t_max, sampling_rate, twoDLabels, paradigms
            )

        self.createdFeatureList = []
        tempData = np.copy(self.data)
        correctedExists = True
        # print(self.chunk)
        for fNr, useFeature in enumerate(featureList, 1):

            del tempData
            tempData = np.copy(self.data)
            # print(tempData.shape)  # made shorter
            if self.chunk:
                tempData = np.reshape(
                    tempData,
                    (tempData.shape[0] * self.chunkAmount,
                     tempData.shape[1], -1),
                )
            # print(tempData.shape)
            # To make sure every feature is created from original data
            if useFeature:
                featureName = None
                if fNr == 1:
                    featureName = "fftData"

                if fNr == 2:
                    featureName = "welchData"

                if fNr == 3:
                    featureName = "dataHR"

                if fNr == 4:
                    print("Powerbands")
                    continue

                if fNr == 5:
                    print("Frequency buckets")
                    continue

                if fNr == 6:
                    featureName = "dataFFTCV"

                if fNr == 7:
                    featureName = "dataWCV"

                if fNr == 8:
                    featureName = "dataHRCV"

                if fNr == 9:
                    featureName = "dataCV"

                if fNr == 10:
                    featureName = "dataCV2"

                if fNr == 11:
                    featureName = "dataCorr1d"

                if fNr == 12:
                    featureName = "dataFFTCV-BC"

                if fNr == 13:
                    featureName = "dataWCV-BC"

                if fNr == 14:
                    featureName = "dataHRCV-BC"

                if fNr == 15:
                    featureName = "fftDataBC"

                if fNr == 16:
                    featureName = "welchDataBC"

                if fNr == 17:
                    featureName = "dataHRBC"

                if fNr == 18:
                    featureName = "dataCVBC"

                # if fNr == 18:
                #     featureName = "Chunk"
                if self.chunk:
                    if "BC" in featureName and "-BC" not in featureName:
                        loadedFeature = self.loadFeatures(
                            self.insert_cn(featureName))

                    else:
                        loadedFeature = self.loadFeatures(
                            f"{featureName}cn{self.chunkAmount}"
                        )
                else:
                    loadedFeature = self.loadFeatures(featureName)

                if loadedFeature is not None:
                    createdFeature = loadedFeature
                else:
                    if "BC" in featureName and "-BC" not in featureName:
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

    def extendFeatureList(self, additionList):
        self.createdFeatureList.extend(additionList)
        # tempFeatureList = dp(self.createdFeatureList)
        # return tempFeatureList

    def getLabelsAux(self):
        tempLabelsAux = dp(self.labelsAux)
        return tempLabelsAux

    def getTrainFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)

        for f in tempFeatureList:
            f[0] = f[0][self.order[0: int(self.labels.shape[0] * 0.8)]]

        return tempFeatureList

    def getTestFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)

        for f in tempFeatureList:
            f[0] = f[0][self.order[int(self.labels.shape[0] * 0.8):]]

        return tempFeatureList

    def getTrainLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels[self.order[0: int(self.labels.shape[0] * 0.8)]]

    def getTestLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels[self.order[int(self.labels.shape[0] * 0.8):]]

    def getLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels

    def setOrder(self, seed):
        # Set the random order of shuffling for the subject/seed test
        np.random.seed(seed)
        self.order = np.arange(self.labels.shape[0])
        np.random.shuffle(self.order)

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
        if self.globalGoodFeatureMask is None:
            for feature in self.getFeatureList():
                if (
                    self.loadAnovaMask(
                        feature[1], f"sign{self.globalSignificance}")
                    is None
                ):
                    return None

                goodFeatures.append(
                    self.loadAnovaMask(
                        feature[1], f"sign{self.globalSignificance}")
                )

            self.globalGoodFeatureMask = goodFeatures

        tempFeatureMask = dp(self.globalGoodFeatureMask)
        return tempFeatureMask

    def getOrder(self):
        return self.order

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
