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
        featureFolder="SavedFeatures",
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
        self.order = None
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

    def createMaskedFeatureList(self):
        featureList = self.getFeatureList()
        goodDataMaskList = self.getGlobalGoodFeaturesMask()
        maskedFeatureList = dp(featureList)
        # print(len(goodDataMaskList))
        # print(len(featureList))
        cleanMaskedFeatureList = []
        for feature, mask, maskedFeature in zip(
            featureList, goodDataMaskList, maskedFeatureList
        ):
            # print(feature[1])
            # print(feature[0].shape)
            maskedFeature[0] = self.onlySignData(feature=feature[0], goodData=mask)
            if maskedFeature[0] is not None:
                cleanMaskedFeatureList.append(maskedFeature)

        self.maskedFeatureList = cleanMaskedFeatureList

    def getMaskedFeatureList(self):
        tempMaskedFeatureList = dp(self.maskedFeatureList)
        return tempMaskedFeatureList

    def onlySignData(self, feature, goodData=None, goodData2=None):
        # One feature at a time. Only feature part.
        flatFdata = self.flattenAllExceptTrial(feature)
        # print(goodData.shape)

        # flatFdata = feature
        if self.signAll and self.signSolo:
            if flatFdata[:, [goodData != 0][0] + [goodData2 != 0][0]].shape[1] < 2:
                return 0.25
            onlySignificantFeatures = flatFdata[
                :, [goodData != 0][0] + [goodData2 != 0][0]
            ]

        elif self.signAll:
            if flatFdata[:, np.where(goodData != 0)[0]].shape[1] < 2:
                return None
            onlySignificantFeatures = flatFdata[:, np.where(goodData != 0)[0]]
            # ndata_test = ndata_test[:, np.where(goodData != 0)[0]]

        elif self.signSolo:
            if flatFdata[:, np.where(goodData2 != 0)[0]].shape[1] < 3:
                return 0.25
            onlySignificantFeatures = flatFdata[:, np.where(goodData2 != 0)[0]]
            # ndata_test = ndata_test[:, np.where(goodData2 != 0)[0]]

        return onlySignificantFeatures

    def saveFeatures(self, name, array):

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
        if self.onlyUniqueFeatures:
            name = f"{name}u{self.uniqueThresh}"

        saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
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
        order,
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
        # gddataList = [] # All data is good now
        nameList = []
        labelsList = []
        dataNrs = np.arange(len(featureList))
        combos = []

        namesAndIndex = np.array([len(featureList), 2], dtype=object)
        namesAndIndexBestFeatures = np.zeros(np.array(bestFeatures, dtype=object).shape)
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
            # print(maxCombinationAmount)

            # print(maxCombinationAmount)
            if maxCombinationAmount > len(dataNrs):
                maxCombinationAmount = len(dataNrs)
            for L in range(1, maxCombinationAmount + 1):
                for subsetNr in itertools.combinations(dataNrs, L):
                    if useBestFeaturesTest:
                        for row in namesAndIndexBestFeatures:
                            # print(
                            #     np.array(
                            #         np.concatenate(
                            #             [np.array(row), cp(subsetNr)], axis=0),
                            #         dtype=int,
                            #     )
                            # )
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
        # print(combos)
        if maxCombinationAmount < 1:
            combos = np.array(combos, dtype=int)
        combos = np.array(combos, dtype=object)
        # print(combos)
        # import re
        # print(combos)
        # print("oka")
        for comb in combos:  # 10000

            # if useBestFeaturesTest:
            #     if len(comb) < maxCombinationAmount - 1:
            #         continue
            #     for row in namesAndIndexBestFeatures:
            #         nrOfBad = 0
            #         for val in comb:
            #             if val not in row:
            #                 nrOfBad = nrOfBad + 1
            #         if nrOfBad < 2:
            #             break
            #     if nrOfBad > 1:
            #         continue

            # for row in namesAndIndexBestFeatures:
            #     nrOfBad = 0
            #     if val not in namesAndIndexBestFeatures:
            #         nrOfBad = nrOfBad + 1
            #     if nrOfBad == 2:
            #         break
            # if nrOfBad < 2:
            #     break

            # # print(comb)
            # if useBestFeaturesTest:

            #     oneGoodRow = False
            #     for row in namesAndIndexBestFeatures:  # 9
            #         notGoodColumns = False
            #         for column in row:  # 4
            #             atleastOne = False
            #             for val in comb:  # at least one
            #                 if column == val:
            #                     atleastOne = True
            #             if atleastOne is False:
            #                 # if column not in comb:
            #                 notGoodColumns = True
            #                 break
            #         if notGoodColumns is False:
            #             oneGoodRow = True
            #     if oneGoodRow is False:
            #         continue
            #     doCombo = False
            #     fNameList = list()
            #     for nr in comb:
            #         fNameList.append(featureList[nr][1])
            #     # if re.search(featUre[1], fName) is not None:

            #     # matchedOne = False
            #     for (
            #         bfeat
            #     ) in bestFeatures:  # list of features that need to match at least one
            #         notThisGoodFeatureCombo = False
            #         if type(bfeat) == list and len(bfeat) > 0:
            #             for bfeat2 in bfeat:
            #                 bNameExists = False
            #                 for nameInCombo in fNameList:  # List of features in combo
            #                     if bfeat2 == nameInCombo:
            #                         # if re.search(bfeat2, nameInCombo) is not None:
            #                         bNameExists = True
            #                 if bNameExists is not True:
            #                     notThisGoodFeatureCombo = True
            #         # else:
            #         #     bfeat2 = bfeat
            #         #     bNameExists = False
            #         #     for nameInCombo in fNameList:  # List of features in combo
            #         #         bNameExists = False
            #         #         for nameInCombo in fNameList:  # List of features in combo
            #         #             if bfeat2 == nameInCombo:
            #         #                 # if re.search(bfeat2, nameInCombo) is not None:
            #         #                 bNameExists = True
            #         #         if bNameExists is not True:
            #         #             notThisGoodFeatureCombo = True

            #         if notThisGoodFeatureCombo is not True:
            #             doCombo = True
            #     if doCombo is not True:
            #         continue

            nameRow = ""
            dataRo = np.copy(featureList[comb[0]][0])
            # dataRo = self.flattenAllExceptTrial( ALREADY FLATTENED
            #     np.copy(featureList[comb[0]][0]))
            # if self.globalGoodFeatureMask is not None:
            #     gddataRo = self.globalGoodFeatureMask[comb[0]]
            labelsRo = np.copy(labels)
            nameRow = nameRow + "-" + featureList[comb[0]][1]

            for nr in comb[1:]:

                data = np.copy(featureList[nr][0])
                # data = self.flattenAllExceptTrial(np.copy(featureList[nr][0])) ALREADY FLATTENED
                # if self.globalGoodFeatureMask is not None:
                #     gddata = self.globalGoodFeatureMask[nr]
                dataRo = np.concatenate([dataRo, data], axis=1)
                # if self.globalGoodFeatureMask is not None:
                #     gddataRo = np.concatenate([gddataRo, gddata], axis=0)
                nameRow = featureList[nr][1] + "-" + nameRow

            dataList.append(dataRo)
            # if self.globalGoodFeatureMask is not None:
            #     gddataList.append(gddataRo)
            # else:
            #     gddataList.append(None)
            nameList.append(nameRow)
            labelsList.append(labelsRo)
        # import re
        normShuffledDataList = []
        for x, dataR in enumerate(dataList):  # Should be zip
            # continueFlag = True
            # for bfeat in bestFeatures:
            #     if type(bfeat) == list and len(bfeat) > 1:
            #         allIn = True
            #         for bfeat2 in bfeat:
            #             # print(bfeat2)
            #             # print(nameList[x])
            #             # print(re.search(bfeat2, nameList[x]))
            #             if re.search(bfeat2, nameList[x]) is not None:
            #                 allIn = False
            #         if allIn:
            #             continueFlag = False
            #     else:
            #         if re.search(bfeat, nameList[x]) is not None:
            #             continueFlag = False
            # # continueFlag = False
            # if continueFlag:
            #     continue

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
                    order=order,
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

    def shuffleSplitData(self, data_t, labels_t, name, order):

        data_s = np.copy(data_t)
        labels_s = np.copy(labels_t)

        data_train = data_s[order[0 : int(labels_s.shape[0] * 0.8)]]
        data_test = data_s[order[int(labels_s.shape[0] * 0.8) :]]
        labels_train = labels_s[order[0 : int(labels_s.shape[0] * 0.8)]]
        labels_test = labels_s[order[int(labels_s.shape[0] * 0.8) :]]

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

        if self.paradigmName == "vviiudsep":
            # print(self.paradigmName)
            # print(self.labels)
            self.labels[self.labels > 1] = self.labels[self.labels > 1] - 2
            # print(self.labels)

        # if self.paradigmName == "vviirl":
        #     # print(self.paradigmName)
        #     # print(self.labels)
        #     self.labels[self.labels > 1] = self.labels[self.labels > 1] - 2
        #     # for labe in self.labels:
        #     if labe > 1
        # self.labels = newLabels

        # print(self.labels)
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
                :, splitNr :: (int(ceil(self.chunkAmount / 2))), :
            ]  # Split Channels
            covSplitList.append(ut.fftCovariance(splitFeature))
            # covSplits.append(splitFeature)

        rejoinedSplits = np.concatenate(covSplitList, axis=1)
        createdFeature = [
            np.array(rejoinedSplits),
            featureNameSaved,
        ]  # dataFFTCV
        return createdFeature

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

        if featureName == "inverseFFT-BC":
            if self.chunk:
                # noReshape = True
                loadedFFTBC = self.loadFeatures(f"fftDatacn{self.chunkAmount}BC")
                loadedFFTAngle = self.loadFeatures(f"anglefftDatacn{self.chunkAmount}")

            else:
                loadedFFTBC = self.loadFeatures("fftDataBC")
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
                ndimage.correlate1d(tempData, weights=weights, axis=2, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "1dataCorr2ax1d":
            weights = np.zeros(shape=[20])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[17:] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "05dataCorr2ax1d":
            channelHop = 7
            weights = np.zeros(shape=[3 + channelHop + 3])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[channelHop + 3 :] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "2dataCorr2ax1d":
            channelHop = 30
            weights = np.zeros(shape=[3 + channelHop + 3])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[channelHop + 3 :] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "3dataCorr2ax1d":
            channelHop = 50
            weights = np.zeros(shape=[3 + channelHop + 3])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[channelHop + 3 :] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "4dataCorr2ax1d":
            channelHop = 70
            weights = np.zeros(shape=[3 + channelHop + 3])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[channelHop + 3 :] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]
        if featureName == "5dataCorr2ax1d":
            channelHop = 90
            weights = np.zeros(shape=[3 + channelHop + 3])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[channelHop + 3 :] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]
        if featureName == "6dataCorr2ax1d":
            channelHop = 110
            weights = np.zeros(shape=[3 + channelHop + 3])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[channelHop + 3 :] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]

        # if featureName == "dataCorr2ax1dall":
        #     firstOne = True
        #     for channelHop in [30, 70, 90]:  # 5,  10, 50,  20, 40, 60, 80, 100

        #         weights = np.zeros(shape=[3 + channelHop + 3])
        #         weights[:3] = [0.25, 0.5, 0.25]
        #         weights[channelHop + 3:] = [0.25, 0.5, 0.25]
        #         if firstOne:
        #             allHops = ndimage.correlate1d(
        #                 tempData, weights=weights, axis=1, mode="wrap")
        #             firstOne = False
        #         else:
        #             allHops = np.concatenate([allHops, ndimage.correlate1d(
        #                 tempData, weights=weights, axis=1, mode="wrap")], axis=1)

        #         createdFeature = [
        #             allHops,
        #             featureNameSaved,
        #         ]

        if featureName == "dataCorr2ax1dthree":
            weights = np.zeros(shape=[58])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[55:] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=1, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "dataCorr1d01s":
            weights = np.zeros(shape=[32])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[29:] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=2, mode="wrap"),
                featureNameSaved,
            ]

        if featureName == "dataCorr1d02s":
            weights = np.zeros(shape=[58])
            weights[:3] = [0.25, 0.5, 0.25]
            weights[55:] = [0.25, 0.5, 0.25]
            createdFeature = [
                ndimage.correlate1d(tempData, weights=weights, axis=2, mode="wrap"),
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
                            int(loadedCorrectFeature[0].shape[2] / self.chunkAmount),
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

        if featureName == "gaussianData":
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
                    # Reshape chunks into more channels
                    # createdFeature[0] =np.reshape(
                    # createdFeature[0],
                    # [
                    #     int(createdFeature[0].shape[0] / self.chunkAmount),
                    #     -1,
                    #     createdFeature[0].shape[2],
                    # ],)
                )

        self.saveFeatures(featureNameSaved, createdFeature)
        return createdFeature

    def insert_cn(self, string, index=-2):
        return string[:index] + f"cn{self.chunkAmount}" + string[index:]

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

            # Splitting the data in time, into chunkAmount of smaller bits, each creating
            # A feature separately
            if self.chunk:
                tempData = np.reshape(
                    tempData,
                    (tempData.shape[0] * self.chunkAmount, tempData.shape[1], -1),
                )

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
                    featureName = "dataGCV"

                if fNr == 10:
                    featureName = "dataGCV2"

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
                    featureName = "gaussianData"

                if fNr == 19:
                    featureName = "dataGCVBC"

                if fNr == 20:
                    featureName = "gaussianDataBC"

                if fNr == 21:
                    featureName = "dataGCV-BC"

                # TODO: ADD BOTH KINDS OF BC. It exists. Just use it!
                if fNr == 22:
                    featureName = "dataFFTCV2-BC"

                if fNr == 23:
                    featureName = "dataGCV2-BC"

                if fNr == 24:
                    featureName = "dataCorr1dBC"

                if fNr == 25:
                    featureName = "inverseFFT-BC"

                if fNr == 26:
                    featureName = "dataCorr1d01s"
                if fNr == 27:
                    featureName = "dataCorr1d02s"
                if fNr == 28:
                    featureName = "iFFTdataCorr1d01s-BC"
                if fNr == 29:
                    featureName = "iFFTdataCorr1d02s-BC"
                if fNr == 30:
                    featureName = "iFFTdataCorr1d005s-BC"
                if fNr == 31:
                    featureName = "dataCorr1d01sBC"
                if fNr == 32:
                    featureName = "dataCorr1d02sBC"
                if fNr == 33:
                    featureName = "1dataCorr2ax1d"
                if fNr == 34:
                    featureName = "iFFTdataCorr2ax1d005s-BC"
                if fNr == 35:
                    featureName = "1dataCorr2ax1dBC"
                if fNr == 36:
                    featureName = "inverseFFTCV-BC"
                if fNr == 37:
                    featureName = "anglefftData"
                if fNr == 38:
                    featureName = "anglefftDataBC"
                if fNr == 39:
                    featureName = "2dataCorr2ax1d"
                if fNr == 40:
                    featureName = "2dataCorr2ax1dBC"
                if fNr == 41:
                    featureName = "3dataCorr2ax1d"
                if fNr == 42:
                    featureName = "3dataCorr2ax1dBC"
                if fNr == 43:
                    featureName = "4dataCorr2ax1d"
                if fNr == 44:
                    featureName = "4dataCorr2ax1dBC"
                if fNr == 45:
                    featureName = "5dataCorr2ax1d"
                if fNr == 46:
                    featureName = "5dataCorr2ax1dBC"
                if fNr == 47:
                    featureName = "6dataCorr2ax1d"
                if fNr == 48:
                    featureName = "6dataCorr2ax1dBC"
                if fNr == 49:
                    featureName = "05dataCorr2ax1d"
                if fNr == 50:
                    featureName = "05dataCorr2ax1dBC"
                # if fNr == 39:
                #     featureName = "dataCorr2ax1dall"
                # if fNr == 40:
                #     featureName = "dataCorr2ax1dallBC"

                if self.chunk:
                    if "BC" in featureName and "-BC" not in featureName:
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
                    if "BC" in featureName and "-BC" not in featureName:
                        if self.chunk:
                            if "split" not in self.paradigmName:
                                print(
                                    f"{featureName[0:-2]}cn{self.chunkAmount}BC not exists"
                                )
                        else:
                            if "split" not in self.paradigmName:
                                print(f"FeatureNameCorrectedNotExist {featureName}")
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

    def getTrainFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)

        for f in tempFeatureList:
            f[0] = f[0][self.order[0 : int(self.labels.shape[0] * 0.8)]]

        return tempFeatureList

    def getTestFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)

        for f in tempFeatureList:
            f[0] = f[0][self.order[int(self.labels.shape[0] * 0.8) :]]

        return tempFeatureList

    def getTrainLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels[self.order[0 : int(self.labels.shape[0] * 0.8)]]

    def getTestLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels[self.order[int(self.labels.shape[0] * 0.8) :]]

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

            self.saveAnovaMask(feature[1], f"sign{self.globalSignificance}", mask)

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
                    self.loadAnovaMask(feature[1], f"sign{self.globalSignificance}")
                    is None
                ):
                    print(feature[1])
                    if self.useSepSubjFS:
                        self.paradigmName = oldparadigmName

                    return None

                goodFeatures.append(
                    self.loadAnovaMask(feature[1], f"sign{self.globalSignificance}")
                )

            self.globalGoodFeatureMask = goodFeatures

        tempFeatureMask = dp(self.globalGoodFeatureMask)
        if self.useSepSubjFS:
            self.paradigmName = oldparadigmName

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
