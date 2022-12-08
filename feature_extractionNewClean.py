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
from loadOwnData import loadOwnData as dl2
from loadOwnData import createEpochs2
# from loadOwnData import preProcessData
# import re
# pylint: disable=C0103


class featureEClass:
    def __init__(
        self,
        subject,
        paradigmName,
        globalSignificance,
        uniqueThresh,
        saveFolderName,
        signAll=True,
        signSolo=False,
        featureFolder="SavedFeaturesNew",
        onlyUniqueFeatures=False,
        useSepSubjFS=False,
        myData=False,
        holdOut=False,
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
        self.chunk = False
        self.chunkAmount = 1
        self.onlyUniqueFeatures = onlyUniqueFeatures
        self.uniqueThresh = uniqueThresh
        self.signAll = signAll
        self.signSolo = signSolo
        self.useSepSubjFS = useSepSubjFS
        self.saveFolderName = saveFolderName
        self.myData = myData
        self.holdOut = holdOut
        if self.signAll or self.signSolo:
            self.onlySign = True
        else:
            self.onlySign = False

        print(f"Feature class for subject {self.subject} created")

    def loadAllMaskedFeatures(self, featureList, folderName):
        maskedFeatureList = self.maskedFeatureList
        for feat in featureList:
            fName = f"{feat[1]}{folderName}"
            # print(fName)
            maskName = f"pt{self.globalSignificance}-u{self.uniqueThresh}"
            maskedFeatureList.append(self.loadMaskedFeature(
                featurename=fName, maskname=maskName))

        self.maskedFeatureList = maskedFeatureList

    def loadMaskedFeature(self, featurename, maskname):
        name = f"{featurename}{maskname}"

        saveDir = f"{os.getcwd()}/{self.saveFolderName}/SavedMaskedFeature/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(saveDir + f"/{name}.npy")
        # print(saveDir)
        # print(name)
        # print(path)

        if len(path) > 0:
            savedAnovaMask = np.load(path[0], allow_pickle=True)
            # savedAnovaMask[savedAnovaMask != 0] = 1
            # savedAnovaMask = np.array(savedAnovaMask, dtype=int)
            return savedAnovaMask
        else:
            return None

    def saveMaskedFeature(self, featurename, maskname, array):
        name = f"{featurename}{maskname}"

        # if self.onlyUniqueFeatures:
        #     name = f"{name}u{self.uniqueThresh}"

        saveDir = f"{os.getcwd()}/{self.saveFolderName}/SavedMaskedFeature/sub-{self.subject}-par-{self.paradigmName}"
        if os.path.exists(saveDir) is not True:
            os.makedirs(saveDir)

        np.save(
            f"{saveDir}/{name}",
            array,
        )

    def getMaskedFeatureList(self):
        tempMaskedFeatureList = dp(self.maskedFeatureList)
        return tempMaskedFeatureList

    def saveFeatures(self, name, array):

        saveDir = f"{os.getcwd()}/{self.saveFolderName}/{self.featureFolder}/sub-{self.subject}-par-{self.paradigmName}"
        if os.path.exists(saveDir) is not True:
            os.makedirs(saveDir)
        print(f"Saving feature{name} in {saveDir}")
        np.save(
            f"{saveDir}/{name}",
            array,
        )

    def loadFeatures(self, name):
        svpath = f"{os.getcwd()}/{self.saveFolderName}/{self.featureFolder}/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(svpath + f"/{name}.npy")
        if len(path) > 0:
            savedFeatures = np.load(path[0], allow_pickle=True)
            savedFeatures[0] = np.array(savedFeatures[0], dtype=np.float32)
            return savedFeatures
        else:
            return None

    def loadAnovaMask(self, featurename, maskname):
        name = f"{featurename}{maskname}"
        if self.onlyUniqueFeatures:
            name = f"{name}u{self.uniqueThresh}"

        saveDir = f"{os.getcwd()}/{self.saveFolderName}/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
        path = glob.glob(saveDir + f"/{name}.npy")
        # print(saveDir)
        # print(name)
        # print(path)

        if len(path) > 0:
            savedAnovaMask = np.load(path[0], allow_pickle=True)
            savedAnovaMask[savedAnovaMask != 0] = 1
            savedAnovaMask = np.array(savedAnovaMask, dtype=int)
            return savedAnovaMask
        else:
            return None

    def saveAnovaMask(self, featurename, maskname, array):
        name = f"{featurename}{maskname}"

        if self.onlyUniqueFeatures:
            name = f"{name}u{self.uniqueThresh}"

        saveDir = f"{os.getcwd()}/{self.saveFolderName}/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
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
        subject,
    ):
        onlyBest = False
        print("Mixing Data")
        dataList = []
        nameList = []
        labelsList = []
        dataNrs = np.arange(len(featureList))
        combos = []
        if useBestFeaturesTest:
            namesAndIndex = np.array([len(featureList), 2], dtype=object)
            namesAndIndexBestFeatures = np.zeros(
                np.array(bestFeatures, dtype=object).shape)
            bestFeatures = np.array(bestFeatures, dtype=object)

            if bestFeatures.shape[0] == 18:
                for subInd in range(9):
                    for fInd in range(2):
                        if bestFeatures[subInd * 2 + fInd].ndim > 1:
                            for x in range(bestFeatures[(subject - 1) * 2 + fInd].shape[0]):
                                bestFeatures[subInd * 2 +
                                             fInd, x] = bestFeatures[(subject - 1) * 2 + fInd, x]
                        else:
                            bestFeatures[subInd * 2 +
                                         fInd] = bestFeatures[(subject - 1) * 2 + fInd]

            if bestFeatures.shape[0] == 36:
                for subInd in range(9):
                    for fInd in range(4):
                        if bestFeatures[subInd * 4 + fInd].ndim > 1:
                            for x in range(bestFeatures[(subject - 1) * 4 + fInd].shape[0]):
                                bestFeatures[subInd * 4 +
                                             fInd, x] = bestFeatures[(subject - 1) * 4 + fInd, x]
                        else:
                            bestFeatures[subInd * 4 +
                                         fInd] = bestFeatures[(subject - 1) * 4 + fInd]

            for index, feature in enumerate(featureList, 0):
                namesAndIndex[0] = feature[1]
                namesAndIndex[1] = index
                if np.where(bestFeatures == feature[1])[0].shape[0] > 0:
                    row = np.where(bestFeatures == feature[1])[0]
                    if bestFeatures.shape[1] > 1:
                        column = np.where(bestFeatures == feature[1])[1]
                        namesAndIndexBestFeatures[row, column] = int(index)
                    else:
                        namesAndIndexBestFeatures[row] = int(index)

        # create All combinations of bestFeatures, dvs bara dem
        # Sen ta all combinations, of them and all other values
        if onlyBest:
            for row in namesAndIndexBestFeatures[:4]:
                combos.append(np.array(row, dtype=int))

        else:
            if useBestFeaturesTest:
                if bestFeatures.ndim > 1:
                    maxCombinationAmount = maxCombinationAmount - \
                        bestFeatures.shape[1]
                else:
                    maxCombinationAmount = 1

            if useBestFeaturesTest and maxCombinationAmount < 1:
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
            badIndexes = np.zeros([len(combos)])

            for ind, combo in enumerate(combos):
                if badIndexes[ind] == 1:
                    continue
                for ind2, combo2 in enumerate(combos):
                    if ind == ind2:
                        continue
                    if sorted(combo) == sorted(combo2):
                        badIndexes[ind2] = 1
            newCombos = []
            for ind, combo in enumerate(combos):
                if badIndexes[ind] == 0:
                    newCombos.append(combo)
            combos = newCombos

            combos = np.array(combos, dtype=object)

        for comb in combos:  # 10000

            nameRow = ""
            dataRo = np.copy(featureList[comb[0]][0])
            labelsRo = np.copy(labels)
            nameRow = nameRow + featureList[comb[0]][1]

            for nr in comb[1:]:

                data = np.copy(featureList[nr][0])

                dataRo = np.concatenate(
                    [dataRo, data], axis=1, dtype=np.float32)

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

            normShuffledDataList.append(sDataRow)
        dataList = None

        # print("Skipping NORMALIZING")
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

    # Loading my own recorded data
    def loadOwnData(self, t_min, t_max, sampling_rate, twoDLabels, paradigms):
        words = [["Sad", 51], ["Angry", 52], ["Happy", 53], ["Disgusted", 54]]
        # words = [["Up", 31], ["Down", 32], [
        #     "Left", 33], ["Right", 34], ["", 88]]
        testName = "SadAngryHappyDisgusted/"
        # testName = "4UpDownLeftRight/"
        wordDict = dict(words)
        exgData, markerData = dl2(
            dataPath=f"{testName}", t_start=t_min, t_end=t_max, words=wordDict)

        allTrials, allTrialsLabels = createEpochs2(
            words=wordDict, eegData=exgData, markerData=markerData)

        allTrials = np.array(allTrials)
        allTrialsLabels = np.array(allTrialsLabels)
        self.data = allTrials[:, :, int(sampling_rate *
                              t_min):int(sampling_rate * t_max)]
        self.labels = allTrialsLabels
        allTrials = None
        allTrialsLabels = None
        print(self.data.shape)

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

        timeSort = np.argsort(self.labelsAux[:, 0], axis=0)
        # tSortedlabelsAux = self.labelsAux[timeSort]
        self.data = self.data[timeSort]
        self.labels = self.labels[timeSort]
        if self.chunk:
            while True:
                if self.data.shape[2] % self.chunkAmount != 0:
                    self.data = self.data[:, :, :-1]
                else:
                    break

        return self.data, self.labels

    def averageChannels(self, preAveragedData):
        if preAveragedData.shape[-2] > 20:
            groupNames = ["OL", "OZ", "OR", "FL", "FZ",
                          "FR", "CL", "CZ", "CR", "PZ", "OPZ"]
            averagedChannelsData = np.zeros(
                [preAveragedData.shape[0], len(
                    groupNames), preAveragedData.shape[2]]
            )
            for avgNr, groupName in enumerate(groupNames):
                cnNrs = ut.channelNumbersFromGroupName(groupName)
                chosenChannels = preAveragedData[:, cnNrs, :]
                averagedChannelsData[:, avgNr, :] = np.mean(
                    chosenChannels, axis=1)
        else:
            averagedChannelsData = preAveragedData
        return averagedChannelsData

    # This covariance works best for features with time
    # If they do not have time. Then this is not necessary
    # I think. Or if they dont have time. Just dont
    def newCovariance(self, preCovFeature, splits=24):
        splits = 26
        smallShape = False
        if preCovFeature.ndim > 3:
            preCovFeature = np.reshape(
                preCovFeature, [preCovFeature.shape[0], preCovFeature.shape[1], -1])
        if preCovFeature.shape[-1] < 26:
            splits = preCovFeature.shape[-1]
            smallShape = True
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

        if smallShape:
            for trial in preCovFeature:
                postCovFeature.append(np.cov(trial, bias=1))
        else:
            for trial in preCovFeature:
                postCovFeature.append(np.cov(trial))
        postCovFeature = np.array(postCovFeature)

        return postCovFeature

    # shape, trial * channels * whatever. Here it does gradient/derivative on first dim after channel
    def gradientFunc(self, preDRFeature):
        postDRFeature = []
        for trial in preDRFeature:
            if trial.ndim < 2:
                postDRFeature.append(np.gradient(trial, axis=0))
            else:
                postDRFeature.append(np.gradient(trial, axis=1))
        postDRFeature = np.array(postDRFeature)
        print(postDRFeature.shape)
        return postDRFeature

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
            # print(featureName)
            # print(featureName[:-3])
            if self.loadFeatures(featureName[:-3]) is not None:
                preCVFeature = self.loadFeatures(featureName[:-3])[0]
                createdFeature = [
                    np.array(self.newCovariance(preCVFeature, splits=splitNr)),
                    featureNameSaved,
                ]
        elif featureName[-2:] == "GR":
            # print(featureName)
            # print(featureName[:-3])
            if self.loadFeatures(featureName[:-3]) is not None:
                preDRFeature = self.loadFeatures(featureName[:-3])[0]
                createdFeature = [
                    np.array(self.gradientFunc(preDRFeature)),
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

            if featureName == "chanEntr":
                import antropy as ant
                tempData = np.swapaxes(tempData, -1, -2)
                chanEntropy = []
                for trial in tempData:
                    chanEntropyTrial = []
                    for time in range(trial.shape[0] // 3):
                        threeTime = np.array(
                            trial[time:time + 2, :]).reshape([-1])
                        chanEntropyTrial.append(
                            ant.perm_entropy(threeTime, normalize=True))
                    # for time in trial:
                    #     chanEntropyTrial.append(
                    #         ant.perm_entropy(time, normalize=True))
                    chanEntropy.append(chanEntropyTrial)
                chanEntropy = np.array(chanEntropy, dtype=np.float32)
                createdFeature = [chanEntropy, featureNameSaved]

            if featureName == "timeEntr":
                import antropy as ant
                timeEntropy = []
                for trial in tempData:
                    timeEntropyTrial = []
                    for channel in trial:
                        windSize = tempData.shape[-1] // 8
                        timeEntropyWindows = []
                        for wind in range(7):
                            window = channel[windSize *
                                             wind: windSize * (wind + 1)]
                            # ant.perm_entropy(window, normalize=True)
                            timeEntropyWindows.append(
                                ant.perm_entropy(window, normalize=True))
                        timeEntropyTrial.append(
                            np.array(timeEntropyWindows).reshape(-1))
                    timeEntropy.append(timeEntropyTrial)

                timeEntropy = np.array(timeEntropy, dtype=np.float32)
                createdFeature = [timeEntropy, featureNameSaved]

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
                arLength = tempData.shape[-1]
                stftFeature = abs(signal.stft(tempData, fs=256, window="blackman", boundary="zeros",
                                              padded=True, axis=-1, nperseg=arLength // 6)[2])  # [:, :, :, :splitNr]
                stftFeature = np.swapaxes(stftFeature, -1, -2)
                print(stftFeature.shape)
                createdFeature = [stftFeature, featureNameSaved]

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

            if featureName == "normDatacor1x1":  # 1dataCorr2ax1d
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
            if featureName == "gausData2":
                createdFeature = [
                    ndimage.gaussian_filter1d(tempData, 2, axis=2),
                    featureNameSaved,
                ]

            if featureName == "normData":
                createdFeature = [
                    tempData,
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
        if createdFeature is not None:
            self.saveFeatures(featureNameSaved, createdFeature)
        return createdFeature

    def insert_cn(self, string, index=-2):
        return string[:index] + f"cn{self.chunkAmount}" + string[index:]

    def getFeatures(
        self,
        featureList,
        verbose,
    ):
        # print("before here?")
        if self.data is None:  # Really should load this separately
            raise Exception("Data should not be None")
        # print("ohiMark")
        # print(featureEClass.plotHeatMaps(self.data[0]))
        self.createdFeatureList = []
        tempData = np.copy(self.data)

        correctedExists = True
        self.allFeaturesAlreadyCreated = False
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
                if fNr == 59:
                    featureName = "chanEntr"
                if fNr == 60:
                    featureName = "timeEntr"
                if fNr == 61:
                    featureName = "timeEntr_CV"
                if fNr == 62:
                    featureName = "chanEntr_BC"
                if fNr == 63:
                    featureName = "timeEntr_BC"
                if fNr == 64:
                    featureName = "timeEntr_CV_BC"
                if fNr == 65:
                    featureName = "stftData_GR"
                if fNr == 66:
                    featureName = "stftData_GR_BC"
                if fNr == 67:
                    featureName = "stftData_GR_CV"
                if fNr == 68:
                    featureName = "chanEntr_GR"
                if fNr == 69:
                    featureName = "chanEntr_GR_BC"
                if fNr == 70:
                    featureName = "stftData_GR_CV_BC"
                if fNr == 71:
                    featureName = "chanEntr_GR_BC_CV"
                if fNr == 72:
                    featureName = "gausData_GR"
                if fNr == 73:
                    featureName = "gausData_GR_CV"
                if fNr == 74:
                    featureName = "gausData_GR_CV_BC"
                if fNr == 75:
                    featureName = "gausData_GR_BC"
                if fNr == 76:
                    featureName = "gausData_BC_GR_CV"
                if fNr == 77:
                    featureName = "gausData2"
                if fNr == 78:
                    featureName = "gausData2_GR"
                if fNr == 79:
                    featureName = "gausData2_GR_BC"
                if fNr == 80:
                    featureName = "gausData2_BC"
                if fNr == 81:
                    featureName = "gausData2_CV_BC"
                if fNr == 82:
                    featureName = "gausData2_GR_CV_BC"
                if fNr == 83:
                    featureName = "gausData2_GR_CV"
                if fNr == 84:
                    featureName = "gausData2_BC_GR_CV"
                if fNr == 85:
                    featureName = "fftData_BC_ifft_GR"
                if fNr == 86:
                    featureName = "fftData_BC_ifft_GR_CV"
                if fNr == 87:
                    featureName = "normData"
                if fNr == 88:
                    featureName = "normData_GR"
                if fNr == 89:
                    featureName = "normData_GR_CV"
                if fNr == 90:
                    featureName = "normData_GR_BC"
                if fNr == 91:
                    featureName = "normData_GR_BC_CV"
                if fNr == 92:
                    featureName = "normData_GR_CV_BC"
                if fNr == 93:
                    featureName = "normData_BC_GR"
                if fNr == 94:
                    featureName = "normData_BC_GR_CV"
                if fNr == 95:
                    featureName = "gausData2_CV"

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
                    # self.allFeaturesAlreadyCreated = True
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
                        # self.allFeaturesAlreadyCreated = False
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

        createdFeature = None
        tempData = None
        return self.createdFeatureList, self.labels, correctedExists

    def getOrigData(self):
        tempData = dp(self.data)
        return tempData

    def getFeatureList(self):
        tempFeatureList = dp(self.createdFeatureList)
        return tempFeatureList

    def addNameFeat(self, name):
        for feat in self.createdFeatureList:
            feat[1] = f"{feat[1]}{name}"
        print(f"addedName {name} to featureNames")

    def getFeatureListFlat(self):

        tempFeatureList = dp(self.createdFeatureList)
        for feature in tempFeatureList:
            if "CV" in feature[1]:
                if feature[0].ndim == 2:
                    raise Exception("Too Few Dims to triu ")
                # print(feature[0].shape)
                feature[0][feature[0] == 0] = 12345678
                trialAmount = feature[0].shape[0]
                feature[0] = np.triu(feature[0], k=-1)
                feature[0] = feature[0][feature[0] != 0]
                feature[0][feature[0] == 12345678] = 0
                feature[0] = np.reshape(feature[0], [trialAmount, -1])
                # print(feature[0].shape)

            feature[0] = self.flattenAllExceptTrial(feature[0])

        return tempFeatureList

    def extendFeatureList(self, additionList):
        self.data = None
        self.createdFeatureList.extend(additionList)

    def getLabelsAux(self):
        tempLabelsAux = dp(self.labelsAux)
        return tempLabelsAux

    def getLabels(self):
        tempLabels = dp(self.labels)
        return tempLabels

    def setTestNr(self, testNr):
        self.testNr = testNr

    def setOrder(self, seed, testSize):
        # Set the random order of shuffling for the subject/seed test
        self.orderList = []
        if self.holdOut:
            trainSplit = 0.7
            testSplit = 0.3
        else:
            trainSplit = 0.8
            testSplit = 0.2

        sss = StratifiedShuffleSplit(
            testSize, train_size=trainSplit, test_size=testSplit, random_state=seed
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
            self.paradigmName = f"{self.paradigmName}usingSoloFsubs"

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
            self.saveMaskedFeature(featurename=feature[1],
                                   maskname=f"pt{self.globalSignificance}-u{self.uniqueThresh}",
                                   array=maskedFeature)
        self.createdFeatureList = None
        self.globalGoodFeatureMask = None
        self.data = None
        self.maskedFeatureList = cleanMaskedFeatureList

    # Also flattens the data.
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

        onlySignificantFeatures = np.array(
            onlySignificantFeatures, dtype=np.float32)
        return onlySignificantFeatures

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

        rejoinedSplits = np.concatenate(covSplitList, axis=1)
        createdFeature = [
            np.array(rejoinedSplits),
            featureNameSaved,
        ]  # dataFFTCV
        return createdFeature

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
