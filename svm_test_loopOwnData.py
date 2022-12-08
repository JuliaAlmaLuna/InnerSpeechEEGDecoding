"""
This class runs a pipeline testing SVM classification on data
"""
from joblib import Parallel, delayed
from copy import deepcopy as dp
import numpy as np
import feature_extractionNewClean as fclass
# from baselineClean import baseLineCorrection
from baselineCorrection import baseLineCorrection
import svmMethods as svmMet
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler
import paradigmSetting
import cProfile
import pstats
import io
import time
# Reduces size of arrays. Float64 precision not always needd.
from numpy import float32
import glob
import os
import dask


def testLoop(
    data_train,
    data_test,
    labels_train,
    labels_test,
    name,
    useAda,
    userndF,
    useMLP,
    useOVR,
    fmetDict,
    sub,
):

    # If else statements that swap between different train/test models.
    if useAda:

        allResults = fmetDict["allSame"].testSuiteAda(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )
    elif userndF:
        allResults = fmetDict["allSame"].testSuiteForest(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )
    elif useMLP:
        allResults = fmetDict["allSame"].testSuiteMLP(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )
    elif useOVR:
        allResults = fmetDict["allSame"].testSuiteOVRHoldOut(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )
    else:
        allResults = fmetDict["allSame"].testSuite(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )

    return allResults


# This function seems quite unecessary!


def mixShuffleSplit(
    createdFeatureList,
    labels,
    featureClass,
    maxCombinationAmount,
    bestFeatures,
    useBestFeaturesTest,
    subject,
):

    # Copy labels and features list to avoid changes to originals. Probly not needed
    tempLabels = dp(labels)
    tempFeatureList = dp(createdFeatureList)

    mDataList = featureClass.createListOfDataMixes(
        featureList=tempFeatureList,
        labels=tempLabels,
        maxCombinationAmount=maxCombinationAmount,
        bestFeatures=bestFeatures,
        useBestFeaturesTest=useBestFeaturesTest,
        subject=subject,
    )
    tempLabels = None
    tempFeatureList = None
    return mDataList

# This class prints to separate files for each subject when using dask multiprocessing.


def printProcess(processName, printText, saveFolderName):
    if os.path.exists(f"{os.getcwd()}/{saveFolderName}/processOutputs/") is not True:
        os.makedirs(f"{os.getcwd()}/{saveFolderName}/processOutputs")
    with open(f"{os.getcwd()}/{saveFolderName}/processOutputs/{processName}Output.txt", "a") as f:
        print(printText, file=f)


def loadFeatureSelectMaskNoClass(
    featurename, maskname, uniqueThresh, paradigmName, subject, onlyUniqueFeatures, saveFolderName
):
    name = f"{featurename}{maskname}"
    if onlyUniqueFeatures:
        name = f"{name}u{uniqueThresh}"

    saveDir = f"{os.getcwd()}/{saveFolderName}/SavedAnovaMask/sub-{subject}-par-{paradigmName}"
    path = glob.glob(saveDir + f"/{name}.npy")
    if len(path) > 0:
        savedAnovaMask = np.load(path[0], allow_pickle=True)
        # TODO make this into a boolean mask
        savedAnovaMask = np.array(savedAnovaMask, dtype=float32)
        return savedAnovaMask
    else:
        return None


def saveFeatureSelectMaskNoClass(
    featurename,
    maskname,
    array,
    uniqueThresh,
    paradigmName,
    subject,
    onlyUniqueFeatures,
    saveFolderName
):
    name = f"{featurename}{maskname}"

    if onlyUniqueFeatures:
        name = f"{name}u{uniqueThresh}"

    saveDir = f"{os.getcwd()}/{saveFolderName}/SavedAnovaMask/sub-{subject}-par-{paradigmName}"
    if os.path.exists(saveDir) is not True:
        import random as rand
        time.sleep(rand.random() * 10)
        if os.path.exists(saveDir) is not True:
            os.makedirs(saveDir)

    np.save(
        f"{saveDir}/{name}",
        array,
    )


@dask.delayed
def featCovCorrMasking(
    flatfeature,
    goodData,
    uniqueThresh,
    featureName,
    subject,
    paradigmName,
    significanceThreshold,
    saveFolderName,
):

    # These goodfeatures need to come with a array of original index
    # Then. When a feature is deleted. Make it zero on goodDataMask
    goodfeature = flatfeature[:, np.where(goodData != 0)[0]]
    flatfeature = None
    indexList = np.where(goodData != 0)[
        0
    ]  # list of indexes for goodFeatures from Anova

    goodfeature = np.swapaxes(goodfeature, 0, 1)

    # If less than 2 features are significant, return mask of only zeros.
    if goodfeature.shape[0] < 2:
        printProcess(f"subj{subject}output",
                     " Too few good features", saveFolderName)
        printProcess(f"subj{subject}output", goodfeature.shape, saveFolderName)
        goodData = np.zeros(goodData.shape)
    else:

        printProcess(
            f"subj{subject}output",
            f"{np.count_nonzero(goodData)} good Features \
                            before covRemoval:{uniqueThresh}in {featureName}", saveFolderName
        )
        printProcess(f"subj{subject}output",
                     time.process_time(), saveFolderName)
        # Create a corrcoef matrix of the features, comparing them to one another

        corrMat = np.corrcoef(goodfeature, dtype=float32)
        # corrMat = np.array(corrMat, dtype=float16)
        goodfeature = None
        printProcess(f"subj{subject}output",
                     time.process_time(), saveFolderName)

        printProcess(f"subj{subject}output", corrMat.shape, saveFolderName)

        # Keep only everything above diagonal
        halfCorrMat = np.triu(corrMat, 1)
        halfCorrMat = np.array(halfCorrMat, dtype=float32)
        corrMat = None
        # Create list of all features that are too correlated, except one of the features ( the one with lower index)
        deleteIndexes = np.where(halfCorrMat > uniqueThresh)[1]
        halfCorrMat = None

        # Delete these features from goodData mask
        goodData[indexList[deleteIndexes]] = 0
        printProcess(
            f"subj{subject}output",
            f"{np.count_nonzero(goodData)} good Features \
                            after covRemoval:{uniqueThresh} in {featureName}", saveFolderName
        )

    saveFeatureSelectMaskNoClass(
        featurename=featureName,
        maskname=f"sign{significanceThreshold}",
        array=goodData,
        uniqueThresh=uniqueThresh,
        paradigmName=paradigmName,
        subject=subject,
        onlyUniqueFeatures=True,
        saveFolderName=saveFolderName
    )


def createFeatureSelectMask(
    featureList,
    labels,
    significanceThreshold,
    onlyUniqueFeatures,
    uniqueThresh,
    paradigmName,
    subject,
    saveFolderName
):

    printProcess(
        f"subj{subject}output",
        f"Running anova Test and masking using sign threshold: {significanceThreshold}", saveFolderName
    )

    # I use the sklearn StandarScaler before the ANOVA test since that is what will do
    # later as well for every feature before test.

    scaler = StandardScaler()

    goodFeatureMaskList = []
    for feature in featureList:  # Features

        featureName = feature[1]
        loadedMask = loadFeatureSelectMaskNoClass(
            featurename=featureName,
            maskname=f"sign{significanceThreshold}",
            uniqueThresh=uniqueThresh,
            paradigmName=paradigmName,
            subject=subject,
            onlyUniqueFeatures=onlyUniqueFeatures,
            saveFolderName=saveFolderName
        )

        if loadedMask is None:

            flatfeature = np.reshape(feature[0], [feature[0].shape[0], -1])

            scaler.fit(flatfeature)
            flatfeature = scaler.transform(flatfeature)

            # Running the ANOVA Test
            f_statistic, p_values = feature_selection.f_classif(
                flatfeature, labels)

            # Create a mask of features with P values below threshold
            p_values[p_values > significanceThreshold] = 0
            # This calculation doesn´t serve a purpose at this point.
            p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2

            goodData = f_statistic * p_values  # This mask contains

            remainingNrOfFeatures = np.count_nonzero(goodData)
            # If loop that reduces amount of kept features to 17000 at max.
            if remainingNrOfFeatures > 17000:
                ratioKeep = int(17000 / len(goodData) * 100)
                bestPercentile = feature_selection.SelectPercentile(
                    feature_selection.f_classif, percentile=ratioKeep
                )
                bestPercentile.fit(flatfeature, labels)
                goodData = bestPercentile.get_support() * 1

            f_statistic = None
            p_values = None

            # If CV in name, then feature is a Covariance matrix. And as such. Only half of it is useful information
            if "CV" in featureName:
                goodData = np.reshape(goodData, [feature[0].shape[1], -1])

                goodData = np.triu(
                    goodData, k=0
                )  # Keep only half of CV matrices. Rest is the same. No need to cull twice.
                goodData = np.reshape(goodData, [-1])
                # Applies to the two last axes.

            feature[0] = None

            # If onlyUniqueFeatures, then uses covcorrelation matrix to remove too similar features from mask.
            if onlyUniqueFeatures:
                # This function is dask delayed so when called later compute() to multiprocess it.
                goodData = featCovCorrMasking(
                    flatfeature,
                    goodData,
                    uniqueThresh,
                    featureName,
                    subject,
                    paradigmName,
                    significanceThreshold,
                    saveFolderName=saveFolderName
                )

        else:
            printProcess(f"subj{subject}output",
                         f"Loaded mask {featureName}", saveFolderName)
            goodData = loadedMask

        goodFeatureMaskList.append(goodData)
        goodData = None
        feature[0] = None

    return goodFeatureMaskList


# This function creates Fselect masks for each subject, then adds all the other subjects ( except one subject ) masks
# Does not seem to help much, but keep it. To check later. Maybe with fixes will be better!
# And for features that are extremely different between subjects.
#
def fSelectUsingSepSubjects(
    fClassDict,
    globalSignificanceThreshold,
    onlyUniqueFeatures,
    uniqueThresh,
    paradigmName,
    subjects,
    saveFolderName,
):
    goodFeatureMaskListList = []
    tempLabels = fClassDict[f"{1}"].getLabels()
    uniqueLabels = np.unique(tempLabels)
    for label in uniqueLabels:
        goodFeatureMaskListList = []
        for sub in subjects:
            # Temporarily makes labels into 0 for chosen label and and 1 for the rest.

            ovrLabels = fClassDict[f"{sub}"].getLabels()
            ovrLabels[ovrLabels != label] = 5
            ovrLabels[ovrLabels == label] = 0
            ovrLabels[ovrLabels != 0] = 1

            goodFeatureMaskList = createFeatureSelectMask(
                featureList=fClassDict[f"{sub}"].getFeatureList(),
                labels=ovrLabels,
                significanceThreshold=globalSignificanceThreshold,
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
                paradigmName=f"{paradigmName}-label-{label}-soloFSSub",
                subject=sub,
                saveFolderName=saveFolderName
            )
            goodFeatureMaskListList.append(goodFeatureMaskList)

        # Computes the delayed dask functions
        compute3 = dask.compute(goodFeatureMaskListList)
        goodFeatureMaskListList = dask.compute(compute3)

    for sub in subjects:
        for feature in fClassDict[f"{sub}"].getFeatureList():
            anovaMask = None
            for sub2 in subjects:
                for label in uniqueLabels:
                    if sub2 == sub:
                        continue
                    if anovaMask is None:
                        anovaMask = loadFeatureSelectMaskNoClass(
                            featurename=feature[1],
                            maskname=f"sign{globalSignificanceThreshold}",
                            uniqueThresh=uniqueThresh,
                            paradigmName=f"{paradigmName}-label-{label}-soloFSSub",
                            subject=sub2,
                            onlyUniqueFeatures=onlyUniqueFeatures,
                            saveFolderName=saveFolderName
                        )
                    else:
                        anovaMask = anovaMask + loadFeatureSelectMaskNoClass(
                            featurename=feature[1],
                            maskname=f"sign{globalSignificanceThreshold}",
                            uniqueThresh=uniqueThresh,
                            paradigmName=f"{paradigmName}-label-{label}-soloFSSub",
                            subject=sub2,
                            onlyUniqueFeatures=onlyUniqueFeatures,
                            saveFolderName=saveFolderName
                        )
            saveFeatureSelectMaskNoClass(
                featurename=feature[1],
                paradigmName=f"{paradigmName}usingSoloFsubs",
                maskname=f"sign{globalSignificanceThreshold}",
                array=anovaMask,
                subject=sub,
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
                saveFolderName=saveFolderName,
            )


# This class combines all subjects except one, so that the combination can be sent into anovaTest for feature selection
# Mask creation
# This code combines the features and labels from multiple subjects into one array.
# It takes in a dictionary of subjects and their corresponding feature and label arrays,
# as well as a subject to leave out (optional) and a boolean for whether to only use the training data (optional).
# It then loops through each subject, concatenating the features and labels into one array.
# After looping through all subjects, it returns the combined feature and label arrays.
# It also prints out the number of features used and the shape of the combined trials.
def combineAllSubjects(fclassDict, subjectLeftOut=None, onlyTrain=False):
    print(f"Combining all subjects except {subjectLeftOut} into one array ")
    first = True
    for subName, fClass in fclassDict.items():
        if subName == f"{subjectLeftOut}":
            continue
        if first:
            if onlyTrain:
                allSubjFList = fClass.getTrainFeatureList()
                allSubjFLabels = fClass.getTrainLabels()
            else:
                allSubjFList = fClass.getFeatureList()
                allSubjFLabels = fClass.getLabels()
            first = False
            continue
        if onlyTrain:
            flist = fClass.getTrainFeatureList()
            flabels = fClass.getTrainLabels()
        else:
            flist = fClass.getFeatureList()
            flabels = fClass.getLabels()
        for allfeature, onefeature in zip(allSubjFList, flist):
            allfeature[0] = np.concatenate([allfeature[0], onefeature[0]], 0)

        allSubjFLabels = np.concatenate([allSubjFLabels, flabels], 0)

    flist = None
    flabels = None
    print(f"{len(allSubjFList)} features used when combining")  # Nr of features
    print(f"{allSubjFLabels.shape} trials combined")

    return allSubjFList, allSubjFLabels


def main():
    ##############################################################
    # Settings to limit maximum amount of processor cores used
    ##############################################################
    import psutil
    p = psutil.Process()
    p.cpu_affinity(list(np.arange(18)))
    p.cpu_affinity()
    ##############################################################
    # Paradigm specific parameters
    ##############################################################
    subjects = [1]
    testSize = 7  # Nr of seed iterations until stopping
    seed = 39  # Arbitrary, could be randomized as well.
    myOwnTest = True
    paradigm = paradigmSetting.sadAngryHappyDisgustedJulia()
    # paradigm = paradigmSetting.UpDownLeftRightJulia()
    paraName = paradigm[0]
    ##############################################################
    # Feature selection parameters
    ##############################################################
    signSolo = False  # Using mask created from training data for the subject
    soloSignificanceThreshold = 0.005
    signAll = False  # Using mask created from other subjects
    globalSignificanceThreshold = 0.1
    useSepSubjFS = True
    if useSepSubjFS:
        globalSignificanceThreshold = 0.01
    onlyUniqueFeatures = True  # Masking for features that are too correlated.
    uniqueThresh = 0.9
    ##############################################################
    # Test parameters
    ##############################################################
    testName = "myOwnTestAvg"
    repNr = 26
    useBestFeaturesTest = True
    useBestFeaturesPerLabel = True
    maxCombinationAmount = 3
    sameSizeBestFeat = True
    holdOut = True
    useMasked = False
    quickTest = True
    useBestAvg = False
    ##############################################################
    # Trial window parameters
    ##############################################################
    testNameNr = 10
    saveFolderName = f"{testName}First{testNameNr}"
    t_min = 6.5
    t_max = 7.5
    sampling_rate = 250
    saveFolderName2 = f"{testName}Second{testNameNr}"
    useWinFeat = True
    t_min2 = 7.5
    t_max2 = 8.5
    saveFolderName3 = f"{testName}Third{testNameNr}"
    useWinFeat2 = True
    t_min3 = 8.5
    t_max3 = 9.5
    ##############################################################
    # Continuation of test parameters
    ##############################################################
    if maxCombinationAmount == 1:
        useBestFeaturesTest = False

    repetitionName = f"{paraName}{maxCombinationAmount}c{testName}"
    repetitionValue = f"{repNr}{repetitionName}"

    # When increasing combination amount by one each test.
    if sameSizeBestFeat:
        bestFeaturesSaveFile = f"top{maxCombinationAmount}{paraName}.npy"
    else:
        bestFeaturesSaveFile = f"top{maxCombinationAmount-1}{paraName}.npy"
    if useBestAvg:
        bestFeaturesSaveFile = f"topAvg{maxCombinationAmount-1}{paraName}.npy"
    worstFeaturesSaveFile = f"worstFeats1{paraName}.npy"
    ################################################################
    # Sklearn/TestTrain parameters
    useAda = False  # Using ADA
    userndF = False  # Sklearn random forest, works a little worse and a little slower than SVM at this point
    useMLP = False  # Sklearn MLP, not made good yet. Works ok
    useOVR = True
    tolerance = 0.001  # Untested
    ##############################################################
    # Other parameters
    ##############################################################
    usefeaturesToTestList = True
    featuresToTestDict = dict()
    ##############################################################
    # Parameters used if onlyCreate is active
    ##############################################################
    onlyCreateFeatures = False
    nrFCOT = 2  # nrOfFeaturesToCreateAtOneTime
    featIndex = 0  # Multiplied by nrFCOT, First features to start creating
    featuresToOnlyCreateNotTest = []
    featuresToTestDict["fftFeatures"] = [
        1,  # fftData, # Needed if create
        6,  # fftData_CV
        15,  # fftData_BC_CV
        12,  # fftData_BC
        55,  # fftData_CV_BC

    ]
    featuresToOnlyCreateNotTest.append(0)
    featuresToTestDict["stftFeatures"] = [
        51,  # stftData,
        52,  # stftData_BC
        65,  # stftData_GR
        53,  # stftData_CV
        66,  # stftData_GR_BC
        # 54,  # stftData_BC_CV
        # 58,  # stftData_CV_BC
        67,  # stftData_GR_CV
        # 70,  # stftData_GR_CV_BC
    ]

    # featuresToTestDict["welchFeatures"] = [
    #     2,  # welchData
    #     7,  # welchData_CV
    #     13,  # welchData_BC
    #     16,  # welchData_BC_CV
    #     56,  # welchData_CV_BC
    # ]
    featuresToTestDict["hilbertFeatures"] = [
        3,  # hilbertData, Needed if create
        8,  # hilbertData_CV Needed if create
        14,  # hilbertData_BC
        57,  # hilbertData_CV_BC
        17,  # hilbertData_BC_CV
    ]
    # featuresToOnlyCreateNotTest.append(3)
    # featuresToOnlyCreateNotTest.append(8)

    featuresToTestDict["gaussianFeatures"] = [
        9,  # "gausData"
        # 10,  # dataGCV2
        18,  # gausData_CV
        # 19,  # gausData_CV_BC
        72,  # gausData_GR
        20,  # gausData_BC
        73,  # gausData_GR_CV
        21,  # gausData_BC_CV
        # 74,  # gausData_GR_CV_BC
        75,  # gausData_GR_BC
        # 76,  # gausData_BC_GR_CV
    ]

    featuresToTestDict["normFeatures"] = [
        87,  # "normData"
        # 10,  # dataGCV2
        # 18,  # gausData_CV
        # 19,  # gausData_CV_BC
        88,  # normData_GR
        90,  # normData_GR_BC
        89,  # normData_GR_CV
        91,  # normData_GR_BC_CV
        92,  # normData_GR_CV_BC
        # 93,  # normData_BC_GR
        # 94,  # normData_BC_GR_CV
    ]

    featuresToTestDict["inversefftFeatures"] = [
        25,  # fftData_BC_ifft
        # 28,  # fftData_BC_ifft_cor2x1
        # 29,  # fftData_BC_ifft_cor2x2
        # 30,  # fftData_BC_ifft_cor2x3
        # 34,  # fftData_BC_ifft_cor1x1
        36,  # fftData_BC_ifft_CV
        85,  # fftData_BC_ifft_GR
        86,  # fftData_BC_ifft_GR_CV
    ]

    featuresToTestDict["gaussianFeatures2"] = [
        77,  # "gausData2"
        # 10,  # dataGCV2
        # 95,  # gausData2_CV
        # 19,  # gausData_CV_BC
        78,  # gausData2_GR
        79,  # gausData2_GR_BC
        83,  # gausData2_GR_CV
        80,  # gausData2_BC
        # 81,  # gausData2_CV_BC
        82,  # gausData2_GR_CV_BC
        # 84,  # gausData2_BC_GR_CV
    ]

    # featuresToTestDict["entropyFeatures"] = [
    #     59,  # chanEntr
    #     # 10,  # dataGCV2
    #     60,  # timeEntr
    #     # 61,  # timeEntr_CV
    #     68,  # chanEntr_GR
    #     69,  # chanEntr_GR_BC
    #     62,  # chanEntr_BC
    #     63,  # timeEntr_BC
    #     # 71,  # chanEntr_GR_BC_CV
    #     # 64,  # timeEntr_CV_BC
    # ]

    # featuresToTestDict["corrFeatures"] = [
    #     33,  # dataCorr2ax1d
    # ]

    featuresToTestList = []
    for featGroupName, featGroup in featuresToTestDict.items():
        print(featGroupName)
        featuresToTestList.extend(featGroup)

    for ind, featureI in enumerate(featuresToTestList):
        featuresToTestList[ind] = featureI - 1

    featuresToOnlyCreateNotTest = [i - 1 for i in featuresToOnlyCreateNotTest]

    # What features that are created and tested
    featureList = [
        True,  # fftData 1
        True,  # welchData 2
        True,  # hilbertData 3 DataHR seems to not add much if any to FFT and Welch
        False,  # Powerbands 4
        False,  # FFT frequency buckets 5
        False,  # FFT fftData_CV 6
        True,  # Welch welchData_CV 7
        True,  # Hilbert hilbertData_CV 8 DataHR seems to not add much if any to FFT and Welch
        True,  # gausData  9 dataGCV
        False,  # Covariance on smoothed Data2 10 dataGCV2 SKIP
        False,  # normDatacor2x1 # SEEMS BAD 11
        True,  # fftData_BC 12 Is this one doing BC before or after? Before right. yes
        True,  # welchData_BC-BC 13
        True,  # hilbertData_BC 14 DataHR seems to not add much if any to FFT and Welch
        True,  # fftData_BC_CV 15
        True,  # welchData_BC_CV 16
        True,  # hilbertData_BC_CV 17 DataHR seems to not add much if any to FFT and Welch
        True,  # gausData_CV 18
        True,  # gausData_CV_BC 19
        True,  # gausData_BC 20
        True,  # gausData_BC_CV       21      -BC means BC before covariance
        False,  # dataFFTCV2-BC 22 With more channels
        False,  # dataGCV2-BC 23 SKIP
        False,  # 24 Correlate1dBC005s
        True,  # 25 fftData_BC_ifft
        False,  # 26 corr01s
        False,  # 27 corr02s
        False,  # 28 iFFTdataCorr1d01s-BC
        False,  # 29 iFFTdataCorr1d02s-BC
        False,  # 30 iFFTdataCorr1d005s-BC
        False,  # 31 dataCorr1d01sBC
        False,  # 32 dataCorr1d02sBC
        False,  # 33 dataCorr2ax1d #
        False,  # 34 iFFTdataCorr2ax1d005s-BC       Try all of these , with new iFFTdata
        False,  # 35 dataCorr2ax1dBC
        True,  # 36 fftData_BC_ifft_CV
        False,  # 37 anglefftData
        False,  # 38 anglefftDataBC
        False,  # 39 2dataCorr2ax1d
        False,  # 40 2dataCorr2ax1dBC
        False,  # 41 3dataCorr2ax1d
        False,  # 42 3dataCorr2ax1dBC
        False,  # 43 4dataCorr2ax1d
        False,  # 44 4dataCorr2ax1dBC
        False,  # 45 5dataCorr2ax1d
        False,  # 46 5dataCorr2ax1dBC
        False,  # 47 6dataCorr2ax1d
        False,  # 48 6dataCorr2ax1dBC
        False,  # 49 05dataCorr2ax1d
        False,  # 50 05dataCorr2ax1dBC
        True,  # 51 stftData
        True,  # 52 stftData_BC
        True,  # 53 stftData_CV
        True,  # 54 stftData_BC_CV
        True,  # 55 fftData_CV_BC
        True,  # 56 welchData_CV_BC
        True,  # 57 hilbertData_CV_BC
        True,  # 58 stftData_CV_BC
        False,  # 59 chanEntr
        False,  # 60 timeEntr
        False,  # 61 timeEntr_CV
        False,  # 62 chanEntr_BC
        False,  # 63 timeEntr_BC
        False,  # 64 timeEntr_CV_BC
        True,  # 65 stftData_GR
        True,  # 66 stftData_GR_BC
        True,  # 67 stftData_GR_CV
        False,  # 68 chanEntr_GR
        False,  # 69 chanEntr_GR_BC
        True,  # 70 stftData_GR_CV_BC
        False,  # 71 chanEntr_GR_BC_CV
        True,  # 72 gausData_GR
        True,  # 73 gausData_GR_CV
        True,  # 74 gausData_GR_CV_BC
        True,  # 75 gausData_GR_BC
        True,  # 76 gausData_BC_GR_CV
        True,  # 77 gausData2
        True,  # 78 gausData2_GR
        True,  # 79 gausData2_GR_BC
        True,  # 80 gausData2_BC
        True,  # 81 gausData2_CV_BC
        True,  # 82 gausData2_GR_CV_BC
        True,  # 83 gausData2_GR_CV
        True,  # 84 gausData2_BC_GR_CV
        True,  # 85 fftData_BC_ifft_GR
        True,  # 86 fftData_BC_ifft_GR_CV
        True,  # 87 normData
        True,  # 88 normData_GR
        True,  # 89 normData_GR_CV
        True,  # 90 normData_GR_BC
        True,  # 91 normData_GR_BC_CV
        True,  # 92 normData_GR_CV_BC
        True,  # 93 normData_BC_GR
        True,  # 94 normData_BC_GR_CV
        True,  # 95 gausData2_CV
        # True,  # FT BC IFFT 24
    ]

    ###############################################################

    if useBestFeaturesTest:
        if useBestFeaturesPerLabel:
            if sameSizeBestFeat:
                bestFeatures = np.load(
                    f"topFeaturesPerLabel/{bestFeaturesSaveFile}", allow_pickle=True)
            else:
                bestFeatures = np.load(
                    f"topFeaturesPerLabel/{bestFeaturesSaveFile}", allow_pickle=True)
        else:
            bestFeatures = np.load(
                f"topFeatures/{bestFeaturesSaveFile}", allow_pickle=True)
        worstFeatures = np.load(
            f"worstFeatures/{worstFeaturesSaveFile}", allow_pickle=True)
        print(bestFeatures)
        print(bestFeatures.shape)
        print(worstFeatures)
    else:
        bestFeatures = None

    featureListIndex = np.arange(len(featureList))
    if onlyCreateFeatures:

        while True:
            if usefeaturesToTestList:
                for featureI in featureListIndex:
                    featureList[featureI] = False
                    if featureI in featuresToTestList[featIndex * nrFCOT:(featIndex + 1) * nrFCOT]:
                        featureList[featureI] = True
            else:
                for featureI in featureListIndex:
                    featureList[featureI] = False
                    if featureI in featureList[featIndex * nrFCOT:(featIndex + 1) * nrFCOT]:
                        featureList[featureI] = True

            if (featIndex * nrFCOT) > (len(featuresToTestList) - 1):
                break

            print(featureList)
            onlyCreateFeaturesFunction(
                subjects,
                paradigm,
                signAll,
                signSolo,
                soloSignificanceThreshold,
                globalSignificanceThreshold,
                onlyUniqueFeatures,
                uniqueThresh,
                t_min,
                t_max,
                sampling_rate,
                maxCombinationAmount,
                featureList,
                useSepSubjFS,
                saveFolderName=saveFolderName,
                holdOut=holdOut,
                testNameNr=testNameNr,
            )

            featIndex = featIndex + 1

        if useWinFeat:
            featIndex = 0
            while True:
                if usefeaturesToTestList:
                    for featureI in featureListIndex:
                        featureList[featureI] = False
                        if featureI in featuresToTestList[featIndex * nrFCOT:(featIndex + 1) * nrFCOT]:
                            featureList[featureI] = True
                else:
                    for featureI in featureListIndex:
                        featureList[featureI] = False
                        if featureI in featureList[featIndex * nrFCOT:(featIndex + 1) * nrFCOT]:
                            featureList[featureI] = True

                if (featIndex * nrFCOT) > (len(featuresToTestList) - 1):
                    break

                print(featureList)
                onlyCreateFeaturesFunction(
                    subjects,
                    paradigm,
                    signAll,
                    signSolo,
                    soloSignificanceThreshold,
                    globalSignificanceThreshold,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    t_min2,
                    t_max2,
                    sampling_rate,
                    maxCombinationAmount,
                    featureList,
                    useSepSubjFS,
                    saveFolderName=saveFolderName2,
                    holdOut=holdOut,
                    testNameNr=testNameNr,
                )

                featIndex = featIndex + 1
                # print(feature)

        if useWinFeat2:
            featIndex = 0
            while True:
                if usefeaturesToTestList:
                    for featureI in featureListIndex:
                        featureList[featureI] = False
                        if featureI in featuresToTestList[featIndex * nrFCOT:(featIndex + 1) * nrFCOT]:
                            featureList[featureI] = True
                else:
                    for featureI in featureListIndex:
                        featureList[featureI] = False
                        if featureI in featureList[featIndex * nrFCOT:(featIndex + 1) * nrFCOT]:
                            featureList[featureI] = True

                if (featIndex * nrFCOT) > (len(featuresToTestList) - 1):
                    break

                print(featureList)
                onlyCreateFeaturesFunction(
                    subjects,
                    paradigm,
                    signAll,
                    signSolo,
                    soloSignificanceThreshold,
                    globalSignificanceThreshold,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    t_min3,
                    t_max3,
                    sampling_rate,
                    maxCombinationAmount,
                    featureList,
                    useSepSubjFS,
                    saveFolderName=saveFolderName3,
                    holdOut=holdOut,
                    testNameNr=testNameNr,
                )

                featIndex = featIndex + 1

    for featureI in featureListIndex:
        featureList[featureI] = False

    if usefeaturesToTestList:
        for featureI in featureListIndex:
            featureList[featureI] = False
            if featureI in featuresToTestList:
                featureList[featureI] = True
            if featureI in featuresToOnlyCreateNotTest:
                featureList[featureI] = False

    print(f"FeatureList so far: {featureList}")
    # Creating the features for each subject and putting them in a dict
    fClassDict = dict()
    fmetDict = dict()
    if useMasked is False:
        bClassDict = dict()
        bFeatClassDict = dict()
        for sub in subjects:  #

            fClassDict[f"{sub}"] = fclass.featureEClass(
                sub,
                paradigm[0],
                globalSignificance=globalSignificanceThreshold,
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
                useSepSubjFS=useSepSubjFS,
                saveFolderName=saveFolderName,
                holdOut=holdOut,

            )
            fClassDict[f"{sub}"].loadOwnData(
                t_min=t_min,
                t_max=t_max,
                sampling_rate=sampling_rate,
                twoDLabels=False,
                paradigms=paradigm[1],
            )

            fmetDict["allSame"] = svmMet.SvmMets(
                significanceThreshold=soloSignificanceThreshold,
                signAll=signAll,
                signSolo=signSolo,
                verbose=False,
                tol=tolerance,
                quickTest=quickTest,
                holdOut=holdOut,
            )
            print(f"Creating features for subject:{sub}")
            createdFeatureList, labels, correctedExists = fClassDict[f"{sub}"].getFeatures(
                featureList=featureList,
                verbose=True,
            )

            print(len(createdFeatureList))
            print(f"Printing features created so far for subject {sub}")
            for createdFeature in createdFeatureList:
                print(createdFeature[1])
            print(f"Corrected Exists = {correctedExists}")
            createdFeatureList = None
            createdFeature = None
            fClassDict[f"{sub}"].data = None

        if signAll:
            if useSepSubjFS is not True:
                allSubjFListList = []
                allSubjFLabelsList = []
                subjectsThatNeedFSelect = []
            for sub in subjects:

                if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is None:
                    if useSepSubjFS:
                        print("using Here")
                        fSelectUsingSepSubjects(
                            fClassDict,
                            globalSignificanceThreshold,
                            onlyUniqueFeatures,
                            uniqueThresh,
                            paradigm[0],
                            subjects,
                            saveFolderName=saveFolderName
                        )
                        break
                    else:
                        # fClass.getFeatureList() for all other subjects into
                        allSubjFList, allSubjFLabels = combineAllSubjects(
                            fClassDict, subjectLeftOut=sub, onlyTrain=False
                        )

                        # add allSubjFlist and Labels to list
                        allSubjFLabelsList.append(allSubjFLabels)
                        allSubjFListList.append(allSubjFList)
                        subjectsThatNeedFSelect.append(sub)

                else:
                    print(
                        f"Feature Mask Already exist for all Features for subject {sub}")

            if useSepSubjFS is not True:
                goodFeatureMaskListList = []
                for subj, features, labels in zip(
                    subjectsThatNeedFSelect, allSubjFListList, allSubjFLabelsList
                ):

                    # Maybe dp before here
                    goodFeatureMaskList = createFeatureSelectMask(
                        features,
                        labels,
                        globalSignificanceThreshold,
                        onlyUniqueFeatures,
                        uniqueThresh,
                        paradigm[0],
                        subj,
                        saveFolderName=saveFolderName
                    )

                    goodFeatureMaskListList.append(goodFeatureMaskList)

                compute3 = dask.compute(goodFeatureMaskListList)
                goodFeatureMaskListList = dask.compute(compute3)

        if useWinFeat:
            fClassDict2 = winFeatFunction(featureList=featureList, subjects=subjects, paradigm=paradigm,
                                          globalSignificanceThreshold=globalSignificanceThreshold,
                                          onlyUniqueFeatures=onlyUniqueFeatures,
                                          uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                          saveFolderName=saveFolderName2,
                                          t_min=t_min2, t_max=t_max2,
                                          sampling_rate=sampling_rate,
                                          soloSignificanceThreshold=soloSignificanceThreshold,
                                          signAll=signAll,
                                          signSolo=signSolo,
                                          tolerance=tolerance,
                                          quickTest=quickTest,
                                          holdOut=holdOut)

            for sub in subjects:
                fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
                fClassDict[f"{sub}"].addNameFeat(saveFolderName)
                fClassDict2[f"{sub}"].addNameFeat(saveFolderName2)

                fClassDict[f"{sub}"].extendFeatureList(
                    fClassDict2[f"{sub}"].getFeatureList()
                )
                if signAll:
                    fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
                        fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
                    )
                fClassDict2[f"{sub}"] = None
            fClassDict2 = None

        if useWinFeat2:
            fClassDict2 = winFeatFunction(featureList=featureList, subjects=subjects, paradigm=paradigm,
                                          globalSignificanceThreshold=globalSignificanceThreshold,
                                          onlyUniqueFeatures=onlyUniqueFeatures,
                                          uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                          saveFolderName=saveFolderName3,
                                          t_min=t_min3, t_max=t_max3,
                                          sampling_rate=sampling_rate,
                                          soloSignificanceThreshold=soloSignificanceThreshold,
                                          signAll=signAll,
                                          signSolo=signSolo,
                                          tolerance=tolerance,
                                          quickTest=quickTest,
                                          holdOut=holdOut)

            for sub in subjects:
                fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
                # fClassDict[f"{sub}"].addNameFeat(saveFolderName)
                fClassDict2[f"{sub}"].addNameFeat(saveFolderName3)

                fClassDict[f"{sub}"].extendFeatureList(
                    fClassDict2[f"{sub}"].getFeatureList()
                )
                if signAll:
                    fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
                        fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
                    )
                fClassDict2[f"{sub}"] = None
                if signAll:
                    fClassDict[f"{sub}"].createMaskedFeatureList()
            fClassDict2 = None

    if useMasked:
        sub = 1
        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            saveFolderName=saveFolderName,
            holdOut=holdOut,
        )
        fClassDict[f"{sub}"].loadOwnData(
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            paradigms=paradigm[1],
        )

        fmetDict["allSame"] = svmMet.SvmMets(
            significanceThreshold=soloSignificanceThreshold,
            signAll=signAll,
            signSolo=signSolo,
            verbose=False,
            tol=tolerance,
            quickTest=quickTest,
            holdOut=holdOut,
        )
        print(f"Creating features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict[f"{sub}"].getFeatures(
            featureList=featureList,
            verbose=True,
        )

        print(len(createdFeatureList))
        print(f"Printing features created so far for subject {sub}")
        if useBestFeaturesTest:
            newCreatedFeatureList = []
            for createdFeature in createdFeatureList:
                if createdFeature[1] not in worstFeatures:
                    newCreatedFeatureList.append(createdFeature)
                else:
                    print(f"Removed {createdFeature[1]} with worst feat check")
            createdFeatureList = newCreatedFeatureList
        else:
            for createdFeature in createdFeatureList:
                print(createdFeature[1])

        for sub in subjects:
            fClassDict[f"{sub}"] = fclass.featureEClass(
                sub,
                paradigm[0],
                globalSignificance=globalSignificanceThreshold,
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
                useSepSubjFS=useSepSubjFS,
                saveFolderName=saveFolderName,
                holdOut=holdOut,
            )
            fClassDict[f"{sub}"].loadOwnData(
                t_min=t_min,
                t_max=t_max,
                sampling_rate=sampling_rate,
                twoDLabels=False,
                paradigms=paradigm[1],
            )
            folderNames = [saveFolderName, saveFolderName2, saveFolderName3]
            for folderName in folderNames:
                fClassDict[f"{sub}"].loadAllMaskedFeatures(
                    createdFeatureList, folderName)

    for sub in subjects:
        fClassDict[f"{sub}"].setOrder(seed, testSize)
    # A for loop just running all subjects using different seeds for train/data split
    for testNr in np.arange(testSize):

        for sub in subjects:  #
            fClassDict[f"{sub}"].setTestNr(testNr)
            print(f"Starting test of subject:{sub} , testNr:{testNr}")

            # Creating masked feature List using ANOVA/cov Mask
            if signAll:
                mDataList = mixShuffleSplit(
                    fClassDict[f"{sub}"].getMaskedFeatureList(),
                    labels=fClassDict[f"{sub}"].getLabels(),
                    featureClass=fClassDict[f"{sub}"],
                    maxCombinationAmount=maxCombinationAmount,
                    bestFeatures=bestFeatures,
                    useBestFeaturesTest=useBestFeaturesTest,
                    subject=sub,
                )
            else:
                mDataList = mixShuffleSplit(
                    fClassDict[f"{sub}"].getFeatureListFlat(),
                    labels=fClassDict[f"{sub}"].getLabels(),
                    featureClass=fClassDict[f"{sub}"],
                    maxCombinationAmount=maxCombinationAmount,
                    bestFeatures=bestFeatures,
                    useBestFeaturesTest=useBestFeaturesTest,
                    subject=sub,
                )
            allResultsPerSubject = []
            # For loop of each combination of features
            # Training a SVM using each one and then saving the result

            if useMLP or sameSizeBestFeat:
                nr_jobs = 4
            else:
                nr_jobs = 9

            allResultsPerSubject = Parallel(n_jobs=nr_jobs, verbose=10)(
                delayed(testLoop)(
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    name,
                    useAda,
                    userndF,
                    useMLP,
                    useOVR,
                    fmetDict,
                    sub,
                )
                for data_train, data_test, labels_train, labels_test, name in mDataList
            )

            # Creating testInfo
            featureNames = []
            featCombos = []
            for feat in fClassDict[f"{sub}"].getMaskedFeatureList():
                featureNames.append(feat[1])
            for combo in mDataList:
                featCombos.append(combo[4])
            if quickTest:
                clist = [2.5]
            else:
                clist = [0.1, 1, 10, 100, 1000]
            if useAda:
                kernels = ["adaBoost"]
            elif useMLP:
                kernels = ["MLP"]
            else:
                kernels = ["linear", "rbf", "sigmoid"]

            hyperParams = [kernels, clist]
            testInfo = [
                featureNames,
                featCombos,
                hyperParams,  # UseSvm Mets to get hyperParams.
                paradigm[0],
            ]
            savearray = np.array(
                [testInfo, sub, allResultsPerSubject], dtype=object)

            # Saving the results
            from datetime import datetime

            now = datetime.now()
            # Month abbreviation, day and year, adding time of save to filename
            now_string = now.strftime("D--%d-%m-%Y-T--%H-%M")

            # A new save directory each day to keep track of results
            foldername = now.strftime("%d-%m")

            foldername = f"{foldername}-{repetitionValue}"
            saveDir = f"{os.getcwd()}/SavedResults/{foldername}"
            if os.path.exists(saveDir) is not True:
                os.makedirs(saveDir)
            if myOwnTest:
                for sub in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    savearray = np.array(
                        [testInfo, sub, allResultsPerSubject], dtype=object)
                    np.save(
                        f"{saveDir}/savedBest-TestNr-{testNr}-Seed-{seed}-Sub-{sub}-Dt-{now_string}",
                        savearray,
                    )
            else:
                np.save(
                    f"{saveDir}/savedBest-TestNr-{testNr}-Seed-{seed}-Sub-{sub}-Dt-{now_string}",
                    savearray,
                )


def winFeatFunction(featureList, subjects, paradigm, globalSignificanceThreshold,
                    onlyUniqueFeatures, uniqueThresh, useSepSubjFS, saveFolderName, t_min, t_max, sampling_rate,
                    soloSignificanceThreshold, signAll, signSolo, tolerance, quickTest, baselineF=False, holdOut=False):
    print(f"FeatureList so far: {featureList}")
    # Creating the features for each subject and putting them in a dict
    fClassDict2 = dict()
    fmetDict2 = dict()
    bClassDict2 = dict()
    for sub in subjects:  #

        fClassDict2[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            saveFolderName=saveFolderName,
            holdOut=holdOut,
        )
        fClassDict2[f"{sub}"].loadOwnData(
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            paradigms=paradigm[1],
        )

        fmetDict2[f"{sub}"] = svmMet.SvmMets(
            significanceThreshold=soloSignificanceThreshold,
            signAll=signAll,
            signSolo=signSolo,
            verbose=False,
            tol=tolerance,
            quickTest=quickTest,
        )
        print(f"Creating features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict2[f"{sub}"].getFeatures(
            featureList=featureList,
            verbose=True,
        )
        fClassDict2[f"{sub}"].data = None
        print(len(createdFeatureList))
        print(f"Printing features created so far for subject {sub}")
        for createdFeature in createdFeatureList:
            print(createdFeature[1])
        print(f"Corrected Exists = {correctedExists}")

    if baselineF is not True:
        if signAll:
            if useSepSubjFS is not True:
                allSubjFListList = []
                allSubjFLabelsList = []
                subjectsThatNeedFSelect = []
            for sub in subjects:

                if fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask() is None:
                    if useSepSubjFS:
                        print("using Here")
                        fSelectUsingSepSubjects(
                            fClassDict2,
                            globalSignificanceThreshold,
                            onlyUniqueFeatures,
                            uniqueThresh,
                            paradigm[0],
                            subjects,
                            saveFolderName=saveFolderName
                        )
                        break
                    else:
                        # fClass.getFeatureList() for all other subjects into
                        allSubjFList, allSubjFLabels = combineAllSubjects(
                            fClassDict2, subjectLeftOut=sub, onlyTrain=False
                        )

                        # add allSubjFlist and Labels to list
                        allSubjFLabelsList.append(allSubjFLabels)
                        allSubjFListList.append(allSubjFList)
                        subjectsThatNeedFSelect.append(sub)

                else:
                    print(
                        f"Feature Mask Already exist for all Features for subject {sub}")

            if useSepSubjFS is not True:
                goodFeatureMaskListList = []
                for subj, features, labels in zip(
                    subjectsThatNeedFSelect, allSubjFListList, allSubjFLabelsList
                ):

                    # Maybe dp before here
                    goodFeatureMaskList = createFeatureSelectMask(
                        features,
                        labels,
                        globalSignificanceThreshold,
                        onlyUniqueFeatures,
                        uniqueThresh,
                        paradigm[0],
                        subj,
                        saveFolderName=saveFolderName
                    )

                    goodFeatureMaskListList.append(goodFeatureMaskList)

                compute3 = dask.compute(goodFeatureMaskListList)
                goodFeatureMaskListList = dask.compute(compute3)
    bClassDict2 = None
    fmetDict2 = None

    return fClassDict2


def onlyCreateFeaturesFunction(
    subjects,
    paradigm,
    signAll,
    signSolo,
    soloSignificanceThreshold,
    globalSignificanceThreshold,
    onlyUniqueFeatures,
    uniqueThresh,
    t_min,
    t_max,
    sampling_rate,
    maxCombinationAmount,
    featureList,
    useSepSubjFS,
    saveFolderName,
    holdOut,
    testNameNr,
):

    # Creating the features for each subject and putting them in a dict
    fClassDict = dict()
    fmetDict = dict()
    bClassDict = dict()
    bFeatClassDict = dict()
    allFeaturesAlreadyCreated = False
    for sub in subjects:  #
        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            saveFolderName=saveFolderName,
            holdOut=holdOut,
        )

        fClassDict[f"{sub}"].loadOwnData(
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            paradigms=paradigm[1],
        )
        fmetDict["allSame"] = svmMet.SvmMets(
            significanceThreshold=soloSignificanceThreshold,
            signAll=signAll,
            signSolo=signSolo,
            verbose=False,
            tol=0.001,
            quickTest=True,
        )
        print(f"Creating features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict[f"{sub}"].getFeatures(
            featureList=featureList,
            verbose=True,
        )
        if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is not None:
            return True
        print(len(createdFeatureList))
        print(f"Printing features created so far for subject {sub}")
        for createdFeature in createdFeatureList:
            print(createdFeature[1])
        print(f"Corrected Exists = {correctedExists}")
        averageBaseline = True
        correctedExists = False
        if correctedExists is False:
            baseLineFolderNames = f"myOwnBaseline{testNameNr}"
            bClassDict[f"{sub}"] = baseLineCorrection(
                subject=sub,
                sampling_rate=sampling_rate,
                saveFolderName=f"{baseLineFolderNames}1",
                chunk=False,
                chunkAmount=1
            )
            if averageBaseline:
                bFeatClassDict1 = winFeatFunction(featureList=featureList, subjects=[sub, ], paradigm=paradigm,
                                                  globalSignificanceThreshold=globalSignificanceThreshold,
                                                  onlyUniqueFeatures=onlyUniqueFeatures,
                                                  uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                                  saveFolderName=f"{baseLineFolderNames}1",
                                                  t_min=0.5, t_max=1.5,
                                                  sampling_rate=sampling_rate,
                                                  soloSignificanceThreshold=soloSignificanceThreshold,
                                                  signAll=signAll,
                                                  signSolo=signSolo, tolerance=0.1, quickTest=True,
                                                  baselineF=True, holdOut=holdOut)
                bFeatClassDict2 = winFeatFunction(featureList=featureList, subjects=[sub, ], paradigm=paradigm,
                                                  globalSignificanceThreshold=globalSignificanceThreshold,
                                                  onlyUniqueFeatures=onlyUniqueFeatures,
                                                  uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                                  saveFolderName=f"{baseLineFolderNames}2",
                                                  t_min=1.5, t_max=2.5,
                                                  sampling_rate=sampling_rate,
                                                  soloSignificanceThreshold=soloSignificanceThreshold,
                                                  signAll=signAll,
                                                  signSolo=signSolo, tolerance=0.1, quickTest=True,
                                                  baselineF=True, holdOut=holdOut)
                bFeatClassDict3 = winFeatFunction(featureList=featureList, subjects=[sub, ], paradigm=paradigm,
                                                  globalSignificanceThreshold=globalSignificanceThreshold,
                                                  onlyUniqueFeatures=onlyUniqueFeatures,
                                                  uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                                  saveFolderName=f"{baseLineFolderNames}3",
                                                  t_min=2.5, t_max=3.5,
                                                  sampling_rate=sampling_rate,
                                                  soloSignificanceThreshold=soloSignificanceThreshold,
                                                  signAll=signAll,
                                                  signSolo=signSolo, tolerance=0.1, quickTest=True,
                                                  baselineF=True, holdOut=holdOut)
                features1 = bFeatClassDict1[f"{sub}"].getFeatureList()
                features2 = bFeatClassDict2[f"{sub}"].getFeatureList()
                features3 = bFeatClassDict3[f"{sub}"].getFeatureList()
                averageFeatures = dp(
                    bFeatClassDict1[f"{sub}"].getFeatureList())
                for f1, f2, f3, af in zip(features1, features2, features3, averageFeatures):
                    feat1, feat2, feat3 = f1[0], f2[0], f3[0]
                    allFeats = np.concatenate(
                        [np.expand_dims(feat1, axis=0), np.expand_dims(feat2, axis=0),
                         np.expand_dims(feat3, axis=0)], axis=0)

                    af[0] = np.mean(allFeats, axis=0)

                features4045 = averageFeatures
            else:
                bFeatClassDict = winFeatFunction(featureList=featureList, subjects=[sub, ], paradigm=paradigm,
                                                 globalSignificanceThreshold=globalSignificanceThreshold,
                                                 onlyUniqueFeatures=onlyUniqueFeatures,
                                                 uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                                 saveFolderName="myOwnBaseline",
                                                 t_min=1.5, t_max=2.5,
                                                 sampling_rate=sampling_rate,
                                                 soloSignificanceThreshold=soloSignificanceThreshold,
                                                 signAll=signAll,
                                                 signSolo=signSolo, tolerance=0.1, quickTest=True,
                                                 baselineF=True, holdOut=holdOut)

                features4045 = bFeatClassDict[f"{sub}"].getFeatureList()

            fClassDict[f"{sub}"].correctedFeatureList = bClassDict[
                f"{sub}"
            ].baselineCorrect(
                fClassDict[f"{sub}"].getFeatureList(),
                features4045,
                fClassDict[f"{sub}"].paradigmName,
                saveFolderName
            )

            print(
                f"Creating features for subject:{sub} after baseline correction")

            createdFeatureList, labels, correctedExists = fClassDict[
                f"{sub}"
            ].getFeatures(
                featureList=featureList,
                verbose=True,
            )

    if signAll:
        if useSepSubjFS is not True:
            allSubjFListList = []
            allSubjFLabelsList = []
            subjectsThatNeedFSelect = []
        for sub in subjects:

            if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is None:
                if useSepSubjFS:
                    print("fSelectSepSub Place Used here")
                    fSelectUsingSepSubjects(
                        fClassDict,
                        globalSignificanceThreshold,
                        onlyUniqueFeatures,
                        uniqueThresh,
                        paradigm[0],
                        subjects,
                        saveFolderName=saveFolderName
                    )
                    break
                else:
                    allSubjFList, allSubjFLabels = combineAllSubjects(
                        fClassDict, subjectLeftOut=sub, onlyTrain=False
                    )

                    # add allSubjFlist and Labels to list
                    allSubjFLabelsList.append(allSubjFLabels)
                    allSubjFListList.append(allSubjFList)
                    subjectsThatNeedFSelect.append(sub)

            else:

                print(
                    f"Feature Mask Already exist for all Features for subject {sub}")

        if useSepSubjFS is not True:

            goodFeatureMaskListList = []
            for subj, features, labels in zip(
                subjectsThatNeedFSelect, allSubjFListList, allSubjFLabelsList
            ):

                goodFeatureMaskList = createFeatureSelectMask(
                    features,
                    labels,
                    globalSignificanceThreshold,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    paradigm[0],
                    subj,
                    saveFolderName=saveFolderName
                )

                goodFeatureMaskListList.append(goodFeatureMaskList)

            compute3 = dask.compute(goodFeatureMaskListList)
            goodFeatureMaskListList = dask.compute(compute3)

    del fClassDict, fmetDict, bClassDict
    print("Done with one loop of onlyCreateFeatures function")
    return True


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats()
    with open("testStats.txt", "w+") as f:
        f.write(s.getvalue())
