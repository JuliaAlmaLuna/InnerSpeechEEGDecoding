"""
This class runs a pipeline testing SVM classification on data
"""
from joblib import Parallel, delayed
from copy import deepcopy as dp
import numpy as np
import feature_extractionClean as fclass
# from baselineClean import baseLineCorrection
from baseCorrectWithEnd import baseLineCorrection
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
        allResults = fmetDict["allSame"].testSuiteOVR(
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

# Added to test linux github
# This class prints to separate files for each subject when using dask multiprocessing.


def printProcess(processName, printText, saveFolderName):
    if os.path.exists(f"{os.getcwd()}/{saveFolderName}/processOutputs/") is not True:
        os.makedirs(f"{os.getcwd()}/{saveFolderName}/processOutputs")
    with open(f"{os.getcwd()}/{saveFolderName}/processOutputs/{processName}Output.txt", "a") as f:
        print(printText, file=f)


def loadAnovaMaskNoClass(
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


def saveAnovaMaskNoClass(
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
def delayedAnovaPart(
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
        # TODO Only part after here needs to be dasked. It needs goodfeature, indexList and goodData,
        # and returns new goodData. It can
        # be used in this way: Send in featurename as well. Return goodData and featureName.
        # Use this to save it correctly afterwards
        # So return will be two things. List of names/subject, and list of goodData arrays.

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
    # This one cant be here then
    saveAnovaMaskNoClass(
        featurename=featureName,
        maskname=f"sign{significanceThreshold}",
        array=goodData,
        uniqueThresh=uniqueThresh,
        paradigmName=paradigmName,
        subject=subject,
        onlyUniqueFeatures=True,
        saveFolderName=saveFolderName
    )


def anovaTest(
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
        loadedMask = loadAnovaMaskNoClass(
            featurename=featureName,
            maskname=f"sign{significanceThreshold}",
            uniqueThresh=uniqueThresh,
            paradigmName=paradigmName,
            subject=subject,
            onlyUniqueFeatures=onlyUniqueFeatures,
            saveFolderName=saveFolderName
        )

        if loadedMask is None:
            # collect garbage on all workers
            flatfeature = np.reshape(feature[0], [feature[0].shape[0], -1])

            scaler.fit(flatfeature)
            flatfeature = scaler.transform(flatfeature)

            # Running the ANOVA Test
            # Try selectFWE or selectFPR as well maybe.
            f_statistic, p_values = feature_selection.f_classif(
                flatfeature, labels)

            # varSelect = feature_selection.VarianceThreshold(0.01)
            # varSelect.fit(flatfeature)
            # varMask = varSelect.get_support()
            # # goodVarflatfeature = varSelect.transform(flatfeature)
            # printProcess(
            #     f"subj{subject}output",
            #     f"varMask Nonzero amount {np.count_nonzero(varMask)}", saveFolderName
            # )

            # Create a mask of features with P values below threshold
            p_values[p_values > significanceThreshold] = 0
            p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2

            goodData = f_statistic * p_values  # * varMask

            remainingNrOfFeatures = np.count_nonzero(goodData)
            if remainingNrOfFeatures > 17000:
                ratioKeep = int(17000 / len(goodData) * 100)
                # keepThresh = np.percentile(goodData, round(ratioKeep))
                bestPercentile = feature_selection.SelectPercentile(
                    feature_selection.f_classif, percentile=ratioKeep
                )
                bestPercentile.fit(flatfeature, labels)
                # goodData[goodData < keepThresh] = 0
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
                # Should apply to only the last two axises.

            feature[0] = None

            # If onlyUniqueFeatures, then uses covcorrelation matrix to remove too similar features from mask.
            if onlyUniqueFeatures:
                # This function is dask delayed so when called later compute() to multiprocess it.
                goodData = delayedAnovaPart(
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
            ovrLabels = fClassDict[f"{sub}"].getLabels()
            ovrLabels[ovrLabels != label] = 5
            ovrLabels[ovrLabels == label] = 0
            ovrLabels[ovrLabels != 0] = 1
            # fClassDict[f"{sub}"].labels[fClassDict[f"{sub}"].labels != label] = 5
            goodFeatureMaskList = anovaTest(
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
                        anovaMask = loadAnovaMaskNoClass(
                            featurename=feature[1],
                            maskname=f"sign{globalSignificanceThreshold}",
                            uniqueThresh=uniqueThresh,
                            paradigmName=f"{paradigmName}-label-{label}-soloFSSub",
                            subject=sub2,
                            onlyUniqueFeatures=onlyUniqueFeatures,
                            saveFolderName=saveFolderName
                        )
                    else:
                        anovaMask = anovaMask + loadAnovaMaskNoClass(
                            featurename=feature[1],
                            maskname=f"sign{globalSignificanceThreshold}",
                            uniqueThresh=uniqueThresh,
                            paradigmName=f"{paradigmName}-label-{label}-soloFSSub",
                            subject=sub2,
                            onlyUniqueFeatures=onlyUniqueFeatures,
                            saveFolderName=saveFolderName
                        )
            saveAnovaMaskNoClass(
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
    # print(allSubjFList[0][0].shape)  # Shape of all trials, feature 1
    print(f"{allSubjFLabels.shape} trials combined")

    return allSubjFList, allSubjFLabels


# This function is or should be the same as in main() but it creates chunked features
# Should be done in a more compact way, which doesnt need another function fully like this.
def createChunkFeatures(
    chunkAmount,
    onlyUniqueFeatures,
    globalSignificanceThreshold,
    uniqueThresh,
    paradigm,
    useSepSubjFS,
    allFeaturesList,
    featuresToTestList,
    useAllFeatures,
    saveFolderName,
    t_min,
    t_max,
    sampling_rate=256,
    subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9],

):
    # Fix it so chunkFeatures are not touched by not chunk functions . And checks alone
    # Remove this when featuresToTestList Works
    badFeatures = [
        1,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        18,
        19,
        20,
        22,
        23,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
    ]
    featureList = allFeaturesList

    for ind, fea in enumerate(badFeatures):
        badFeatures[ind] = fea - 1

    # Just to fix numbering system from starting at 1 to starting at zero
    for ind, fea in enumerate(featuresToTestList):
        featuresToTestList[ind] = fea - 1

    featureListIndex = np.arange(len(featureList))

    if useAllFeatures:
        for featureI in featureListIndex:
            if featureI in featuresToTestList:
                featureList[featureI] = True
            else:
                featureList[featureI] = False

        for featureI in featureListIndex:
            featureList[featureI] = False

        for featureI in featureListIndex:
            if featureI in badFeatures:
                continue
            featureList[featureI] = True

    # Creating the features for each subject and putting them in a dict
    fClassDict2 = dict()
    bClassDict2 = dict()
    for sub in subjects:  #

        fClassDict2[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            chunk=True,
            chunkAmount=chunkAmount,
            onlyUniqueFeatures=onlyUniqueFeatures,  # Doesn't matter if chunk = False
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            saveFolderName=saveFolderName
        )
        fClassDict2[f"{sub}"].loadData(
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            paradigms=paradigm[1],
        )

        print(f"Creating chunk features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict2[f"{sub}"].getFeatures(
            featureList=featureList,
            verbose=True,
        )
        print(f"Nr of Features created to far: {len(createdFeatureList)}")
        for createdFeature in createdFeatureList:
            print(createdFeature[1])

        print(f"Baseline corrected features exists = {correctedExists}")
        if correctedExists is False:
            print("Since baseline corrected features did not exist, creating them now")
            bClassDict2[f"{sub}"] = baseLineCorrection(
                subject=sub,
                sampling_rate=sampling_rate,
                chunk=True,
                chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            )

            bClassDict2[f"{sub}"].loadBaselineData()

            bClassDict2[f"{sub}"].getBaselineFeatures(
                trialSampleAmount=fClassDict2[f"{sub}"].getOrigData().shape[2],
                featureList=featureList,
            )

            fClassDict2[f"{sub}"].correctedFeatureList = bClassDict2[
                f"{sub}"
            ].baselineCorrect(
                fClassDict2[f"{sub}"].getFeatureList(),
                fClassDict2[f"{sub}"].getLabelsAux(),
                fClassDict2[f"{sub}"].paradigmName,
            )

            print(f"Creating features again after BC for subject:{sub}")
            createdFeatureList, labels, correctedExists = fClassDict2[
                f"{sub}"
            ].getFeatures(
                featureList=featureList,
                verbose=True,
            )
            print("But I dont care about BC not existing here for some reason")
    bClassDict2 = None
    return fClassDict2


def main():
    import psutil
    p = psutil.Process()
    p.cpu_affinity(list(np.arange(18)))
    p.cpu_affinity()
    ##############################################################
    # Testloop parameters
    # paradigm = paradigmSetting.upDownRightLeftInnerSpecialPlot()
    # paradigm = paradigmSetting.upDownRightLeftInner()
    # paradigm = paradigmSetting.upDownRightLeftVis()
    # paradigm = paradigmSetting.upDownVisFixedWorse()
    # paradigm = paradigmSetting.upDownInnerFixed()
    # paradigm = paradigmSetting.upDownInnerSpecial()
    # paradigm = paradigmSetting.upDownVisInnersep()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.rightLeftInner()
    # paradigm = paradigmSetting.rightLeftVis()
    # paradigm = paradigmSetting.upDownRightLeftVis()
    # paradigm = paradigmSetting.rightLeftVis()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.upDownRightLeftInner()
    paradigm = paradigmSetting.upDownInner()
    # paradigm = paradigmSetting.rightLeftInner()
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    testSize = 7  # Nr of seed iterations until stopping
    seed = 39  # Arbitrary, could be randomized as well.
    validationRepetition = True
    useAllFeatures = True
    chunkFeatures = False
    # "peak4-const3-i-ud-global-10-3c"  # "udrliplotnoAda1hyperparams"
    signAll = True
    globalSignificanceThreshold = 0.1  # 0.1  #
    useSepSubjFS = True
    if useSepSubjFS:
        # CHANGED FROM 0.01 REMEMBER. Goes too slow for larger ones? Maybe not now with masked feature tactic?
        # If it helps. Maybe find 0.02 or 0.03 that helps but isnt horribly slow! Honestly. Lower seems better? Somehow!
        globalSignificanceThreshold = 0.01
    # Currently the best. Try with lower fselect threshold and usesepsubjects
    cmbSize = 7
    paraName = paradigm[0]
    repetitionName = f"{paraName}{cmbSize}cOnlySepOnlyCurr01th"
    repetitionValue = f"{49}{repetitionName}"
    onlyCreateFeatures = False
    useBestFeaturesTest = True
    useMasked = True
    # When increasing combination amount by one each test.
    bestFeaturesSaveFile = f"top{cmbSize-1}{paraName}.npy"
    worstFeaturesSaveFile = f"worstFeats1{paraName}.npy"
    quickTest = True
    ##############################################################
    # Loading parameters, what part of the trials to load and test
    # saveFolderName = "peakTime2"
    # t_min = 1.1
    # t_max = 1.7
    # sampling_rate = 256
    # saveFolderName2 = "constTime2"
    # useWinFeat = True
    # t_min2 = 1.7
    # t_max2 = 2.9
    maxCombinationAmount = cmbSize
    saveFolderName = "peakTime4SortedSep14"
    t_min = 1.1
    t_max = 1.6
    sampling_rate = 256
    saveFolderName2 = "constTimeSortedSep14"
    useWinFeat = True
    t_min2 = 1.6
    t_max2 = 2.1
    saveFolderName3 = "lateTimeSortedSep14"
    useWinFeat2 = True
    t_min3 = 2.1
    t_max3 = 2.6
    # saveFolderName2 = "baseLineTime"
    # useWinFeat2 = True
    # t_min2 = 4.0
    # t_max2 = 4.5
    ##############################################################
    # Feature selection parameters
    onlyUniqueFeatures = True
    uniqueThresh = 0.9

    signSolo = False
    soloSignificanceThreshold = 0.005
    # Does not seem to help at all. Could be useful for really individual features.

    ################################################################
    # Sklearn/TestTrain parameters
    useAda = False  # Using ADA
    userndF = False  # Sklearn random forest, works a little worse and a little slower than SVM at this point
    useMLP = False  # Sklearn MLP, not made good yet. Works ok
    useOVR = True
    tolerance = 0.001  # Untested
    ################################################################
    # Feature creation/extraction parameters
    chunkAmount = 3
    nrFCOT = 2  # nrOfFeaturesToCreateAtOneTime
    featIndex = 0  # Multiplied by nrFCOT, First features to start creating
    usefeaturesToTestList = True
    featuresToTestDict = dict()
    stftSplit = 8  # Not used
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

    # # # # # featuresToTestDict["welchFeatures"] = [
    # # # # #     2,  # welchData
    # # # # #     7,  # welchData_CV
    # # # # #     13,  # welchData_BC
    # # # # #     16,  # welchData_BC_CV
    # # # # #     56,  # welchData_CV_BC
    # # # # # ]
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

    # featuresToTestDict["inversefftFeatures"] = [
    #     25,  # fftData_BC_ifft
    #     # 28,  # fftData_BC_ifft_cor2x1
    #     # 29,  # fftData_BC_ifft_cor2x2
    #     # 30,  # fftData_BC_ifft_cor2x3
    #     # 34,  # fftData_BC_ifft_cor1x1
    #     36,  # fftData_BC_ifft_CV
    #     85,  # fftData_BC_ifft_GR
    #     86,  # fftData_BC_ifft_GR_CV
    # ]

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
        False,  # dataFFTCV2-BC 22 With more channels. Only useful for chunks
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
        # More to be added
    ]

    ###############################################################

    if useBestFeaturesTest:
        bestFeatures = np.load(
            f"topFeatures/{bestFeaturesSaveFile}", allow_pickle=True)
        worstFeatures = np.load(
            f"worstFeatures/{worstFeaturesSaveFile}", allow_pickle=True)
        print(bestFeatures)
        print(bestFeatures.shape)
    else:
        bestFeatures = None

    # For loops to fit with numbering from 0 to len()-1 instead of 1 to len()
    # for ind, fea in enumerate(badFeatures):
    #     badFeatures[ind] = fea - 1

    # print(badFeatures)

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
                chunkFeatures,
                chunkAmount,
                onlyUniqueFeatures,
                uniqueThresh,
                t_min,
                t_max,
                sampling_rate,
                maxCombinationAmount,
                featureList,
                useSepSubjFS,
                stftSplit=stftSplit,
                saveFolderName=saveFolderName
            )

            featIndex = featIndex + 1

        # If Chunkfeatures, run create again but for non Chunk features
        if useWinFeat:
            featIndex = 0
            # Turn it off to create None chunk features as well
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
                    chunkFeatures,
                    chunkAmount,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    t_min2,
                    t_max2,
                    sampling_rate,
                    maxCombinationAmount,
                    featureList,
                    useSepSubjFS,
                    stftSplit=stftSplit,
                    saveFolderName=saveFolderName2
                )

                featIndex = featIndex + 1
                # print(feature)

        if useWinFeat2:
            featIndex = 0
            # Turn it off to create None chunk features as well
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
                    chunkFeatures,
                    chunkAmount,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    t_min3,
                    t_max3,
                    sampling_rate,
                    maxCombinationAmount,
                    featureList,
                    useSepSubjFS,
                    stftSplit=stftSplit,
                    saveFolderName=saveFolderName3
                )

                featIndex = featIndex + 1

    if useAllFeatures:
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
                chunk=False,
                chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
                useSepSubjFS=useSepSubjFS,
                stftSplit=stftSplit,
                saveFolderName=saveFolderName,
            )
            fClassDict[f"{sub}"].loadData(
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
                    goodFeatureMaskList = anovaTest(
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
                                          signSolo=signSolo, tolerance=tolerance, quickTest=quickTest)

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
                                          signSolo=signSolo, tolerance=tolerance, quickTest=quickTest)

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
                fClassDict[f"{sub}"].createMaskedFeatureList()
            fClassDict2 = None

    if useMasked:
        sub = 1
        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            chunk=False,
            chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            stftSplit=stftSplit,
            saveFolderName=saveFolderName,
        )
        fClassDict[f"{sub}"].loadData(
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
                chunk=False,
                chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
                useSepSubjFS=useSepSubjFS,
                stftSplit=stftSplit,
                saveFolderName=saveFolderName,
            )
            fClassDict[f"{sub}"].loadData(
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
        # print(f"Corrected Exists = {correctedExists}")
        # createdFeatureList = None
        # createdFeature = None
        # fClassDict[f"{sub}"].data = None

    for sub in subjects:
        fClassDict[f"{sub}"].setOrder(seed, testSize)
    # A for loop just running all subjects using different seeds for train/data split
    for testNr in np.arange(testSize):

        for sub in subjects:  #
            fClassDict[f"{sub}"].setTestNr(testNr)
            print(f"Starting test of subject:{sub} , testNr:{testNr}")

            # Creating masked feature List using ANOVA/cov Mask
            # signAll = True
            # Then only create new combos containing that best combos + 1 or 2 more features
            if signAll:
                # print(fClassDict[f"{sub}"].getLabels())
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

            if useMLP:
                nr_jobs = 1
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

            if validationRepetition:
                foldername = f"{foldername}-{repetitionValue}"
            saveDir = f"{os.getcwd()}/SavedResults/{foldername}"
            if os.path.exists(saveDir) is not True:
                os.makedirs(saveDir)

            np.save(
                f"{saveDir}/savedBest-TestNr-{testNr}-Seed-{seed}-Sub-{sub}-Dt-{now_string}",
                savearray,
            )


def winFeatFunction(featureList, subjects, paradigm, globalSignificanceThreshold,
                    onlyUniqueFeatures, uniqueThresh, useSepSubjFS, saveFolderName, t_min, t_max, sampling_rate,
                    soloSignificanceThreshold, signAll, signSolo, tolerance, quickTest, baselineF=False):
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
            chunk=False,
            chunkAmount=1,  # Doesn't matter if chunk = False
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            stftSplit=8,  # Dont use this
            saveFolderName=saveFolderName,
        )
        fClassDict2[f"{sub}"].loadData(
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
                    goodFeatureMaskList = anovaTest(
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
    chunkFeatures,
    chunkAmount,
    onlyUniqueFeatures,
    uniqueThresh,
    t_min,
    t_max,
    sampling_rate,
    maxCombinationAmount,
    featureList,
    useSepSubjFS,
    stftSplit,
    saveFolderName,
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
            chunk=chunkFeatures,
            chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
            stftSplit=stftSplit,
            saveFolderName=saveFolderName
        )

        fClassDict[f"{sub}"].loadData(
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

        correctedExists = False
        if correctedExists is False:

            bClassDict[f"{sub}"] = baseLineCorrection(
                subject=sub,
                sampling_rate=sampling_rate,
                chunk=False,
                chunkAmount=chunkAmount,
                saveFolderName="afterBaselineSortedSep13"  # Doesn't matter if chunk = False
            )

            bFeatClassDict = winFeatFunction(featureList=featureList, subjects=[sub, ], paradigm=paradigm,
                                             globalSignificanceThreshold=globalSignificanceThreshold,
                                             onlyUniqueFeatures=onlyUniqueFeatures,
                                             uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
                                             saveFolderName="afterBaselineSortedSep13",
                                             t_min=4, t_max=4.492,
                                             sampling_rate=sampling_rate,
                                             soloSignificanceThreshold=soloSignificanceThreshold,
                                             signAll=signAll,
                                             signSolo=signSolo, tolerance=0.1, quickTest=True,
                                             baselineF=True)

            # bFeatClassDict2 = winFeatFunction(featureList=featureList, subjects=[sub, ], paradigm=paradigm,
            #                                   globalSignificanceThreshold=globalSignificanceThreshold,
            #                                   onlyUniqueFeatures=onlyUniqueFeatures,
            #                                   uniqueThresh=uniqueThresh, useSepSubjFS=useSepSubjFS,
            #                                   saveFolderName="afterBaseline2Sorted6",
            #                                   t_min=3.1, t_max=3.592,
            #                                   sampling_rate=sampling_rate,
            #                                   soloSignificanceThreshold=soloSignificanceThreshold,
            #                                   signAll=signAll,
            #                                   signSolo=signSolo, tolerance=0.1, quickTest=True,
            #                                   baselineF=True)

            features4045 = bFeatClassDict[f"{sub}"].getFeatureList()
            # features3136 = bFeatClassDict2f"{sub}"].getFeatureList()
            features4045Two = bFeatClassDict[f"{sub}"].getFeatureList()
            # features3136Two = bFeatClassDict2[f"{sub}"].getFeatureList()
            # zip(features4045, features4045Two):
            # for fInd, feat in enumerate(features4045):

            #     for ind, trial in enumerate(feat[0][:-1], 0):

            #         onlytrial1 = features4045Two[fInd][0][ind - 1]
            #         # onlytrial2 = features4045Two[fInd][0][ind]
            #         onlytrial3 = features4045Two[fInd][0][ind + 1]

            #         # onlytrial1x = features3136Two[fInd][0][ind - 1]
            #         # onlytrial2x = features3136Two[fInd][0][ind]
            #         # onlytrial3x = features3136Two[fInd][0][ind + 1]

            #         print(feat[0].shape)
            #         onlytrial1 = np.expand_dims(onlytrial1, axis=0)
            #         # onlytrial2 = np.expand_dims(onlytrial2, axis=0)
            #         onlytrial3 = np.expand_dims(onlytrial3, axis=0)
            #         # onlytrial1x = np.expand_dims(onlytrial1x, axis=0)
            #         # onlytrial2x = np.expand_dims(onlytrial2x, axis=0)
            #         # onlytrial3x = np.expand_dims(onlytrial3x, axis=0)
            #         print(onlytrial1.shape)
            #         together = np.concatenate(
            #             [onlytrial1, onlytrial3], axis=0)  # o onlytrial3
            #         print(together.shape)
            #         avgBaselineTrial = np.mean(together, axis=0)
            #         print(avgBaselineTrial.shape)
            #         feat[0][ind] = avgBaselineTrial
            #         print(feat[0][ind].shape)
            #         print("HeyJulia")
            #     print(feat[0].shape)

            # for index, feat in  enumerate(features4045,0)  #  zip(features4045, features4045Two):
            #     onlyfeat1, onlyfeat2 = feat1[0], feat2[0]
            #     print(feat1[0].shape)
            #     onlyfeat1 = np.expand_dims(onlyfeat1, axis=0)
            #     onlyfeat2 = np.expand_dims(onlyfeat2, axis=0)
            #     print(onlyfeat1.shape)
            #     together = np.concatenate([onlyfeat1, onlyfeat2], axis=0)
            #     print(together.shape)
            #     avgBaseline = np.mean(together, axis=0)
            #     print(avgBaseline.shape)
            #     feat1[0] = avgBaseline
            #     print(feat1[0].shape)
            #     print("HeyJulia")

            # bClassDict[f"{sub}"].saveFolderName = saveFolderName
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
        # if fClassDict[f"{sub}"].allFeaturesAlreadyCreated is True:
        #     allFeaturesAlreadyCreated = True
        #     return True

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

                # Maybe dp before here
                goodFeatureMaskList = anovaTest(
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
            # print(compute3)
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
