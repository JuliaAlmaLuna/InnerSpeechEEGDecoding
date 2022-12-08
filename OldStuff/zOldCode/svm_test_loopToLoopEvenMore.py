"""
This class runs a pipeline testing SVM classification on data
"""
from joblib import Parallel, delayed
from copy import deepcopy as dp
import numpy as np
import feature_extraction as fclass
from baselineBefore import baseLineCorrection
import svmMethods as svmMet
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler
import paradigmSetting
import cProfile
import pstats
import io
import time

import glob
import os
import dask


def testLoop(
    data_train,
    data_test,
    labels_train,
    labels_test,
    name,
    testNr,
    testSize,
    count,
    lengthDataList,
    useAda,
    fmetDict,
    sub,
):
    # print(f"\n Running dataset: {name} \n")
    # print(
    #     f" Test {testNr}/{testSize} - Progress {count}/{lengthDataList}")
    # count = count + 1

    # Below here can be switch to NN ? Create method? Or just different testSuite. Right now using Adaboost.
    # TODO, use joblib parallel to spread this over as many cpu as possible
    # Would say 4 or 5 is reasonable.
    if useAda:

        allResults = fmetDict[f"{sub}"].testSuiteAda(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )
    else:
        allResults = fmetDict[f"{sub}"].testSuite(
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
    order,
    featureClass,
    maxCombinationAmount,
    bestFeatures,
    useBestFeaturesTest,
):

    # Copy labels and features list to avoid changes to originals. Probly not needed
    tempLabels = dp(labels)
    tempFeatureList = dp(createdFeatureList)

    mDataList = featureClass.createListOfDataMixes(
        featureList=tempFeatureList,
        labels=tempLabels,
        order=order,
        maxCombinationAmount=maxCombinationAmount,
        bestFeatures=bestFeatures,
        useBestFeaturesTest=useBestFeaturesTest,
    )
    return mDataList


# This class prints to separate files for each subject when using dask multiprocessing.
def printProcess(processName, printText):
    with open(f"processOutputs/{processName}Output.txt", "a") as f:
        print(printText, file=f)


def loadAnovaMaskNoClass(
    featurename, maskname, uniqueThresh, paradigmName, subject, onlyUniqueFeatures
):
    name = f"{featurename}{maskname}"
    if onlyUniqueFeatures:
        name = f"{name}u{uniqueThresh}"

    saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedAnovaMask/sub-{subject}-par-{paradigmName}"
    path = glob.glob(saveDir + f"/{name}.npy")
    if len(path) > 0:
        savedAnovaMask = np.load(path[0], allow_pickle=True)
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
):
    name = f"{featurename}{maskname}"

    if onlyUniqueFeatures:
        name = f"{name}u{uniqueThresh}"

    saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedAnovaMask/sub-{subject}-par-{paradigmName}"
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
        printProcess(f"subj{subject}output", " Too few good features")
        printProcess(f"subj{subject}output", goodfeature.shape)
        goodData = np.zeros(goodData.shape)
    else:

        printProcess(f"subj{subject}output", time.clock())
        # Create a corrcoef matrix of the features, comparing them to one another
        corrMat = np.corrcoef(goodfeature)
        goodfeature = None
        printProcess(f"subj{subject}output", time.clock())

        printProcess(f"subj{subject}output", corrMat.shape)

        # Keep only everything above diagonal
        halfCorrMat = np.triu(corrMat, 1)

        corrMat = None
        # Create list of all features that are too correlated, except one of the features ( the one with lower index)
        deleteIndexes = np.where(halfCorrMat > uniqueThresh)[1]
        halfCorrMat = None

        printProcess(
            f"subj{subject}output",
            f"{np.count_nonzero(goodData)} good Features \
                            before covRemoval:{uniqueThresh}in {featureName}",
        )
        # Delete these features from goodData mask
        goodData[indexList[deleteIndexes]] = 0
        printProcess(
            f"subj{subject}output",
            f"{np.count_nonzero(goodData)} good Features \
                            after covRemoval:{uniqueThresh} in {featureName}",
        )

    saveAnovaMaskNoClass(
        featurename=featureName,
        maskname=f"sign{significanceThreshold}",
        array=goodData,
        uniqueThresh=uniqueThresh,
        paradigmName=paradigmName,
        subject=subject,
        onlyUniqueFeatures=True,
    )


def anovaTest(
    featureList,
    labels,
    significanceThreshold,
    onlyUniqueFeatures,
    uniqueThresh,
    paradigmName,
    subject,
):

    printProcess(
        f"subj{subject}output",
        f"Running anova Test and masking using sign threshold: {significanceThreshold}",
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
        )

        if loadedMask is None:

            # if "CV" in featureName:
            #     feature[0] = np.triu

            flatfeature = np.reshape(feature[0], [feature[0].shape[0], -1])

            scaler.fit(flatfeature)
            flatfeature = scaler.transform(flatfeature)

            # Running the ANOVA Test
            f_statistic, p_values = feature_selection.f_classif(flatfeature, labels)

            # Create a mask of features with P values below threshold
            p_values[p_values > significanceThreshold] = 0
            p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2

            goodData = f_statistic * p_values

            # Use covcorrelation matrix to remove too similar features from mask.
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
                )

        else:
            printProcess(f"subj{subject}output", f"Loaded mask {featureName}")
            goodData = loadedMask

        goodFeatureMaskList.append(goodData)
        goodData = None
        feature[0] = None

    return goodFeatureMaskList


# This function creates Fselect masks for each subject, then adds all the other subjects ( except one subject ) masks
# Does not seem to help much, but keep it. To check later. Maybe with fixes will be better!
def fSelectUsingSepSubjects(
    fClassDict,
    globalSignificanceThreshold,
    onlyUniqueFeatures,
    uniqueThresh,
    paradigmName,
    subjects,
):
    goodFeatureMaskListList = []

    for sub in subjects:
        goodFeatureMaskList = anovaTest(
            featureList=fClassDict[f"{sub}"].getFeatureList(),
            labels=fClassDict[f"{sub}"].getLabels(),
            significanceThreshold=globalSignificanceThreshold,
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=uniqueThresh,
            paradigmName=f"{paradigmName}soloFSSub",
            subject=sub,
        )
        goodFeatureMaskListList.append(goodFeatureMaskList)

    compute3 = dask.compute(goodFeatureMaskListList)
    goodFeatureMaskListList = dask.compute(compute3)

    for sub in subjects:
        for feature in fClassDict[f"{sub}"].getFeatureList():
            anovaMask = None
            for sub2 in subjects:
                if sub2 == sub:
                    continue
                if anovaMask is None:
                    anovaMask = loadAnovaMaskNoClass(
                        featurename=feature[1],
                        maskname=f"sign{globalSignificanceThreshold}",
                        uniqueThresh=uniqueThresh,
                        paradigmName=f"{paradigmName}soloFSSub",
                        subject=sub2,
                        onlyUniqueFeatures=onlyUniqueFeatures,
                    )
                else:
                    anovaMask = anovaMask + loadAnovaMaskNoClass(
                        featurename=feature[1],
                        maskname=f"sign{globalSignificanceThreshold}",
                        uniqueThresh=uniqueThresh,
                        paradigmName=f"{paradigmName}soloFSSub",
                        subject=sub2,
                        onlyUniqueFeatures=onlyUniqueFeatures,
                    )
            saveAnovaMaskNoClass(
                featurename=feature[1],
                paradigmName=f"{paradigmName}usingSoloFsubs",
                maskname=f"sign{globalSignificanceThreshold}",
                array=anovaMask,
                subject=sub,
                onlyUniqueFeatures=onlyUniqueFeatures,
                uniqueThresh=uniqueThresh,
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
    signAll,
    signSolo,
    onlyUniqueFeatures,
    globalSignificanceThreshold,
    uniqueThresh,
    paradigm,
    useSepSubjFS,
):
    # Fix it so chunkFeatures are not touched by not chunk functions . And checks alone

    # Loading parameters, what part of the trials to load and test
    t_min = 1.8
    t_max = 3
    sampling_rate = 256

    # Parameters for ANOVA test and ANOVA Feature Mask
    # signAll = True
    # globalSignificanceThreshold = 0.05
    # onlyUniqueFeatures = True
    # All the subjects that are tested, and used to create ANOVA Mask
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,

    # What paradigm to test

    # paradigm = paradigmSetting.upDownInner()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.upDownVisSpecial()
    # paradigm = paradigmSetting.upDownRightLeftInnerSpecial()
    # paradigm = paradigmSetting.upDownRightLeftInner()
    # paradigm = paradigmSetting.upDownRightLeftVis()
    # paradigm = paradigmSetting.rightLeftInner()

    # What chunk features that are created and tested
    featureList = [
        False,  # FFT 1
        False,  # Welch 2
        False,  # Hilbert 3 DataHR seems to not add much if any to FFT and Welch
        False,  # Powerbands 4
        False,  # FFT frequency buckets 5
        False,  # FFT Covariance 6
        False,  # Welch Covariance 7
        False,  # Hilbert Covariance 8 DataHR seems to not add much if any to FFT and Welch
        False,  # Covariance on smoothed Data 9 dataGCV
        False,  # Covariance on smoothed Data2 10
        False,  # Correlate1d # SEEMS BAD 11
        False,  # dataFFTCV-BC 12 Is this one doing BC before or after? Before right. yes
        False,  # dataWCV-BC 13
        False,  # dataHRCV-BC 14 DataHR seems to not add much if any to FFT and Welch
        True,  # fftDataBC 15
        False,  # welchDataBC 16
        False,  # dataHRBC 17 DataHR seems to not add much if any to FFT and Welch
        False,  # gaussianData 18
        False,  # dataGCVBC 19
        True,  # gaussianDataBC 20
        True,  # dataGCV-BC       21      - BC means BC before covariance
        False,  # dataFFTCV2-BC 22 With more channels. Only useful for chunks
        False,  # dataGCV2-BC 23 With more channels. Only useful for chunks
        True,  # dataCorrBC 24
        # More to be added
    ]

    # badFeatures = [2, 3, 4, 5, 6, 7, 8, 9, 21, 22]
    # badFeatures = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 19, 22, 23]
    badFeatures = [4, 5, 6, 7, 8, 9, 10, 19, 21, 22, 23]
    # goodFeatures = []

    for ind, fea in enumerate(badFeatures):
        badFeatures[ind] = fea - 1

    # for ind, fea in enumerate(goodFeatures):
    #     goodFeatures[ind] = fea - 1
    featureListIndex = np.arange(len(featureList))
    useAllFeatures = True

    if useAllFeatures:
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
        )

        print(f"Creating chunk features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict2[f"{sub}"].getFeatures(
            paradigms=paradigm[1],
            subject=sub,
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            maxCombinationAmount=2,
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
                paradigms=paradigm[1],
                subject=sub,
                t_min=t_min,
                t_max=t_max,
                sampling_rate=sampling_rate,
                twoDLabels=False,
                maxCombinationAmount=2,
                featureList=featureList,
                verbose=True,
            )
            print("But I dont care about BC not existing here for some reason")

    # if signAll, then create or get globalGoodFeatures mask
    needsFix = True
    if needsFix is not True:
        if signAll:

            # TODO One process/thread per subject
            # Create list of procesess. One per subject
            # Assign each one to check/create/get globalGoodFeatureMask

            # After all joined back. Continue

            for sub in subjects:
                if (
                    fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask() is None
                    and useSepSubjFS is not True
                ):
                    print(
                        f"Anova Mask for sub:{sub}, sign:{globalSignificanceThreshold} was not complete, creating new"
                    )

                    allSubjFList, allSubjFLabels = combineAllSubjects(
                        fClassDict2, subjectLeftOut=sub, onlyTrain=False
                    )
                    goodFeatureList, goodFeatureMaskList = anovaTest(
                        allSubjFList,
                        allSubjFLabels,
                        globalSignificanceThreshold,
                        fClass=fClassDict2[f"{sub}"],
                        onlyUniqueFeatures=onlyUniqueFeatures,
                        uniqueThresh=uniqueThresh,
                    )
                    fClassDict2[f"{sub}"].setGlobalGoodFeaturesMask(goodFeatureMaskList)
                else:
                    goodFeatureMaskList = fClassDict2[
                        f"{sub}"
                    ].getGlobalGoodFeaturesMask()
        else:
            print("Not using Anova Test, not creating Mask")
    return fClassDict2, bClassDict2

    # Instead, just loop through all subjects, doing Anova on each one.
    # Saving it as something not used in testing
    # then for each subject, take all of the other subjects saved ones and add them together.
    # Tomorrow. Stop work now!


def main():

    testSize = 10  # Nr of seed iterations until stopping
    seedStart = 39  # Arbitrary, could be randomized as well.

    # TODO
    # Here, wrap all in another for loop that tests:
    # signAll, SignSolo, thresholds, paradigms, and saves them all in separateFolders
    # Fix soloSignThreshold ( Make sure it is correct)
    # Fix a timer for each loop of this for loop, as well as for each subject
    # and seed in the other two for loops. Save all times in a separate folder

    # Loading parameters, what part of the trials to load and test
    t_min = 1.8
    t_max = 3
    sampling_rate = 256

    # Parameters for ANOVA test and ANOVA Feature Mask
    # Does ANOVA on all subjects except the one tested and uses as mask
    signAll = True
    # 0.1 seems best, 0.05 a little faster
    # if useSepSubFS then this is also used for them
    globalSignificanceThreshold = 0.05
    useSepSubjFS = False  # Does not seem to help at all.
    # Not noticably. And seems slower
    if useSepSubjFS:
        # Harder limit when solo sep Subj, otherwise. They are too big!
        globalSignificanceThreshold = 0.05

    # Does ANOVA on training set of each subject by itself and uses as mask
    signSolo = False
    soloSignificanceThreshold = 0.005  # Not really used anymore!
    useAda = False  # For 1 feauture combo amount actually hurts.

    onlyUniqueFeatures = True
    uniqueThresh = 0.8

    # Tolerance for SVM SVC
    tolerance = 0.001  # Untested

    # Name for this test, what it is saved as
    validationRepetition = True
    repetitionName = "udv-2feat"  # "udrliplotnoAda1hyperparams"
    repetitionValue = f"{28}{repetitionName}"

    # Best feature Combo allow in function only needs to done once! Then which combos that are okay
    # Can be saved. Like index of them.
    useBestFeaturesTest = True
    bestFeaturesSaveFile = "top1udv.npy"
    bestFeatures = np.load(bestFeaturesSaveFile, allow_pickle=True)

    # How many features that are maximally combined and tested together
    maxCombinationAmount = 2

    # All the subjects that are tested, and used to create ANOVA Mask
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,

    quickTest = True  # For testing best Feature. Works well enough
    # Rare to see increase in more than 1-2% accuracy without it.
    # Runs less hyperparameters, currently only C =  2.5

    # What paradigm to test
    # paradigm = paradigmSetting.upDownInnerSpecial11()
    # paradigm = paradigmSetting.upDownInnerSpecial12()
    # paradigm = paradigmSetting.upDownInnerSpecial14()
    # paradigm = paradigmSetting.upDownInnerSpecialPlot()
    paradigm = paradigmSetting.upDownVisSpecialPlot()
    # paradigm = paradigmSetting.upDownInnerSpecial4()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.upDownVisSpecial()
    # paradigm = paradigmSetting.upDownRightLeftInner()
    # paradigm = paradigmSetting.upDownVisInner()
    # paradigm = paradigmSetting.upDownVisInnersep()
    # paradigm = paradigmSetting.upDownRightLeftInnerSpecial2()
    # paradigm = paradigmSetting.upDownRightLeftInnerSpecial()
    # paradigm = paradigmSetting.upDownRightLeftVis()
    # paradigm = paradigmSetting.rightLeftInner()

    # What features that are created and tested
    featureList = [
        False,  # FFT 1
        True,  # Welch 2
        False,  # Hilbert 3 DataHR seems to not add much if any to FFT and Welch
        False,  # Powerbands 4
        False,  # FFT frequency buckets 5
        False,  # FFT Covariance 6
        False,  # Welch Covariance 7
        False,  # Hilbert Covariance 8 DataHR seems to not add much if any to FFT and Welch
        False,  # Covariance on smoothed Data 9 dataGCV
        False,  # Covariance on smoothed Data2 10 dataGCV2 SKIP
        False,  # Correlate1d # SEEMS BAD 11
        False,  # dataFFTCV-BC 12 Is this one doing BC before or after? Before right. yes
        False,  # dataWCV-BC 13
        True,  # dataHRCV-BC 14 DataHR seems to not add much if any to FFT and Welch
        True,  # fftDataBC 15
        False,  # welchDataBC 16
        False,  # dataHRBC 17 DataHR seems to not add much if any to FFT and Welch
        False,  # gaussianData 18
        True,  # dataGCVBC 19
        True,  # gaussianDataBC 20
        True,  # dataGCV-BC       21      - BC means BC before covariance
        False,  # dataFFTCV2-BC 22 With more channels. Only useful for chunks
        False,  # dataGCV2-BC 23 SKIP
        # With more channels. Only useful for chunks For 3 chunks.
        True,  # 24 Correlate1dBC
        # True,  # FFT BC IFFT 24
        # Takes up 50 GB apparently. So. no.
        # Corr1dBC
        # More to be added
    ]
    # badFeatures = [2, 3, 4, 5, 6, 7, 8, 9, 22, 23]
    # badFeatures = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 22]
    # badFeatures = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 19, 22, 23]
    badFeatures = [4, 5, 6, 7, 8, 9, 10, 21, 22, 23]

    goodFeatures = []

    for ind, fea in enumerate(badFeatures):
        badFeatures[ind] = fea - 1

    for ind, fea in enumerate(goodFeatures):
        goodFeatures[ind] = fea - 1

    print(badFeatures)
    # badFeatures = badFeatures - 1
    chunkFeatures = True
    chunkAmount = 3
    onlyCreateFeatures = False
    useAllFeatures = True
    nrFCOT = 3  # nrOfFeaturesToCreateAtOneTime
    featIndex = 0
    featureListIndex = np.arange(len(featureList))
    if onlyCreateFeatures:

        while True:

            for featureI in featureListIndex:
                featureList[featureI] = False

            if (featIndex * nrFCOT) > len(featureList) - 1:
                break
            if featIndex > len(featureList) - (nrFCOT + 1):
                featIndex = len(featureList) - (nrFCOT + 1)
            print(len(featureList))
            print(featIndex)
            for featureI in featureListIndex[
                featIndex * nrFCOT : (featIndex + 1) * nrFCOT
            ]:
                featureList[featureI] = True

            featureList[3] = False  # 4
            featureList[4] = False  # 5
            featureList[9] = False  # 10
            featureList[21] = False  # 23
            featureList[22] = False  # 22
            if chunkFeatures:
                featureList[20] = False  # 21 Not okay for chunks
                featureList[18] = False  # 19 Not okay for chunks

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
            )

            featIndex = featIndex + 1
            # print(feature)
        # If Chunkfeatures, run create again but for non Chunk features
        if chunkFeatures:
            chunkFeatures = False
            while True:

                for featureI in featureListIndex:
                    featureList[featureI] = False

                if (featIndex * nrFCOT) > len(featureList) - 1:
                    chunkFeatures = True
                    break
                if featIndex > len(featureList) - (nrFCOT + 1):
                    featIndex = len(featureList) - (nrFCOT + 1)
                print(len(featureList))
                print(featIndex)
                for featureI in featureListIndex[
                    featIndex * nrFCOT : (featIndex + 1) * nrFCOT
                ]:
                    featureList[featureI] = True

                featureList[3] = False  # 4
                featureList[4] = False  # 5
                featureList[9] = False  # 10
                featureList[21] = False  # 23
                featureList[22] = False  # 22
                if chunkFeatures:
                    featureList[20] = False  # 21 Not okay for chunks
                    featureList[18] = False  # 19 Not okay for chunks

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
                )

                featIndex = featIndex + 1
                # print(feature)

    if useAllFeatures:
        for featureI in featureListIndex:
            featureList[featureI] = False

        for featureI in featureListIndex:
            if featureI in badFeatures:
                continue
            featureList[featureI] = True

    print(featureList)
    # Creating the features for each subject and putting them in a dict
    fClassDict = dict()
    fmetDict = dict()
    bClassDict = dict()
    for sub in subjects:  #

        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            chunk=False,
            chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=0.8,
            useSepSubjFS=useSepSubjFS,
        )
        fmetDict[f"{sub}"] = svmMet.SvmMets(
            significanceThreshold=soloSignificanceThreshold,
            signAll=signAll,
            signSolo=signSolo,
            verbose=False,
            tol=tolerance,
            quickTest=quickTest,
        )
        print(f"Creating features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict[f"{sub}"].getFeatures(
            paradigms=paradigm[1],
            subject=sub,
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            maxCombinationAmount=maxCombinationAmount,
            featureList=featureList,
            verbose=True,
        )

        # time.sleep(2000)
        print(len(createdFeatureList))
        print(f"Printing features created so far for subject {sub}")
        for createdFeature in createdFeatureList:
            print(createdFeature[1])
        print(f"Corrected Exists = {correctedExists}")

        # correctedExists = False
        if correctedExists is False:

            bClassDict[f"{sub}"] = baseLineCorrection(
                subject=sub,
                sampling_rate=sampling_rate,
                chunk=False,
                chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            )

            bClassDict[f"{sub}"].loadBaselineData()

            bClassDict[f"{sub}"].getBaselineFeatures(
                trialSampleAmount=fClassDict[f"{sub}"].getOrigData().shape[2],
                featureList=featureList,
            )

            fClassDict[f"{sub}"].correctedFeatureList = bClassDict[
                f"{sub}"
            ].baselineCorrect(
                fClassDict[f"{sub}"].getFeatureList(),
                fClassDict[f"{sub}"].getLabelsAux(),
                fClassDict[f"{sub}"].paradigmName,
            )

            print(f"Creating features for subject:{sub} after baseline correction")
            createdFeatureList, labels, correctedExists = fClassDict[
                f"{sub}"
            ].getFeatures(
                paradigms=paradigm[1],
                subject=sub,
                t_min=t_min,
                t_max=t_max,
                sampling_rate=sampling_rate,
                twoDLabels=False,
                maxCombinationAmount=maxCombinationAmount,
                featureList=featureList,
                verbose=True,
            )

    if signAll:
        if useSepSubjFS is not True:
            allSubjFListList = []
            allSubjFLabelsList = []
            # goodFeatureMaskListList = []
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
                print(f"Feature Mask Already exist for all Features for subject {sub}")

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
                )

                goodFeatureMaskListList.append(goodFeatureMaskList)

            compute3 = dask.compute(goodFeatureMaskListList)
            goodFeatureMaskListList = dask.compute(compute3)

    if chunkFeatures:
        fClassDict2, bClassDict2 = createChunkFeatures(
            chunkAmount=chunkAmount,
            signAll=signAll,
            signSolo=signSolo,
            onlyUniqueFeatures=onlyUniqueFeatures,
            globalSignificanceThreshold=globalSignificanceThreshold,
            paradigm=paradigm,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
        )

        for sub in subjects:
            fClassDict[f"{sub}"].extendFeatureList(
                fClassDict2[f"{sub}"].getFeatureList()
            )
            fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
                fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
            )

    testNr = 0
    count = 0

    # A for loop just running all subjects using different seeds for train/data split
    for seed in np.arange(seedStart * testSize, (seedStart + 1) * testSize):
        testNr += 1

        # Setting random order for test/train split for all subjects for this seed

        for sub in subjects:
            fClassDict[f"{sub}"].setOrder(seed)
            fClassDict[f"{sub}"].createMaskedFeatureList()

        # For loop running pipeline on each subject
        for sub in subjects:  #

            print(f"Starting test of subject:{sub} , seed:{seed}")

            # Creating masked feature List using ANOVA/cov Mask

            # Then only create new combos containing that best combos + 1 or 2 more features
            # print(fClassDict[f"{sub}"].getLabels())
            mDataList = mixShuffleSplit(
                fClassDict[f"{sub}"].getMaskedFeatureList(),
                labels=fClassDict[f"{sub}"].getLabels(),
                order=fClassDict[f"{sub}"].getOrder(),
                featureClass=fClassDict[f"{sub}"],
                maxCombinationAmount=maxCombinationAmount,
                bestFeatures=bestFeatures,
                useBestFeaturesTest=useBestFeaturesTest,
            )

            # # Create a new list in Features, called Masked Features.
            # # Which are all features except only the good data left after mask
            # mDataList = mixShuffleSplit(
            #     fClassDict[f"{sub}"].getFeatureList(),
            #     labels=fClassDict[f"{sub}"].getLabels(),
            #     order=fClassDict[f"{sub}"].getOrder(),
            #     featureClass=fClassDict[f"{sub}"],
            #     maxCombinationAmount=maxCombinationAmount,
            # )

            allResultsPerSubject = []
            # For loop of each combination of features
            # Training a SVM using each one and then saving the results
            count = 1

            # for x in range(10000):
            #     h = 4 + x

            allResultsPerSubject = Parallel(n_jobs=-5, verbose=10, batch_size=1)(
                delayed(testLoop)(
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    name,
                    testNr,
                    testSize,
                    count,
                    len(mDataList),
                    useAda,
                    fmetDict,
                    sub,
                )
                for data_train, data_test, labels_train, labels_test, name in mDataList
            )

            # for (
            #     data_train,
            #     data_test,
            #     labels_train,
            #     labels_test,
            #     name,
            #     # gdData,
            # ) in mDataList:

            #     # print(f"\n Running dataset: {name} \n")
            #     print(
            #         f" Test {testNr}/{testSize} - Progress {count}/{len(mDataList)}")
            #     count += 1

            #     # Below here can be switch to NN ? Create method? Or just different testSuite.
            #     # Right now using Adaboost.
            #     # TODO, use joblib parallel to spread this over as many cpu as possible
            #     # Would say 4 or 5 is reasonable.
            #     if useAda:

            #         allResults = fmetDict[f"{sub}"].testSuiteAda(
            #             data_train,
            #             data_test,
            #             labels_train,
            #             labels_test,
            #             name,
            #             # gdData,
            #             kernels=["linear", "sigmoid", "rbf"],  #
            #         )
            #     else:
            #         allResults = fmetDict[f"{sub}"].testSuite(
            #             data_train,
            #             data_test,
            #             labels_train,
            #             labels_test,
            #             name,
            #             # gdData,
            #             kernels=["linear", "sigmoid", "rbf"],  #
            #         )

            #     allResultsPerSubject.append(allResults)

            savearray = np.array([seed, sub, allResultsPerSubject], dtype=object)

            # Saving the results
            from datetime import datetime

            # import os

            now = datetime.now()
            # Month abbreviation, day and year, adding time of save to filename
            now_string = now.strftime("D--%d-%m-%Y-T--%H-%M")

            # A new save directory each day to keep track of results
            foldername = now.strftime("%d-%m")

            if validationRepetition:
                foldername = f"{foldername}-{repetitionValue}"
            saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedResults/{foldername}"
            if os.path.exists(saveDir) is not True:
                os.makedirs(saveDir)

            np.save(
                f"{saveDir}/savedBestSeed-{seed}-Subject-{sub}-Date-{now_string}",
                savearray,
            )


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
):

    # Creating the features for each subject and putting them in a dict
    fClassDict = dict()
    fmetDict = dict()
    bClassDict = dict()
    for sub in subjects:  #

        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            chunk=chunkFeatures,
            chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=0.8,
            useSepSubjFS=useSepSubjFS,
        )
        fmetDict[f"{sub}"] = svmMet.SvmMets(
            significanceThreshold=soloSignificanceThreshold,
            signAll=signAll,
            signSolo=signSolo,
            verbose=False,
            tol=0.001,
            quickTest=True,
        )
        print(f"Creating features for subject:{sub}")
        createdFeatureList, labels, correctedExists = fClassDict[f"{sub}"].getFeatures(
            paradigms=paradigm[1],
            subject=sub,
            t_min=t_min,
            t_max=t_max,
            sampling_rate=sampling_rate,
            twoDLabels=False,
            maxCombinationAmount=maxCombinationAmount,
            featureList=featureList,
            verbose=True,
        )
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
                chunk=chunkFeatures,
                chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            )

            bClassDict[f"{sub}"].loadBaselineData()

            bClassDict[f"{sub}"].getBaselineFeatures(
                trialSampleAmount=fClassDict[f"{sub}"].getOrigData().shape[2],
                featureList=featureList,
            )

            fClassDict[f"{sub}"].correctedFeatureList = bClassDict[
                f"{sub}"
            ].baselineCorrect(
                fClassDict[f"{sub}"].getFeatureList(),
                fClassDict[f"{sub}"].getLabelsAux(),
                fClassDict[f"{sub}"].paradigmName,
            )

            print(f"Creating features for subject:{sub} after baseline correct")
            createdFeatureList, labels, correctedExists = fClassDict[
                f"{sub}"
            ].getFeatures(
                paradigms=paradigm[1],
                subject=sub,
                t_min=t_min,
                t_max=t_max,
                sampling_rate=sampling_rate,
                twoDLabels=False,
                maxCombinationAmount=maxCombinationAmount,
                featureList=featureList,
                verbose=True,
            )

    # useSepSubjFS = True
    if signAll:
        if useSepSubjFS is not True:
            allSubjFListList = []
            allSubjFLabelsList = []
            # goodFeatureMaskListList = []
            subjectsThatNeedFSelect = []
        for sub in subjects:

            if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is None:
                if useSepSubjFS:
                    print("Other Place Used")
                    fSelectUsingSepSubjects(
                        fClassDict,
                        globalSignificanceThreshold,
                        onlyUniqueFeatures,
                        uniqueThresh,
                        paradigm[0],
                        subjects,
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

                print(f"Feature Mask Already exist for all Features for subject {sub}")

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
                )

                goodFeatureMaskListList.append(goodFeatureMaskList)

            compute3 = dask.compute(goodFeatureMaskListList)
            # print(compute3)
            goodFeatureMaskListList = dask.compute(compute3)


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


# print(goodFeatureMaskListList)

# for goodFeatureMaskList in goodFeatureMaskListList:
#     saveAnovaMaskNoClass()

# for rp in range(1):
#     processList = []
#     # for subj, features, labels in zip(subjectsThatNeedFSelect[rp * 3:(rp + 1) * 3],
#     #                                   allSubjFListList[rp *
#     #                                                    3:(rp + 1) * 3],
#     #                                   allSubjFLabelsList[rp * 3:(rp + 1) * 3]):
#     for subj, features, labels in zip(subjectsThatNeedFSelect,
#                                       allSubjFListList,
#                                       allSubjFLabelsList):
#         # procSubj = dp(subj)
#         print("start")
#         procFeatures = features
#         print("hoppla")
#         procLabels = labels
#         # print("CreatingNewFclass")
#         # procFclass = fclass.featureEClass(
#         #     procSubj,
#         #     dp(paradigm[0]),
#         #     globalSignificance=globalSignificanceThreshold,
#         #     chunk=False,
#         #     chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
#         #     onlyUniqueFeatures=onlyUniqueFeatures,
#         #     uniqueThresh=0.8
#         # )

#         # kwargsSubj = {"featureList": procFeatures,
#         #               "labels": procLabels,
#         #               "significanceThreshold": globalSignificanceThreshold,
#         #               "fClass": procFclass,
#         #               "onlyUniqueFeatures": onlyUniqueFeatures,
#         #               "uniqueThresh": uniqueThresh}

#         print(f"Creating process for subj {subj}")
#         # procFclass = "hello"

#         # featureList, labels, significanceThreshold, fClass, onlyUniqueFeatures,
#         # uniqueThresh, paradigmName, subject
#         # p = multiprocessing.Process(target=anovaTest, kwargs=kwargsSubj)
#         p = multiprocessing.Process(target=anovaTest, args=(
#             procFeatures, procLabels, globalSignificanceThreshold,
#             onlyUniqueFeatures, uniqueThresh, paradigm[0], subj))
#         processList.append(p)
#         # print(p, p.is_alive())
#         # p.start()
#         # print(p, p.is_alive())
#         # time.sleep(10)

#     for process in processList:
#         # print(process, process.is_alive())
#         time.sleep(15)
#         process.start()
#         # while True:
#         #     time.sleep(1)
#         #     # print(process)
#         #     if process.is_alive():
#         #         break
#         # print(process, process.is_alive())
#         # Create processList
#         #
#         # iterate through subjects, creating processes running AnovaTest

#     while True:
#         time.sleep(5)
#         print(len(multiprocessing.active_children()))
#         if len(multiprocessing.active_children()) < 1:
#             break

#         # goodFeatureList, goodFeatureMaskList = anovaTest(
#         #     allSubjFList,
#         #     allSubjFLabels,
#         #     globalSignificanceThreshold,
#         #     fClass=fClassDict[f"{sub}"],
#         #     onlyUniqueFeatures=onlyUniqueFeatures,
#         #     uniqueThresh=uniqueThresh
#         # )
#     for process in processList:
#         process.join()

# for subj in subjects:
#     fClassDict[f"{sub}"].setGlobalGoodFeaturesMask(
#         goodFeatureMaskList
#     )  # WHY IS THIS WEIRD SHAPE???

# if chunkFeatures:
#     fClassDict2, bClassDict2 = createChunkFeatures(chunkAmount=chunkAmount, signAll=signAll,
#                                                    signSolo=signSolo, onlyUniqueFeatures=onlyUniqueFeatures,
#                                                    globalSignificanceThreshold=globalSignificanceThreshold,
#                                                    paradigm=paradigm,
#                                                    uniqueThresh=uniqueThresh)

#     for sub in subjects:
#         fClassDict[f"{sub}"].extendFeatureList(
#             fClassDict2[f"{sub}"].getFeatureList()
#         )
#         fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
#             fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask())
