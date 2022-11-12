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
    useAda,
    userndF,
    useMLP,
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
    elif userndF:
        allResults = fmetDict[f"{sub}"].testSuiteForest(
            data_train,
            data_test,
            labels_train,
            labels_test,
            name,
            # gdData,
            kernels=["linear", "sigmoid", "rbf"],  #
        )
    elif useMLP:
        allResults = fmetDict[f"{sub}"].testSuiteMLP(
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
        # TODO: Only part after here needs to be dasked. It needs goodfeature, indexList and goodData,
        # and returns new goodData. It can
        # be used in this way: Send in featurename as well. Return goodData and featureName.
        # Use this to save it correctly afterwards
        # So return will be two things. List of names/subject, and list of goodData arrays.
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
    # This one cant be here then
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

            flatfeature = np.reshape(feature[0], [feature[0].shape[0], -1])

            scaler.fit(flatfeature)
            flatfeature = scaler.transform(flatfeature)

            # Running the ANOVA Test
            f_statistic, p_values = feature_selection.f_classif(
                flatfeature, labels)

            # Create a mask of features with P values below threshold
            p_values[p_values > significanceThreshold] = 0
            p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2

            goodData = f_statistic * p_values
            f_statistic = None
            p_values = None
            if "CV" in featureName:
                goodData = np.reshape(goodData, [feature[0].shape[1], -1])

                goodData = np.triu(
                    goodData, k=0
                )  # Keep only half of CV matrices. Rest is the same. No need to cull twice.
                goodData = np.reshape(goodData, [-1])
                # Should apply to only the last two axises.
            feature[0] = None
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
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,
    badFeatures = [4, 5, 6, 7, 8, 9, 10, 19, 21, 22,
                   23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

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
        False,  # 25 invFFT
        False,  # 26 realfftData
        False,  # 27 imfftDataBCBC
        False,  # 28 realfftDataBCBC
        False,  # 29 inverseFFT-BC
        False,  # 30 inverseFFTCV-BC
        # More to be added
    ]

    for ind, fea in enumerate(badFeatures):
        badFeatures[ind] = fea - 1

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
    userndF = False  # For 1 feauture combo amount actually hurts.
    useMLP = True  # For 1 feauture combo amount actually hurts

    onlyUniqueFeatures = True
    uniqueThresh = 0.8

    # Tolerance for SVM SVC
    tolerance = 0.001  # Untested

    # Name for this test, what it is saved as
    validationRepetition = True
    repetitionName = "rndMlP3"  # "udrliplotnoAda1hyperparams"
    repetitionValue = f"{56}{repetitionName}"
    maxCombinationAmount = 3
    chunkFeatures = True
    chunkAmount = 3

    # Run test using all features except "BadFeatures"
    useAllFeatures = True
    badFeatures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 21,
                   22, 23, 26, 27]  # 26, 27, 28, 29, 30, 31, 32
    # badFeatures = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    #                16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    # Using numbering in list below

    # Settings when using running a loop to create all features.
    onlyCreateFeatures = False
    nrFCOT = 3  # nrOfFeaturesToCreateAtOneTime
    featIndex = 6  # Multiplied by nrFCOT, First features to start creating

    # Best feature Combo allow in function only needs to done once! Then which combos that are okay
    # Can be saved. Like index of them.
    useBestFeaturesTest = True
    bestFeaturesSaveFile = "top2mlpudrlv.npy"
    bestFeatures = np.load(
        f"topFeatures/{bestFeaturesSaveFile}", allow_pickle=True)
    if useBestFeaturesTest:
        print(bestFeatures)
        print(bestFeatures.shape)
    # How many features that are maximally combined and tested together

    # All the subjects that are tested, and used to create ANOVA Mask
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,

    quickTest = True  # For testing best Feature. Works well enough
    # Rare to see increase in more than 1-2% accuracy without it.
    # Runs less hyperparameters, currently only C =  2.5

    # What paradigm to test
    # paradigm = paradigmSetting.upDownInnerSpecialPlot()
    # paradigm = paradigmSetting.upDownInnerSpecialTest2()
    # paradigm = paradigmSetting.upDownVisSpecialPlot()
    paradigm = paradigmSetting.upDownRightLeftVisSpecialPlot()
    # paradigm = paradigmSetting.rightLeftInnerSpecialPlot()
    # paradigm = paradigmSetting.rightLeftVisSpecialPlot()

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
        True,  # dataGCV-BC       21      -BC means BC before covariance
        False,  # dataFFTCV2-BC 22 With more channels. Only useful for chunks
        False,  # dataGCV2-BC 23 SKIP
        # With more channels. Only useful for chunks For 3 chunks.
        True,  # 24 Correlate1dBC005s
        False,  # 25 inverseFFT-BC
        False,  # 26 corr01s
        False,  # 27 corr02s
        False,  # 28 iFFTdataCorr1d01s-BC
        False,  # 29 iFFTdataCorr1d02s-BC
        False,  # 30 iFFTdataCorr1d005s-BC
        False,  # 31 dataCorr1d01sBC
        False,  # 32 dataCorr1d02sBC
        False,  # 33 dataCorr2ax1d
        False,  # 34 iFFTdataCorr2ax1d005s-BC
        False,  # 35 dataCorr2ax1dBC

        # True,  # FFT BC IFFT 24
        # More to be added
    ]

    # Features that aren´t tested when using useAllFeatures

    goodFeatures = []  # To be implemented. Opposite way with only good features tested

    # For loops to fit with numbering from 0 to len()-1 instead of 1 to len()
    for ind, fea in enumerate(badFeatures):
        badFeatures[ind] = fea - 1

    for ind, fea in enumerate(goodFeatures):
        goodFeatures[ind] = fea - 1

    print(badFeatures)

    featureListIndex = np.arange(len(featureList))
    if onlyCreateFeatures:

        while True:

            for featureI in featureListIndex:
                featureList[featureI] = False

            if (featIndex * nrFCOT) > len(featureList) - 1:
                break
            if featIndex > len(featureList) - (nrFCOT + 1):
                featIndex = len(featureList) - (nrFCOT + 1)

            for featureI in featureListIndex[
                featIndex * nrFCOT: (featIndex + 1) * nrFCOT
            ]:
                featureList[featureI] = True

            featureList[3] = False  # 4
            featureList[4] = False  # 5
            featureList[5] = False  # 6
            featureList[6] = False  # 7
            featureList[7] = False  # 8
            featureList[9] = False  # 10
            featureList[21] = False  # 22
            featureList[22] = False  # 23
            # featureList[24] = False  # 25
            # featureList[25] = False  # 26
            # featureList[26] = False  # 27
            # featureList[27] = False  # 28
            # featureList[28] = False  # 29
            # featureList[29] = False  # 30
            # featureList[30] = False  # 31
            # featureList[31] = False  # 32
            if chunkFeatures:
                # featureList[21] = True  # 22
                # featureList[20] = False  # 21 Not okay for chunks
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

        # If Chunkfeatures, run create again but for non Chunk features
        if chunkFeatures:
            featIndex = 0
            chunkFeatures = False  # Turn it off to create None chunk features as well
            while True:

                for featureI in featureListIndex:
                    featureList[featureI] = False

                if (featIndex * nrFCOT) > len(featureList) - 1:
                    chunkFeatures = True  # Then turn it back on again after
                    break

                if featIndex > len(featureList) - (nrFCOT + 1):
                    featIndex = len(featureList) - (nrFCOT + 1)

                for featureI in featureListIndex[
                    featIndex * nrFCOT: (featIndex + 1) * nrFCOT
                ]:
                    featureList[featureI] = True

                    featureList[3] = False  # 4
                    featureList[4] = False  # 5
                    featureList[5] = False  # 6
                    featureList[6] = False  # 7
                    featureList[7] = False  # 8
                    featureList[9] = False  # 10
                    featureList[21] = False  # 22
                    featureList[22] = False  # 23
                    # featureList[24] = False  # 25
                    featureList[25] = False  # 26
                    featureList[26] = False  # 27
                    featureList[27] = False  # 28
                    featureList[28] = False  # 29
                    featureList[29] = False  # 30
                    if chunkFeatures:
                        # featureList[21] = True  # 22
                        # featureList[20] = False  # 21 Not okay for chunks
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

    print(f"FeatureList so far: {featureList}")
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

            print(
                f"Creating features for subject:{sub} after baseline correction")
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
                )

                goodFeatureMaskListList.append(goodFeatureMaskList)

            compute3 = dask.compute(goodFeatureMaskListList)
            goodFeatureMaskListList = dask.compute(compute3)

    if chunkFeatures:
        fClassDict2 = createChunkFeatures(
            chunkAmount=chunkAmount,
            onlyUniqueFeatures=onlyUniqueFeatures,
            globalSignificanceThreshold=globalSignificanceThreshold,
            paradigm=paradigm,
            uniqueThresh=uniqueThresh,
            useSepSubjFS=useSepSubjFS,
        )[0]

        for sub in subjects:
            fClassDict[f"{sub}"].extendFeatureList(
                fClassDict2[f"{sub}"].getFeatureList()
            )
            fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
                fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
            )

    # A for loop just running all subjects using different seeds for train/data split
    for seed in np.arange(seedStart * testSize, (seedStart + 1) * testSize):

        # Setting random order for test/train split for all subjects for this seed

        for sub in subjects:
            fClassDict[f"{sub}"].setOrder(seed)
            fClassDict[f"{sub}"].createMaskedFeatureList()

        # For loop running pipeline on each subject
        for sub in subjects:  #

            print(f"Starting test of subject:{sub} , seed:{seed}")

            # Creating masked feature List using ANOVA/cov Mask

            # Then only create new combos containing that best combos + 1 or 2 more features
            if signAll:

                mDataList = mixShuffleSplit(
                    fClassDict[f"{sub}"].getMaskedFeatureList(),
                    labels=fClassDict[f"{sub}"].getLabels(),
                    order=fClassDict[f"{sub}"].getOrder(),
                    featureClass=fClassDict[f"{sub}"],
                    maxCombinationAmount=maxCombinationAmount,
                    bestFeatures=bestFeatures,
                    useBestFeaturesTest=useBestFeaturesTest,
                )
            else:
                mDataList = mixShuffleSplit(
                    fClassDict[f"{sub}"].getFeatureListFlat(),
                    labels=fClassDict[f"{sub}"].getLabels(),
                    order=fClassDict[f"{sub}"].getOrder(),
                    featureClass=fClassDict[f"{sub}"],
                    maxCombinationAmount=maxCombinationAmount,
                    bestFeatures=bestFeatures,
                    useBestFeaturesTest=useBestFeaturesTest,
                )

            allResultsPerSubject = []
            # For loop of each combination of features
            # Training a SVM using each one and then saving the result

            allResultsPerSubject = Parallel(n_jobs=-5, verbose=10, batch_size=1)(
                delayed(testLoop)(
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    name,
                    useAda,
                    userndF,
                    useMLP,
                    fmetDict,
                    sub,
                )
                for data_train, data_test, labels_train, labels_test, name in mDataList
            )

            savearray = np.array(
                [seed, sub, allResultsPerSubject], dtype=object)

            # Saving the results
            from datetime import datetime

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

            print(
                f"Creating features for subject:{sub} after baseline correct")
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
            subjectsThatNeedFSelect = []
        for sub in subjects[:4]:

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
                )

                goodFeatureMaskListList.append(goodFeatureMaskList)

            compute3 = dask.compute(goodFeatureMaskListList)
            # print(compute3)
            goodFeatureMaskListList = dask.compute(compute3)

        for sub in subjects[4:]:

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
