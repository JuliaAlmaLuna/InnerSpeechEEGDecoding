"""
This class runs a pipeline testing SVM classification on data
"""

from copy import deepcopy as dp
import numpy as np
import feature_extraction as fclass
from baselineCorr import baseLineCorrection
import svmMethods as svmMet
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler
import paradigmSetting
import cProfile
import pstats
import io
import time
import multiprocessing
from multiprocessing import Value
import glob
import os
# import dask


def mixShuffleSplit(
    createdFeatureList, labels, order, featureClass, maxCombinationAmount
):

    # Copy labels and features list to avoid changes to originals. Probly not needed
    tempLabels = dp(labels)
    tempFeatureList = dp(createdFeatureList)

    mDataList = featureClass.createListOfDataMixes(
        featureList=tempFeatureList,
        labels=tempLabels,
        order=order,
        maxCombinationAmount=maxCombinationAmount,
    )
    return mDataList


def printProcess(processName, printText):
    with open(f"processOutputs/{processName}Output.txt", "a") as f:
        print(printText, file=f)


# def loadAnovaMaskNoClass(featurename, maskname, onlyUniqueFeatures, subject, paradigmName):
#         name = f"{featurename}{maskname}"
#         if self.onlyUniqueFeatures:
#             name = f"{name}u{self.uniqueThresh}"

#         saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedAnovaMask/sub-{self.subject}-par-{self.paradigmName}"
#         path = glob.glob(saveDir + f"/{name}.npy")
#         if len(path) > 0:
#             savedAnovaMask = np.load(path[0], allow_pickle=True)
#             return savedAnovaMask
#         else:
#             return None


def loadAnovaMaskNoClass(featurename, maskname, uniqueThresh, paradigmName, subject, onlyUniqueFeatures):
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


def saveAnovaMaskNoClass(featurename, maskname, array, uniqueThresh, paradigmName, subject, onlyUniqueFeatures):
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


def anovaTest(featureList, labels, significanceThreshold, onlyUniqueFeatures,
              uniqueThresh, paradigmName, subject, startValue
              ):
    # print(
    #     f"Running anova Test and masking using sign threshold: {significanceThreshold}"
    # )

    # printProcess(f"subj{fClass.subject}",
    #              f"Running anova Test and masking using sign threshold: {significanceThreshold}")

    # time.sleep(180)  # To have time to start every process

    while True:
        time.sleep(2)
        if startValue.value == 1:
            break

    time.sleep(subject * 2)
    # Add a lock to these if it bugs!
    printProcess(f"subj{subject}output",
                 f"Running anova Test and masking using sign threshold: {significanceThreshold}")
    # time.sleep(20 + fClass.subject * 5)

    # I use the sklearn StandarScaler before the ANOVA test since that is what will do
    # later as well for every feature before test.

    # TODO
    # Use mutual info classifier to remove all features that are too similar
    # Remove* from mask that is

    # If the mask for specifc feature/subject/signficance already exists. Load it instead

    scaler = StandardScaler()

    goodFeatureList = featureList  # Copy to avoid leak?
    goodFeatureMaskList = []
    for feature, goodfeature in zip(featureList, goodFeatureList):  # Features

        featureName = feature[1]
        loadedMask = loadAnovaMaskNoClass(featurename=featureName,
                                          maskname=f"sign{significanceThreshold}",
                                          uniqueThresh=uniqueThresh,
                                          paradigmName=paradigmName,
                                          subject=subject,
                                          onlyUniqueFeatures=onlyUniqueFeatures
                                          )

        # fClass.loadAnovaMask(
        #     featurename=featureName, maskname=f"sign{significanceThreshold}"
        # )
        # loadedMask = None

        flatfeature = np.reshape(feature[0], [feature[0].shape[0], -1])
        flatgoodfeature = np.reshape(
            goodfeature[0], [goodfeature[0].shape[0], -1])

        scaler.fit(flatfeature)
        flatfeature = scaler.transform(flatfeature)
        flatgoodfeature = scaler.transform(flatgoodfeature)

        if loadedMask is None:

            # Running the ANOVA Test
            f_statistic, p_values = feature_selection.f_classif(
                flatfeature, labels)

            # Create a mask of features with P values below threshold
            p_values[p_values > significanceThreshold] = 0
            p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2

            goodData = f_statistic * p_values

            # TODO:
            # This part seems to be possibly to heavily multiThread
            if onlyUniqueFeatures:
                # These goodfeatures need to come with a array of original index
                # Then. When a feature is deleted. Make it zero on goodDataMask
                goodfeature = flatfeature[:, np.where(goodData != 0)[0]]
                indexList = np.where(goodData != 0)[0]

                goodfeature = np.swapaxes(goodfeature, 0, 1)

                printProcess(f"subj{subject}output",
                             time.clock())
                corrMat = np.corrcoef(goodfeature)
                printProcess(f"subj{subject}output",
                             time.clock())

                printProcess(f"subj{subject}output",
                             corrMat.shape)

                deleteIndexes = []
                for ind, feat in enumerate(corrMat):
                    if ind in deleteIndexes:
                        continue
                    # SHOULD PROBABLY BE 0.8-0.9, maybe upper limit 0.9, lower limit 0.7
                    # Check what limit would  give 10 percent left, and then use limits
                    deleteIndexes.extend(np.where(feat > uniqueThresh)[0][1:])

                printProcess(f"subj{subject}output",
                             f"{np.count_nonzero(goodData)} good Features \
                                 before covRemoval:{uniqueThresh}in {feature[1]}")
                goodData[indexList[deleteIndexes]] = 0
                printProcess(f"subj{subject}output",
                             f"{np.count_nonzero(goodData)} good Features \
                                 after covRemoval:{uniqueThresh} in {feature[1]}")

        else:
            printProcess(f"subj{subject}output",
                         f"Loaded mask {featureName}")
            goodData = loadedMask

        printProcess(f"subj{subject}output",
                     f"{np.count_nonzero(goodData)} good Features in {feature[1]}")
        goodFeatureMaskList.append(goodData)
        feature[0] = None
        goodfeature[0] = None
        # Here, I can delete feature[0] from list to save ram space!

    for feature, mask in zip(featureList, goodFeatureMaskList):

        saveAnovaMaskNoClass(
            featurename=feature[1],
            maskname=f"sign{significanceThreshold}",
            array=mask,
            uniqueThresh=uniqueThresh,
            paradigmName=paradigmName,
            subject=subject,
            onlyUniqueFeatures=onlyUniqueFeatures
        )

    # fClass.setGlobalGoodFeaturesMask(goodFeatureMaskList)

    return goodFeatureList, goodFeatureMaskList


def createChunkFeatures(chunkAmount, signAll,
                        signSolo, onlyUniqueFeatures,
                        globalSignificanceThreshold,
                        uniqueThresh,
                        paradigm):
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
        True,  # dataFFTCV-BC 12 Is this one doing BC before or after? Before right. yes
        True,  # dataWCV-BC 13
        True,  # dataHRCV-BC 14 DataHR seems to not add much if any to FFT and Welch
        True,  # fftDataBC 15
        True,  # welchDataBC 16
        True,  # dataHRBC 17 DataHR seems to not add much if any to FFT and Welch
        False,  # gaussianData 18
        True,  # dataGCVBC 19
        True,  # gaussianDataBC 20
        True,  # dataGCV-BC       21      - BC means BC before covariance
        False,  # dataFFTCV2-BC 22 With more channels. Only useful for chunks
        False,  # dataGCV2-BC 23 With more channels. Only useful for chunks
        # More to be added
    ]

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
            uniqueThresh=uniqueThresh
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

    # if signAll, then create or get globalGoodFeatures mask

    if signAll:

        # TODO One process/thread per subject
        # Create list of procesess. One per subject
        # Assign each one to check/create/get globalGoodFeatureMask

        # After all joined back. Continue

        for sub in subjects:
            if fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask() is None:
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
                    uniqueThresh=uniqueThresh
                )
                fClassDict2[f"{sub}"].setGlobalGoodFeaturesMask(
                    goodFeatureMaskList)
            else:
                goodFeatureMaskList = fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask(
                )
    else:
        print("Not using Anova Test, not creating Mask")
    return fClassDict2, bClassDict2


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
    globalSignificanceThreshold = 0.05
    # Does ANOVA on training set of each subject by itself and uses as mask
    signSolo = False
    soloSignificanceThreshold = 0.005

    onlyUniqueFeatures = True
    uniqueThresh = 0.8

    # Tolerance for SVM SVC
    tolerance = 0.001  # Untested

    # Name for this test, what it is saved as
    validationRepetition = True
    repetitionName = "udrlBC4CVTest"
    repetitionValue = f"{6}{repetitionName}"

    # How many features that are maximally combined and tested together
    maxCombinationAmount = 3

    # All the subjects that are tested, and used to create ANOVA Mask
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,

    quickTest = True  # Runs less hyperparameters
    onlyCreateFeatures = True
    # What paradigm to test

    # paradigm = paradigmSetting.upDownInner()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.upDownVisSpecial()
    # paradigm = paradigmSetting.upDownRightLeftInner()
    # paradigm = paradigmSetting.upDownRightLeftInnerSpecial()
    # paradigm = paradigmSetting.upDownRightLeftVis()
    # paradigm = paradigmSetting.rightLeftInner()

    chunkFeatures = False
    chunkAmount = 3
    # What features that are created and tested
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
        False,  # Covariance on smoothed Data2 10 "dataGCV2"
        False,  # Correlate1d # SEEMS BAD 11
        False,  # dataFFTCV-BC 12 Is this one doing BC before or after? Before right. yes
        False,  # dataWCV-BC 13
        True,  # dataHRCV-BC 14 DataHR seems to not add much if any to FFT and Welch
        False,  # fftDataBC 15
        False,  # welchDataBC 16
        False,  # dataHRBC 17 DataHR seems to not add much if any to FFT and Welch
        False,  # gaussianData 18
        False,  # dataGCVBC 19
        True,  # gaussianDataBC 20
        True,  # dataGCV-BC       21      - BC means BC before covariance
        False,  # dataFFTCV2-BC 22 With more channels. Only useful for chunks
        False,  # dataGCV2-BC 23 With more channels. Only useful for chunks
        # More to be added
    ]

    featIndex = 0
    featureListIndex = np.arange(len(featureList))
    if onlyCreateFeatures:

        while True:

            for featureI in featureListIndex:
                featureList[featureI] = False

            if featIndex * 3 > len(featureList) - 1:
                break
            if featIndex > len(featureList) - 4:
                featIndex = len(featureList) - 4
            print(len(featureList))
            print(featIndex)
            for featureI in featureListIndex[featIndex * 3:(featIndex + 1) * 3]:
                featureList[featureI] = True

            print(featureList)
            onlyCreateFeaturesFunction(subjects,
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
                                       featureList)

            featIndex = featIndex + 1
            # print(feature)

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
            uniqueThresh=0.8
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
        print(len(createdFeatureList))
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

            print(f"Creating features for subject:{sub}")
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

            # del (createdFeatureList)
            # del ()

    # if signAll, then create or get globalGoodFeatures mask

    # TODO One process/thread per subject
    # Create list of procesess. One per subject
    # Assign each one to check/create/get globalGoodFeatureMask

    # After all joined back. Continue
    # Use Sentinel
    # Or waitfor Multiple Objects OS Handle

    # TODO, DASK THIS!
    if signAll:
        allSubjFListList = []
        allSubjFLabelsList = []
        # goodFeatureMaskListList = []
        subjectsThatNeedFSelect = []
        for sub in subjects:

            if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is None:

                allSubjFList, allSubjFLabels = combineAllSubjects(
                    fClassDict, subjectLeftOut=sub, onlyTrain=False
                )

                # add allSubjFlist and Labels to list
                allSubjFLabelsList.append(allSubjFLabels)
                allSubjFListList.append(allSubjFList)
                subjectsThatNeedFSelect.append(sub)

            else:
                # goodFeatureMaskList = fClassDict[f"{sub}"].getGlobalGoodFeaturesMask(
                # )
                print(
                    f"Feature Mask Already exist for all Features for subject {sub}")

        startValue = Value("i", 0)
        for rp in range(1):
            processList = []
            # for subj, features, labels in zip(subjectsThatNeedFSelect[rp * 3:(rp + 1) * 3],
            #                                   allSubjFListList[rp *
            #                                                    3:(rp + 1) * 3],
            #                                   allSubjFLabelsList[rp * 3:(rp + 1) * 3]):
            for subj, features, labels in zip(subjectsThatNeedFSelect,
                                              allSubjFListList,
                                              allSubjFLabelsList):
                # procSubj = dp(subj)
                print("start")
                procFeatures = features
                print("hoppla")
                procLabels = labels
                # print("CreatingNewFclass")
                # procFclass = fclass.featureEClass(
                #     procSubj,
                #     dp(paradigm[0]),
                #     globalSignificance=globalSignificanceThreshold,
                #     chunk=False,
                #     chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
                #     onlyUniqueFeatures=onlyUniqueFeatures,
                #     uniqueThresh=0.8
                # )

                # kwargsSubj = {"featureList": procFeatures,
                #               "labels": procLabels,
                #               "significanceThreshold": globalSignificanceThreshold,
                #               "fClass": procFclass,
                #               "onlyUniqueFeatures": onlyUniqueFeatures,
                #               "uniqueThresh": uniqueThresh}

                print(f"Creating process for subj {subj}")
                # procFclass = "hello"

                # featureList, labels, significanceThreshold, fClass, onlyUniqueFeatures,
                # uniqueThresh, paradigmName, subject
                # p = multiprocessing.Process(target=anovaTest, kwargs=kwargsSubj)
                p = multiprocessing.Process(target=anovaTest, args=(
                    procFeatures,
                    procLabels,
                    globalSignificanceThreshold,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    paradigm[0],
                    subj,
                    startValue))

                processList.append(p)
                # print(p, p.is_alive())
                # p.start()
                # print(p, p.is_alive())
                # time.sleep(10)

            for process in processList:
                # print(process, process.is_alive())
                time.sleep(5)
                process.start()
                # while True:
                #     time.sleep(1)
                #     # print(process)
                #     if process.is_alive():
                #         break
                # print(process, process.is_alive())
                # Create processList
                #
                # iterate through subjects, creating processes running AnovaTest

            while True:
                startValue.value = 1
                time.sleep(5)
                print(len(multiprocessing.active_children()))
                if len(multiprocessing.active_children()) < 1:
                    break

                # goodFeatureList, goodFeatureMaskList = anovaTest(
                #     allSubjFList,
                #     allSubjFLabels,
                #     globalSignificanceThreshold,
                #     fClass=fClassDict[f"{sub}"],
                #     onlyUniqueFeatures=onlyUniqueFeatures,
                #     uniqueThresh=uniqueThresh
                # )
            for process in processList:
                process.join()

        # for subj in subjects:
        #     fClassDict[f"{sub}"].setGlobalGoodFeaturesMask(
        #         goodFeatureMaskList
        #     )  # WHY IS THIS WEIRD SHAPE???

    if chunkFeatures:
        fClassDict2, bClassDict2 = createChunkFeatures(chunkAmount=chunkAmount, signAll=signAll,
                                                       signSolo=signSolo, onlyUniqueFeatures=onlyUniqueFeatures,
                                                       globalSignificanceThreshold=globalSignificanceThreshold,
                                                       paradigm=paradigm,
                                                       uniqueThresh=uniqueThresh)

        for sub in subjects:
            fClassDict[f"{sub}"].extendFeatureList(
                fClassDict2[f"{sub}"].getFeatureList()
            )
            fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
                fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
            )

    testNr = 0

    # A for loop just running all subjects using different seeds for train/data split
    for seed in np.arange(seedStart * testSize, (seedStart + 1) * testSize):
        testNr += 1

        # Setting random order for test/train split for all subjects for this seed

        # TODO Use DASK for this!
        for sub in subjects:
            fClassDict[f"{sub}"].setOrder(seed)
            fClassDict[f"{sub}"].createMaskedFeatureList()

        # For loop running pipeline on each subject
        for sub in subjects:  #

            print(f"Starting test of subject:{sub} , seed:{seed}")
            # TODO: Only send in good Features, since they are so much smaller!

            # Creating masked feature List using ANOVA/cov Mask

            mDataList = mixShuffleSplit(
                fClassDict[f"{sub}"].getMaskedFeatureList(),
                labels=fClassDict[f"{sub}"].getLabels(),
                order=fClassDict[f"{sub}"].getOrder(),
                featureClass=fClassDict[f"{sub}"],
                maxCombinationAmount=maxCombinationAmount,
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

            for (
                data_train,
                data_test,
                labels_train,
                labels_test,
                name,
                # gdData,
            ) in mDataList:

                # print(f"\n Running dataset: {name} \n")
                print(
                    f" Test {testNr}/{testSize} - Progress {count}/{len(mDataList)}")
                count += 1

                # Below here can be switch to NN ? Create method? Or just different testSuite

                allResults = fmetDict[f"{sub}"].testSuiteAda(
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    name,
                    # gdData,
                    kernels=["linear", "sigmoid", "rbf"],  #
                )

                allResultsPerSubject.append(allResults)

            savearray = np.array(
                [seed, sub, allResultsPerSubject], dtype=object)

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


def onlyCreateFeaturesFunction(subjects,
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
                               featureList
                               ):

    # Creating the features for each subject and putting them in a dict
    fClassDict = dict()
    bClassDict = dict()
    for sub in subjects:  #

        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub,
            paradigm[0],
            globalSignificance=globalSignificanceThreshold,
            chunk=False,
            chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
            onlyUniqueFeatures=onlyUniqueFeatures,
            uniqueThresh=0.8
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

            print(f"Creating features for subject:{sub}")
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

            # del (createdFeatureList)
            # del ()

    # if signAll, then create or get globalGoodFeatures mask

    # TODO One process/thread per subject
    # Create list of procesess. One per subject
    # Assign each one to check/create/get globalGoodFeatureMask

    # After all joined back. Continue
    # Use Sentinel
    # Or waitfor Multiple Objects OS Handle

    # TODO, DASK THIS!
    if signAll:
        allSubjFListList = []
        allSubjFLabelsList = []
        # goodFeatureMaskListList = []
        subjectsThatNeedFSelect = []
        for sub in subjects:

            if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is None:

                allSubjFList, allSubjFLabels = combineAllSubjects(
                    fClassDict, subjectLeftOut=sub, onlyTrain=False
                )

                # add allSubjFlist and Labels to list
                allSubjFLabelsList.append(allSubjFLabels)
                allSubjFListList.append(allSubjFList)
                subjectsThatNeedFSelect.append(sub)

            else:
                # goodFeatureMaskList = fClassDict[f"{sub}"].getGlobalGoodFeaturesMask(
                # )
                print(
                    f"Feature Mask Already exist for all Features for subject {sub}")

        startValue = Value("i", 0)
        for rp in range(1):
            processList = []
            # for subj, features, labels in zip(subjectsThatNeedFSelect[rp * 3:(rp + 1) * 3],
            #                                   allSubjFListList[rp *
            #                                                    3:(rp + 1) * 3],
            #                                   allSubjFLabelsList[rp * 3:(rp + 1) * 3]):
            for subj, features, labels in zip(subjectsThatNeedFSelect,
                                              allSubjFListList,
                                              allSubjFLabelsList):
                # procSubj = dp(subj)
                print("start")
                procFeatures = features
                print("hoppla")
                procLabels = labels
                # print("CreatingNewFclass")
                # procFclass = fclass.featureEClass(
                #     procSubj,
                #     dp(paradigm[0]),
                #     globalSignificance=globalSignificanceThreshold,
                #     chunk=False,
                #     chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
                #     onlyUniqueFeatures=onlyUniqueFeatures,
                #     uniqueThresh=0.8
                # )

                # kwargsSubj = {"featureList": procFeatures,
                #               "labels": procLabels,
                #               "significanceThreshold": globalSignificanceThreshold,
                #               "fClass": procFclass,
                #               "onlyUniqueFeatures": onlyUniqueFeatures,
                #               "uniqueThresh": uniqueThresh}

                print(f"Creating process for subj {subj}")
                # procFclass = "hello"

                # featureList, labels, significanceThreshold, fClass, onlyUniqueFeatures,
                # uniqueThresh, paradigmName, subject
                # p = multiprocessing.Process(target=anovaTest, kwargs=kwargsSubj)
                p = multiprocessing.Process(target=anovaTest, args=(
                    procFeatures,
                    procLabels,
                    globalSignificanceThreshold,
                    onlyUniqueFeatures,
                    uniqueThresh,
                    paradigm[0],
                    subj,
                    startValue))

                processList.append(p)
                # print(p, p.is_alive())
                # p.start()
                # print(p, p.is_alive())
                # time.sleep(10)

            for process in processList:
                # print(process, process.is_alive())
                time.sleep(5)
                process.start()
                # while True:
                #     time.sleep(1)
                #     # print(process)
                #     if process.is_alive():
                #         break
                # print(process, process.is_alive())
                # Create processList
                #
                # iterate through subjects, creating processes running AnovaTest

            while True:
                startValue.value = 1
                time.sleep(5)
                print(len(multiprocessing.active_children()))
                if len(multiprocessing.active_children()) < 1:
                    break

                # goodFeatureList, goodFeatureMaskList = anovaTest(
                #     allSubjFList,
                #     allSubjFLabels,
                #     globalSignificanceThreshold,
                #     fClass=fClassDict[f"{sub}"],
                #     onlyUniqueFeatures=onlyUniqueFeatures,
                #     uniqueThresh=uniqueThresh
                # )
            for process in processList:
                process.join()

        # for subj in subjects:
        #     fClassDict[f"{sub}"].setGlobalGoodFeaturesMask(
        #         goodFeatureMaskList
        #     )  # WHY IS THIS WEIRD SHAPE???

    if chunkFeatures:
        fClassDict2, bClassDict2 = createChunkFeatures(chunkAmount=chunkAmount, signAll=signAll,
                                                       signSolo=signSolo, onlyUniqueFeatures=onlyUniqueFeatures,
                                                       globalSignificanceThreshold=globalSignificanceThreshold,
                                                       paradigm=paradigm,
                                                       uniqueThresh=uniqueThresh)

        for sub in subjects:
            fClassDict[f"{sub}"].extendFeatureList(
                fClassDict2[f"{sub}"].getFeatureList()
            )
            fClassDict[f"{sub}"].extendGlobalGoodFeaturesMaskList(
                fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask()
            )


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
