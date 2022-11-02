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


def mixShuffleSplit(
    createdFeatureList, labels, order, featureClass, maxCombinationAmount
):

    # np.random.shuffle(order)

    tempLabels = dp(labels)  # To avoid any changes to original labels
    tempFeatureList = dp(
        createdFeatureList
    )  # To avoid any changes to original feature list
    mDataList = featureClass.createListOfDataMixes(
        featureList=tempFeatureList,
        labels=tempLabels,
        order=order,
        maxCombinationAmount=maxCombinationAmount,
    )
    return mDataList


def anovaTest(featureList, labels, significanceThreshold):

    scaler = StandardScaler()

    goodFeatureList = dp(featureList)
    goodFeatureMaskList = []
    # Anova Test and keep only features with p value less than 0.05

    for feature, goodfeature in zip(featureList, goodFeatureList):  # Features

        # for feature, goodfeature in zip(features, goodfeatures): # FeatureList
        # normalShape = feature.shape
        flatfeature = np.reshape(feature[0], [feature[0].shape[0], -1])
        flatgoodfeature = np.reshape(
            goodfeature[0], [goodfeature[0].shape[0], -1])
        print(flatfeature.shape)
        print(goodfeature[0].shape)
        print(flatgoodfeature.shape)

        scaler.fit(flatfeature)
        flatfeature = scaler.transform(flatfeature)
        flatgoodfeature = scaler.transform(flatgoodfeature)
        print("julia")
        print(flatfeature.shape)
        print(labels.shape)
        print(feature[1])
        f_statistic, p_values = feature_selection.f_classif(
            flatfeature, labels)
        print("julia2")
        p_values[
            p_values > significanceThreshold
        ] = 0  # Use sklearn selectpercentile instead?
        print("julia3")
        p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2
        print("julia4")
        goodData = f_statistic * p_values
        print("julia5")
        goodFeatureMaskList.append(goodData)
        print("julia6")
        goodfeature = flatfeature[:, np.where(goodData != 0)[0]]

        print(goodfeature.shape)
        print(np.count_nonzero(goodData))
        print(goodData.shape)
    return goodFeatureList, goodFeatureMaskList


def createChunkFeatures(chunkAmount):
    # Fix it so chunkFeatures are not touched by not chunk functions . And checks alone

    # Loading parameters, what part of the trials to load and test
    t_min = 1.8
    t_max = 3
    sampling_rate = 256

    # Parameters for ANOVA test and ANOVA Feature Mask
    signAll = True
    # globalSignificanceThreshold = 0.1  # 0.1 seems best, 0.05 a little faster
    globalSignificanceThreshold = 0.005
    # All the subjects that are tested, and used to create ANOVA Mask
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,

    # What paradigm to test

    # paradigm = paradigmSetting.upDownInner()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.upDownVisSpecial()
    paradigm = paradigmSetting.upDownRightLeftInnerSpecial()
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
        False,  # Covariance on smoothed Data 9
        False,  # Covariance on smoothed Data2 10
        False,  # Correlate1d # SEEMS BAD 11
        False,  # dataFFTCVBC 12                 # THIS ONE WEIRD WHEN CHUNK = 6
        True,  # dataWCVBC 13
        False,  # dataHRCVBC 14 DataHR seems to not add much if any to FFT and Welch
        False,  # fftDataBC 15 # THIS ONE WEIRD WHEN CHUNK = 6
        True,  # welchDataBC 16
        False,  # dataHRBC 17 DataHR seems to not add much if any to FFT and Welch
        False,  # dataCVBC 18
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
            chunkAmount=chunkAmount,  # Doesn't matter if chunk = False
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
        print(len(createdFeatureList))
        for createdFeature in createdFeatureList:
            print(createdFeature[1])
        print(f"Corrected Exists = {correctedExists}")
        # correctedExists = False
        if correctedExists is False:

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

            print(f"Creating features for subject:{sub}")
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
        for sub in subjects:
            if fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask() is None:
                allSubjFList, allSubjFLabels = combineAllSubjects(
                    fClassDict2, subjectLeftOut=sub, onlyTrain=False
                )
                goodFeatureList, goodFeatureMaskList = anovaTest(
                    allSubjFList, allSubjFLabels, globalSignificanceThreshold
                )
                fClassDict2[f"{sub}"].setGlobalGoodFeaturesMask(
                    goodFeatureMaskList
                )  # WHY IS THIS WEIRD SHAPE???
            else:
                goodFeatureMaskList = fClassDict2[f"{sub}"].getGlobalGoodFeaturesMask(
                )

    return fClassDict2, bClassDict2


def combineAllSubjects(fclassDict, subjectLeftOut=None, onlyTrain=False):

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

    # print(type(allSubjFList))
    # print(type(allSubjFList[0]))
    print(len(allSubjFList))  # Nr of features
    print(len(allSubjFList[0]))  # Feature and name of feature
    print(allSubjFList[0][0].shape)  # Shape of all trials, feature 1
    # Shape of labels, first dim should be same as above
    print(allSubjFLabels.shape)
    # print(allSubjFList[1][0].shape)
    # print(allSubjFLabels.shape)

    return allSubjFList, allSubjFLabels


def main():

    testSize = 10  # Nr of seed iterations
    seedStart = 39  # Arbitrary, could be randomized as well.

    # Here, wrap all in another for loop that tests:
    # signAll, SignSolo, thresholds, paradigms, and saves them all in separateFolders
    # Fix soloSignThreshold ( Make sure it is correct)
    # Fix a timer for each loop of this for loop, as well as for each subject
    # and seed in the other two for loops. Save all times in a separate folder
    #

    # Loading parameters, what part of the trials to load and test
    t_min = 1.8
    t_max = 3
    sampling_rate = 256

    # Parameters for ANOVA test and ANOVA Feature Mask
    signAll = True
    signSolo = False
    globalSignificanceThreshold = 0.005  # 0.1 seems best, 0.05 a little faster
    soloSignificanceThreshold = 0.005

    # Tolerance for SVM SVC
    tolerance = 0.001  # Untested

    # Name for this test, what it is saved as
    validationRepetition = True
    repetitionName = "udrlBC3special005"
    repetitionValue = f"{46}{repetitionName}"

    # How many features that are maximally combined and tested together
    maxCombinationAmount = 3  # Depends on features. 3 can help with current

    # All the subjects that are tested, and used to create ANOVA Mask
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,

    quickTest = True  # Runs less hyperparameters

    # What paradigm to test

    # paradigm = paradigmSetting.upDownInner()
    # paradigm = paradigmSetting.upDownVis()
    # paradigm = paradigmSetting.upDownVisSpecial()
    # paradigm = paradigmSetting.upDownRightLeftInner()
    paradigm = paradigmSetting.upDownRightLeftInnerSpecial()
    # paradigm = paradigmSetting.upDownRightLeftVis()
    # paradigm = paradigmSetting.rightLeftInner()

    chunkFeatures = True
    chunkAmount = 6
    # What features that are created and tested
    featureList = [
        True,  # FFT 1
        False,  # Welch 2
        False,  # Hilbert 3 DataHR seems to not add much if any to FFT and Welch
        False,  # Powerbands 4
        False,  # FFT frequency buckets 5
        True,  # FFT Covariance 6
        False,  # Welch Covariance 7
        False,  # Hilbert Covariance 8 DataHR seems to not add much if any to FFT and Welch
        False,  # Covariance on smoothed Data 9
        False,  # Covariance on smoothed Data2 10
        False,  # Correlate1d # SEEMS BAD 11
        True,  # dataFFTCVBC 12
        False,  # dataWCVBC 13
        False,  # dataHRCVBC 14 DataHR seems to not add much if any to FFT and Welch
        True,  # fftDataBC 15
        False,  # welchDataBC 16
        False,  # dataHRBC 17 DataHR seems to not add much if any to FFT and Welch
        True,  # dataCVBC 18
        # More to be added
    ]

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

    # if signAll, then create or get globalGoodFeatures mask
    if signAll:
        for sub in subjects:
            if fClassDict[f"{sub}"].getGlobalGoodFeaturesMask() is None:
                allSubjFList, allSubjFLabels = combineAllSubjects(
                    fClassDict, subjectLeftOut=sub, onlyTrain=False
                )
                goodFeatureList, goodFeatureMaskList = anovaTest(
                    allSubjFList, allSubjFLabels, globalSignificanceThreshold
                )
                fClassDict[f"{sub}"].setGlobalGoodFeaturesMask(
                    goodFeatureMaskList
                )  # WHY IS THIS WEIRD SHAPE???
            else:
                goodFeatureMaskList = fClassDict[f"{sub}"].getGlobalGoodFeaturesMask(
                )

    if chunkFeatures:
        fClassDict2, bClassDict2 = createChunkFeatures(chunkAmount=chunkAmount)

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
        for sub in subjects:
            fClassDict[f"{sub}"].setOrder(seed)

        # For loop running pipeline on each subject
        for sub in subjects:  #

            print(f"Starting test of subject:{sub} , seed:{seed}")

            mDataList = mixShuffleSplit(
                fClassDict[f"{sub}"].getFeatureList(),
                labels=fClassDict[f"{sub}"].getLabels(),
                order=fClassDict[f"{sub}"].getOrder(),
                featureClass=fClassDict[f"{sub}"],
                maxCombinationAmount=maxCombinationAmount,
            )

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
                gdData,
            ) in mDataList:

                # print(f"\n Running dataset: {name} \n")
                print(
                    f" Test {testNr}/{testSize} - Progress {count}/{len(mDataList)}")
                count += 1

                # Below here can be switch to NN ? Create method? Or just different testSuite

                # allResults = fmetDict[f"{sub}"].testSuite(
                #     data_train,
                #     data_test,
                #     labels_train,
                #     labels_test,
                #     name,
                #     gdData,
                #     kernels=["linear", "sigmoid", "rbf"],  #
                # )

                allResults = fmetDict[f"{sub}"].testSuite(
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    name,
                    gdData,  #
                )

                allResultsPerSubject.append(allResults)

            savearray = np.array(
                [seed, sub, allResultsPerSubject], dtype=object)

            # Saving the results
            from datetime import datetime
            import os

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
