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

        print(goodfeature[0].shape)
        print(flatgoodfeature.shape)

        scaler.fit(flatfeature)
        flatfeature = scaler.transform(flatfeature)
        flatgoodfeature = scaler.transform(flatgoodfeature)

        f_statistic, p_values = feature_selection.f_classif(
            flatfeature, labels)

        p_values[
            p_values > significanceThreshold
        ] = 0  # Use sklearn selectpercentile instead?
        p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2
        goodData = f_statistic * p_values
        goodFeatureMaskList.append(goodData)
        goodfeature = flatfeature[:, np.where(goodData != 0)[0]]

        print(goodfeature.shape)
        print(np.count_nonzero(goodData))
        print(goodData.shape)
    return goodFeatureList, goodFeatureMaskList


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
    fClassDict = dict()
    fmetDict = dict()
    bClassDict = dict()
    testSize = 10
    seedStart = 39  # Arbitrary, could be randomized as well.

    # Try using ANOVA from all other subjects only. Because then you only need to do it
    # Once for every subject. Not every seed ( Done?!Yes?)

    # Here, wrap all in another for loop that tests:
    # signAll, SignSolo, thresholds, paradigms, and saves them all in separateFolders
    # Fix soloSignThreshold ( Make sure it is correct)
    # Fix a timer for each loop of this for loop, as well as for each subject
    # and seed in the other two for loops. Save all times in a separate folder
    #

    # Loading parameters
    t_min = 1.8
    t_max = 3
    sampling_rate = 256

    signAll = True
    signSolo = False
    # 0.1 seems best, barely any difference though , 0.05 a little faster
    globalSignificanceThreshold = 0.1
    soloSignificanceThreshold = 0.005
    tolerance = 0.001  # Untested
    validationRepetition = True
    repetitionName = "udrlBC2"
    repetitionValue = f"{24}{repetitionName}"
    maxCombinationAmount = 2  # Depends on features. 3 can help with current
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2,
    quickTest = True  # Runs less hyperparameters

    # paradigm = paradigmSetting.upDownInner()
    paradigm = paradigmSetting.upDownRightLeftInner()
    # paradigm = paradigmSetting.rightLeftInner()
    featureList = [
        True,  # FFT
        True,  # Welch
        True,  # Hilbert
        False,  # Powerbands
        False,  # FFT frequency buckets
        True,  # FFT Covariance
        True,  # Welch Covariance
        True,  # Hilbert Covariance
        True,  # Covariance on smoothed Data
        True,  # Covariance on smoothed Data 2
        False,  # Correlate1d # SEEMS BAD
        True,  # dataFFTCVBC
        True,  # dataWCVBC
        True,  # dataHRCVBC
        # More to be added
    ]

    # Creating the features for each subject and putting them in a dict
    for sub in subjects:  #

        fClassDict[f"{sub}"] = fclass.featureEClass(
            sub, paradigm[0], globalSignificance=globalSignificanceThreshold
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
        print(correctedExists)
        if correctedExists is False:
            # If loop here, checking if the specific corrected
            # Features have already been created. If so.
            # Load them and move on

            # Make it so they are saved at end of if loop if not
            #

            # Creating baselineData and Features
            bClassDict[f"{sub}"] = baseLineCorrection(
                subject=sub, sampling_rate=sampling_rate)

            bClassDict[f"{sub}"].loadBaselineData()

            bClassDict[f"{sub}"].getBaselineFeatures(
                t_min=t_min, t_max=t_max, featureList=featureList)

            fClassDict[f"{sub}"].correctedFeatureList = bClassDict[f"{sub}"].baselineCorrect(
                fClassDict[f"{sub}"].getFeatureList(
                ), fClassDict[f"{sub}"].getLabelsAux(),
                fClassDict[f"{sub}"].paradigmName)

        # fClassDict[f"{sub}"].saved

        # I only really want to correct some features

        # TODO: Here, get baseline as well and create the same features for them
        #
        # Then correct using that baseline to form new features
        # Bline corrected Features. Name them as such when saving
        # So two new things per feature to save. Baselines per day/subject
        # and corrected features

    # allSubjFList, allSubjFLabels = combineAllSubjects(fClassDict)
    # goodFeatureList, goodDataList = anovaTest(
    #     allSubjFList, allSubjFLabels, significanceThreshold
    # )
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
    testNr = 0
    # A for loop just running all subjects using different seeds for train/data split
    for seed in np.arange(seedStart * testSize, (seedStart + 1) * testSize):
        testNr += 1
        for sub in subjects:
            fClassDict[f"{sub}"].setOrder(seed)

        # For loop running pipeline on each subject
        for sub in subjects:  #

            print(f"Starting test of subject:{sub} , seed:{seed}")

            # Set the random order of shuffling for the subject/seed test

            # order = fClassDict[f"{sub}"].getOrder()

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
                allResults = fmetDict[f"{sub}"].testSuite(
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    name,
                    gdData,
                    kernels=["linear", "sigmoid", "rbf"],  #
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
