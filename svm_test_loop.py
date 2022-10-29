"""
This class runs a pipeline testing SVM classification on data
"""

from copy import deepcopy as dp
import numpy as np
import feature_extraction as fclass
import svmMethods as svmMet
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler
import paradigmSetting


def mixShuffleSplit(
    createdFeatureList, labels, order, featureClass, maxCombinationAmount, goodDataList
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
        goodDataList=goodDataList,
    )
    return mDataList


def anovaTest(featureList, labels, significanceThreshold):

    scaler = StandardScaler()

    goodFeatureList = dp(featureList)
    goodDataList = []
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
        goodDataList.append(goodData)
        goodfeature = flatfeature[:, np.where(goodData != 0)[0]]

        print(goodfeature.shape)
        print(np.count_nonzero(goodData))
        print(goodData.shape)
    return goodFeatureList, goodDataList


def combineAllSubjects(fclassDict, onlyTrain=False):

    first = True
    for subName, fClass in fclassDict.items():
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
    testSize = 50
    seedStart = 39  # Arbitrary, could be randomized as well.
    # Since Anova on one Subject gives immense results. There is lots of subject specific data
    # onlySignificantFeatures = True

    # Try using ANOVA from all other subjects only. Because then you only need to do it
    # Once for every subject. Not every seed
    signAll = False
    signSolo = False
    # 0.1 seems best, barely any difference though , 0.05 a little faster
    significanceThreshold = 0.1
    tolerance = 0.001  # Untested
    validationRepetition = True
    repetitionName = "x01sign4labels"
    repetitionValue = f"{15}{repetitionName}"
    maxCombinationAmount = 3  # Depends on features. 3 can help with current
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]   # 2,
    quickTest = True  # Runs less hyperparameters

    # paradigms = paradigmSetting.upDownInner()
    paradigms = paradigmSetting.upDownRightLeftInner()
    # paradigms = paradigmSetting.rightLeftInner()

    # Running the same seed and subjects again in a different folder after changing code to see
    # if the results are the same

    # Creating the features for each subject and putting them in a dict
    for sub in subjects:  #

        fClassDict[f"{sub}"] = fclass.featureEClass(sub)
        fmetDict[f"{sub}"] = svmMet.SvmMets(
            significanceThreshold=significanceThreshold,
            signAll=signAll,
            signSolo=signSolo,
            verbose=False,
            tol=tolerance,
            quickTest=quickTest
        )
        print(f"Creating features for subject:{sub}")
        createdFeatureList, labels = fClassDict[f"{sub}"].getFeatures(
            paradigms=paradigms,
            subject=sub,
            t_min=1.8,
            t_max=3,
            sampling_rate=256,
            twoDLabels=False,
            maxCombinationAmount=maxCombinationAmount,
            featureList=[
                True,  # FFT
                True,  # Welch
                True,  # Hilbert
                False,  # Powerbands
                False,  # FFT frequency buckets
                True,  # FFT Covariance
                True,  # Welch Covariance
                True,  # Hilbert Covariance
                True,  # Covariance on smoothed Data
                False,  # Covariance on smoothed Data 2
                # More to be added
            ],
            verbose=True,
        )

    # allSubjFList, allSubjFLabels = combineAllSubjects(fClassDict)
    # goodFeatureList, goodDataList = anovaTest(
    #     allSubjFList, allSubjFLabels, significanceThreshold
    # )

    # A for loop just running all subjects using different seeds for train/data split
    for seed in np.arange(seedStart * testSize, (seedStart + 1) * testSize):

        for sub in subjects:
            fClassDict[f"{sub}"].setOrder(seed)

        # For loop running pipeline on each subject
        for sub in subjects:  #

            print(f"Starting test of subject:{sub} , seed:{seed}")

            # Set the random order of shuffling for the subject/seed test

            # order = fClassDict[f"{sub}"].getOrder()

            # Combine all subjects, only training sets. And do Anova on
            allSubjFList, allSubjFLabels = combineAllSubjects(
                fClassDict, onlyTrain=True
            )
            goodFeatureList, goodDataList = anovaTest(
                allSubjFList, allSubjFLabels, significanceThreshold
            )

            mDataList = mixShuffleSplit(
                fClassDict[f"{sub}"].getFeatureList(),
                labels=fClassDict[f"{sub}"].getLabels(),
                order=fClassDict[f"{sub}"].getOrder(),
                featureClass=fClassDict[f"{sub}"],
                maxCombinationAmount=maxCombinationAmount,
                goodDataList=goodDataList,
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
                print(f" Progress {count}/{len(mDataList)}")
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
    main()
