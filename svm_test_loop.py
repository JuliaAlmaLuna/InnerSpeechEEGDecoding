"""
This class runs a pipeline testing SVM classification on data
"""

import numpy as np
import feature_extraction as fclass
import svmMethods as svmMet

fClassDict = dict()
fmetDict = dict()
testSize = 100
seedStart = 19  # Arbitrary, could be randomized as well.

# A for loop just running all subjects using different seeds for train/data split
for seed in np.arange(seedStart*testSize, (seedStart+1)*testSize):

    np.random.seed(seed)

    # For loop running pipeline on each subject
    for sub in [1, 2, 3, 4, 5, 6, 7, 8, 9]:

        print(f"Starting test of subject:{sub} , seed:{seed}")

        # Adding featureclass and svm Methods class to a dict, so they can be separate when
        # running multiple subjects/seeds at the same time using multiprocessing.
        fClassDict[f"{seed},{sub}"] = fclass.featureEClass()
        fmetDict[f"{seed},{sub}"] = svmMet.SvmMets()

        specificSubject = sub

        # uses class feature_extraction to get combinations of features, split into test and training. With labels
        mDataList = fClassDict[f"{seed},{sub}"].getFeatures(specificSubject, t_min=2, t_max=3,
                                                            sampling_rate=256,
                                                            twoDLabels=False)

        allResultsPerSubject = []
        # For loop of each combination of features
        # Training a SVM using each one and then saving the results
        for data_train, data_test, labels_train, labels_test, name in mDataList:

            print(f"\n Running dataset: {name} \n")

            allResults = fmetDict[f"{seed},{sub}"].testSuite(
                data_train, data_test, labels_train, labels_test, name)

            allResultsPerSubject.append(allResults)

        savearray = np.array([seed,
                             specificSubject, allResultsPerSubject], dtype=object)

        # Saving the results
        from datetime import datetime
        import os

        now = datetime.now()
        # Month abbreviation, day and year
        now_string = now.strftime("D--%d-%m-%Y-T--%H-%M")
        foldername = now.strftime("%d-%m")

        saveDir = f"F:/PythonProjects/NietoExcercise-1/SavedResults/{foldername}"
        os.makedirs(saveDir)
        np.save(
            f"{saveDir}/savedBestSeed-{seed}-Subject-{specificSubject}-Date-{now_string}", savearray)
