import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# import torch
# from torch import nn
# # from torch.utils.data import DataLoader
# # from torchvision import datasets, transforms
# import torch.optim as optim
# import torch.nn.functional as F
# # import cv2
# from tqdm import tqdm
# from random import shuffle


class SvmMets:

    # Init has many unnecessary things
    def __init__(
        self,
        signAll=False,
        signSolo=True,
        tol=0.001,
        significanceThreshold=0.1,
        verbose=True,
        quickTest=False,
        holdOut=False,
        calmTrials=None,
    ):
        print("new SvmMets")
        self.signAll = signAll
        self.signSolo = signSolo
        self.significanceThreshold = significanceThreshold  # like this and those above
        self.verbose = verbose
        self.tol = tol
        self.quickTest = quickTest
        self.hyperParams = None
        self.featCombos = []
        self.holdOut = holdOut
        self.calmTrials = calmTrials

        if verbose is not True:
            import warnings

            warnings.filterwarnings(
                action="ignore", category=UserWarning
            )  # setting ignore as a parameter and further adding category
            warnings.filterwarnings(
                action="ignore", category=RuntimeWarning
            )  # setting ignore as a parameter and further adding category

        if self.signAll or self.signSolo:
            self.onlySign = True
        else:
            self.onlySign = False
        """
        This class handles SVM pipeline testing.
        Right now it is very janky!
        """

    def addCalmTrials(self, calmTrialList):
        self.calmTrials = calmTrialList
        print("Added calmTrials")

    # Does not seem to improve much at the moment if at all! I think, some sore of early stopping is probably better.
    # or regularization
    # Which is the C value. So check hyper parameters at the end, at 3 or 4 features. Do 3.
    def svmPipelineAda(
        self,
        ndata_train,
        ndata_test,
        labels_train,
        labels_test,
        kernel="linear",
        degree=3,
        gamma="auto",
        C=1,
    ):
        # coefs=None,
        # goodData,
        """
        Pipeline using SVM

        Args:
            data_train (np.array): Training data for SVM pipeline
            data_test (np.array): Test data for SVM pipeline
            labels_train (np.array): Training labels for SVM pipeline
            labels_test (np.array): Test labels for SVM pipeline
            kernel (str, optional): What kernel the SVM pipeline should use. Defaults to "linear".
            degree (int, optional): Degree of SVM pipeline. Defaults to 3.
            gamma (str, optional): Gamma of SVM pipeline. Defaults to "auto".
            C (int, optional): Learning coeffecient for SVM Pipeline. Defaults to 1.
            coefs (_type_, optional): When SelectKBest is used, these are its coefficients
            . Defaults to None.

        Returns:
            _type_: _description_
        """
        # if coefs is None:
        #     coefs = np.zeros([1, ndata_train.shape[1]])

        # from sklearn import multioutput as multiO, create new class for multiOutput

        from sklearn.ensemble import AdaBoostClassifier

        ada = AdaBoostClassifier(
            SVC(  # anova_filter/#,
                gamma=gamma,
                kernel=kernel,
                degree=degree,
                verbose=False,
                C=C,
                cache_size=1800,
                tol=self.tol,
                probability=True,
            ),
            n_estimators=20,
            learning_rate=1.0,
        )

        ada.fit(ndata_train, labels_train)
        predictions = ada.predict(ndata_test)

        correct = np.zeros(labels_test.shape)
        correctamount = 0
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        return correctamount / labels_test.shape[0]  # , coefs

    def svmPipeline(
        self,
        ndata_train,
        ndata_test,
        labels_train,
        labels_test,
        kernel="linear",
        degree=3,
        gamma="auto",
        C=1,
    ):
        # coefs=None,
        # goodData,
        """
        Pipeline using SVM

        Args:
            data_train (np.array): Training data for SVM pipeline
            data_test (np.array): Test data for SVM pipeline
            labels_train (np.array): Training labels for SVM pipeline
            labels_test (np.array): Test labels for SVM pipeline
            kernel (str, optional): What kernel the SVM pipeline should use. Defaults to "linear".
            degree (int, optional): Degree of SVM pipeline. Defaults to 3.
            gamma (str, optional): Gamma of SVM pipeline. Defaults to "auto".
            C (int, optional): Learning coeffecient for SVM Pipeline. Defaults to 1.
            coefs (_type_, optional): When SelectKBest is used, these are its coefficients
            . Defaults to None.

        Returns:
            _type_: _description_
        """
        # if coefs is None:
        #     coefs = np.zeros([1, ndata_train.shape[1]])

        # from sklearn import multioutput as multiO, create new class for multiOutput

        clf = make_pipeline(
            SVC(  # anova_filter/#,
                gamma=gamma,
                kernel=kernel,
                degree=degree,
                verbose=False,
                C=C,
                cache_size=1800,
                tol=self.tol,

            ),
        )

        clf.fit(ndata_train, labels_train)
        predictions = clf.predict(ndata_test)
        correct = np.zeros(labels_test.shape)
        correctamount = 0
        # print(f"Labels test : {labels_test}")
        # print(f"Predictions test : {predictions}")
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        return correctamount / labels_test.shape[0]  # , coefs

    def svmPipelineOVRHoldOut(
        self,
        ndata_train,
        ndata_test,
        labels_train,
        labels_test,
        kernel="linear",
        degree=3,
        gamma="auto",
        C=1,
        data_testC=None,
        data_trainC=None
    ):
        # coefs=None,
        # goodData,
        """
        Pipeline using SVM

        Args:
            data_train (np.array): Training data for SVM pipeline
            data_test (np.array): Test data for SVM pipeline
            labels_train (np.array): Training labels for SVM pipeline
            labels_test (np.array): Test labels for SVM pipeline
            kernel (str, optional): What kernel the SVM pipeline should use. Defaults to "linear".
            degree (int, optional): Degree of SVM pipeline. Defaults to 3.
            gamma (str, optional): Gamma of SVM pipeline. Defaults to "auto".
            C (int, optional): Learning coeffecient for SVM Pipeline. Defaults to 1.
            coefs (_type_, optional): When SelectKBest is used, these are its coefficients
            . Defaults to None.

        Returns:
            _type_: _description_
        """
        # if coefs is None:
        #     coefs = np.zeros([1, ndata_train.shape[1]])

        # from sklearn import multioutput as multiO, create new class for multiOutput
        from sklearn.multiclass import OneVsRestClassifier

        if data_trainC is not None:
            clf = OneVsRestClassifier(SVC(gamma=gamma,
                                          kernel=kernel,
                                          degree=degree,
                                          verbose=False,
                                          C=C,
                                          cache_size=1800,
                                          tol=self.tol,
                                          probability=True
                                          ))
        else:
            clf = OneVsRestClassifier(SVC(gamma=gamma,
                                          kernel=kernel,
                                          degree=degree,
                                          verbose=False,
                                          C=C,
                                          cache_size=1800,
                                          tol=self.tol,
                                          ))
        if self.holdOut:

            sss = StratifiedShuffleSplit(1, train_size=0.66, test_size=0.33, random_state=1
                                         )

            for train_index, test_index in sss.split(
                X=np.zeros(labels_test.shape[0]), y=labels_test
            ):
                ndata_test2 = ndata_test[test_index]
                ndata_test = ndata_test[train_index]
                labels_test2 = labels_test[test_index]
                labels_test = labels_test[train_index]
        else:
            ndata_test2 = ndata_test
            ndata_test = ndata_test
            labels_test2 = labels_test
            labels_test = labels_test

        clf.fit(ndata_train, labels_train)
        scoresList = []

        if data_trainC is not None:
            probsCalmData = clf.predict_proba(data_trainC)
            probsCalmDataTest = clf.predict_proba(data_testC)
            # print("Labels below")
            # print(labels_test)
            diffList = []
            uniqueLabels = np.unique(labels_test)
            for labelInd, label in enumerate(uniqueLabels):
                indexes = np.where(labels_test == label)
                probsLabeledData = clf.predict_proba(ndata_test[indexes])
                probsCalmData = np.sort(probsCalmData, axis=1)
                probsLabeledData = np.sort(probsLabeledData, axis=1)
                highestToSecondHighestLabeled = probsLabeledData[:, -
                                                                 1] - probsLabeledData[:, -2]
                highestToSecondHighestCalm = probsCalmData[:, -
                                                           1] - probsCalmData[:, -2]
                diff = np.mean(
                    highestToSecondHighestLabeled) - np.mean(highestToSecondHighestCalm)
                diffList.append(diff)
            # # print(probsLabeledData)

            # # print(probsCalmData)
            # print("Calm Below")
            # print(np.max(probsCalmData, axis=1))
            # print("Prob labels below")
            # print(np.max(probsLabeledData, axis=1))
            probsLabeledData = clf.predict_proba(ndata_test)
            probsCalmData = np.sort(probsCalmData, axis=1)
            probsCalmDataTest = np.sort(probsCalmDataTest, axis=1)

            probsLabeledData = np.sort(probsLabeledData, axis=1)
            highestToSecondHighestLabeled = probsLabeledData[:, -
                                                             1] - probsLabeledData[:, -2]
            highestToSecondHighestCalm = probsCalmData[:, -
                                                       1] - probsCalmData[:, -2]
            highestToSecondHighestCalmTest = probsCalmDataTest[:, -
                                                               1] - probsCalmDataTest[:, -2]
            diff = np.mean(
                highestToSecondHighestLabeled) - np.mean(highestToSecondHighestCalm)
            diffList.append(diff)
            diffThresh = np.percentile(highestToSecondHighestCalm, 90)

            highestToSecondHighestCalmTest[highestToSecondHighestCalmTest > diffThresh] = 0
            correctCalmTest = np.count_nonzero(
                highestToSecondHighestCalmTest) / len(highestToSecondHighestCalmTest)

            # probsLabeledData = clf.predict_proba(ndata_test)
            # diff = np.mean(np.max(probsLabeledData, axis=1)) - \
            #     np.mean(np.max(probsCalmData, axis=1))

            # print(diff)
        for ndata_test, labels_test in zip([ndata_test, ndata_test2],
                                           [labels_test, labels_test2]):

            predictions = clf.predict(ndata_test)
            probsLabeledData = clf.predict_proba(ndata_test)
            probsLabeledDataSorted = np.sort(probsLabeledData, axis=1)
            highestToSecondHighestLabeled = probsLabeledDataSorted[:, -
                                                                   1] - probsLabeledDataSorted[:, -2]
            BadPredictions = np.where(
                highestToSecondHighestLabeled < diffThresh)[0]
            correct = np.zeros(labels_test.shape)
            correctamount = 0
            # print(clf.score(ndata_test, labels_test))
            # print(clf.decision_function(ndata_train))
            # print(f"Labels test : {labels_test}")
            # print(f"Predictions test : {predictions}")
            for nr, pred in enumerate(predictions, 0):
                if pred == labels_test[nr]:
                    correct[nr] = 1
                    correctamount += 1
            # sepScores1 = self.scoresSepLabels(clf, ndata_test, labels_test)
            sepScores = self.scoresSepLabels2(
                clf, ndata_test, labels_test, BadPredictions)
            sepScores.append(correctCalmTest)
            wrongCorrect = 0
            for label, pred, lInd in zip(labels_test, predictions, np.arange(len(labels_test))):
                if lInd in BadPredictions and label == pred:
                    wrongCorrect += 1
            wrongCorrectPercent = wrongCorrect / len(labels_test)
            score = clf.score(ndata_test, labels_test)
            score = score - wrongCorrectPercent
            scores = []
            scores.append(score)
            for sepscore in sepScores:
                scores.append(sepscore)
            # print(sepScores)
            scoresList.append(scores)

        # correctamount / labels_test.shape[0]  # , coefs
        return scoresList[0], scoresList[1], diffList

    def svmPipelineOVR(
        self,
        ndata_train,
        ndata_test,
        labels_train,
        labels_test,
        kernel="linear",
        degree=3,
        gamma="auto",
        C=1,
    ):
        # coefs=None,
        # goodData,
        """
        Pipeline using SVM

        Args:
            data_train (np.array): Training data for SVM pipeline
            data_test (np.array): Test data for SVM pipeline
            labels_train (np.array): Training labels for SVM pipeline
            labels_test (np.array): Test labels for SVM pipeline
            kernel (str, optional): What kernel the SVM pipeline should use. Defaults to "linear".
            degree (int, optional): Degree of SVM pipeline. Defaults to 3.
            gamma (str, optional): Gamma of SVM pipeline. Defaults to "auto".
            C (int, optional): Learning coeffecient for SVM Pipeline. Defaults to 1.
            coefs (_type_, optional): When SelectKBest is used, these are its coefficients
            . Defaults to None.

        Returns:
            _type_: _description_
        """
        # if coefs is None:
        #     coefs = np.zeros([1, ndata_train.shape[1]])

        # from sklearn import multioutput as multiO, create new class for multiOutput
        from sklearn.multiclass import OneVsRestClassifier

        clf = OneVsRestClassifier(SVC(gamma=gamma,
                                      kernel=kernel,
                                      degree=degree,
                                      verbose=False,
                                      C=C,
                                      cache_size=1800,
                                      tol=self.tol,
                                      ))

        clf.fit(ndata_train, labels_train)
        predictions = clf.predict(ndata_test)
        correct = np.zeros(labels_test.shape)
        correctamount = 0
        # print(clf.score(ndata_test, labels_test))
        # print(clf.decision_function(ndata_train))
        # print(f"Labels test : {labels_test}")
        # print(f"Predictions test : {predictions}")
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1
        sepScores = self.scoresSepLabels(clf, ndata_test, labels_test)
        score = clf.score(ndata_test, labels_test)
        scores = []
        scores.append(score)
        for sepscore in sepScores:
            scores.append(sepscore)
        # print(sepScores)
        return scores  # correctamount / labels_test.shape[0]  # , coefs

    def scoresSepLabels(self, clf, data, labels):
        uniqueLabels = np.unique(labels)
        scoresSep = []
        for label in uniqueLabels:
            indexes = np.where(labels == label)

            scoresSep.append(clf.score(data[indexes], labels[indexes]))
        return scoresSep

    def scoresSepLabels2(self, clf, data, labels, badPredictions):
        uniqueLabels = np.unique(labels)
        scoresSep = []
        predLabels = clf.predict(data)
        for label in uniqueLabels:
            correct = 0
            total = 0
            for plabel, tlabel, ind in zip(predLabels, labels, np.arange(len(labels))):
                if tlabel == label:
                    if plabel == label:
                        if ind not in badPredictions:
                            correct += 1
                else:
                    if plabel != label:
                        correct += 1
                total += 1

            # indexes = np.where(labels == label)

            scoresSep.append(correct / total)
        return scoresSep

    def testSuiteOVR(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,
        kernels=["linear", "rbf", "sigmoid"],
    ):
        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)
        # print("Shuffling")
        # np.random.shuffle(labels_train)
        # print(labels_train.shape)
        # print("here")
        # labels_train = np.random.shuffle(labels_train)
        allResults = []
        if self.quickTest:
            clist = [2.5]
        else:
            clist = [0.1, 1, 10, 100, 1000]
            # Regularization parameter.
            # The strength of the regularization is inversely proportional to C.
            # Must be strictly positive. The penalty is a squared l2 penalty.
        # testing using different kernels, C and degrees.
        for kernel in kernels:
            if kernel == "linear":
                c = clist[0]
                for degree in range(1, 2):
                    res = self.svmPipelineOVR(
                        ndata_train,
                        ndata_test,
                        labels_train,
                        labels_test,
                        # goodData=goodData,
                        degree=degree,
                        kernel=kernel,
                        C=c,
                        # coefs=coefs,
                    )
                    if self.verbose:
                        print(
                            "Result for degree {}, kernel {}, C = {}: {}".format(
                                degree, kernel, (c * 100 // 10) / 10, res
                            )
                        )
                    allResults.append([name, res, kernel, c])

            else:
                for c in clist:
                    for gamma in ["auto"]:
                        res = self.svmPipelineOVR(
                            ndata_train,
                            ndata_test,
                            labels_train,
                            labels_test,
                            # goodData=goodData,
                            degree=degree,
                            kernel=kernel,
                            gamma=gamma,
                            C=c,
                        )
                        if self.verbose:
                            print(
                                "Result for gamma {}, kernel {}, C = {}: {}".format(
                                    gamma, kernel, (c * 100 // 10) / 10, res[0]
                                )
                            )
                        allResults.append([name, res, kernel, c])

        # coefs = np.reshape(coefs, [128, -1])
        # hyperParams = []
        # hyperParams.append([kernels, clist])
        if name not in self.featCombos:
            self.featCombos.append(name)
        self.hyperParams = [kernels, clist]
        return np.array(allResults, dtype=object)

    def testSuiteOVRHoldOut(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,
        kernels=["linear", "rbf", "sigmoid"],
        data_testC=None,
        data_trainC=None,
    ):
        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        # if "stftData" not in name:
        #     return np.array([name, 0, "linear", 2.5, 0], dtype=object)
        scaler = StandardScaler()
        scaler = scaler.fit(data_trainC)
        ndata_testC = scaler.transform(data_testC)
        ndata_trainC = scaler.transform(data_trainC)

        # print("Shuffling")
        # np.random.shuffle(labels_train)
        # print(labels_train.shape)
        # print("here")
        # labels_train = np.random.shuffle(labels_train)
        allResults = []
        if self.quickTest:
            clist = [2.5]
        else:
            clist = [0.1, 1, 10, 100, 1000]
            # Regularization parameter.
            # The strength of the regularization is inversely proportional to C.
            # Must be strictly positive. The penalty is a squared l2 penalty.
        # testing using different kernels, C and degrees.
        for kernel in kernels:
            if kernel == "linear":
                c = clist[0]
                for degree in range(1, 2):
                    res, hres, diff = self.svmPipelineOVRHoldOut(
                        ndata_train,
                        ndata_test,
                        labels_train,
                        labels_test,
                        # goodData=goodData,
                        degree=degree,
                        kernel=kernel,
                        C=c,
                        data_testC=ndata_testC,
                        data_trainC=ndata_trainC,
                        # coefs=coefs,
                    )
                    if self.verbose:
                        print(
                            "Result for degree {}, kernel {}, C = {}: {}".format(
                                degree, kernel, (c * 100 // 10) / 10, res
                            )
                        )
                    allResults.append([name, res, kernel, c, hres, diff])

            else:
                for c in clist:
                    for gamma in ["auto"]:
                        res, hres, diff = self.svmPipelineOVRHoldOut(
                            ndata_train,
                            ndata_test,
                            labels_train,
                            labels_test,
                            # goodData=goodData,
                            degree=degree,
                            kernel=kernel,
                            gamma=gamma,
                            C=c,
                            data_testC=ndata_testC,
                            data_trainC=ndata_trainC,
                        )
                        if self.verbose:
                            print(
                                "Result for gamma {}, kernel {}, C = {}: {}".format(
                                    gamma, kernel, (c * 100 // 10) / 10, res[0]
                                )
                            )
                        allResults.append([name, res, kernel, c, hres, diff])

        # coefs = np.reshape(coefs, [128, -1])
        # hyperParams = []
        # hyperParams.append([kernels, clist])
        if name not in self.featCombos:
            self.featCombos.append(name)
        self.hyperParams = [kernels, clist]
        return np.array(allResults, dtype=object)

    def testSuite(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,
        kernels=["linear", "rbf", "sigmoid"],
    ):
        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)
        # print("Shuffling")
        # np.random.shuffle(labels_train)
        # print(labels_train.shape)
        # print("here")
        # labels_train = np.random.shuffle(labels_train)
        allResults = []
        if self.quickTest:
            clist = [2.5]
        else:
            clist = [0.1, 1, 10, 100, 1000]
            # Regularization parameter.
            # The strength of the regularization is inversely proportional to C.
            # Must be strictly positive. The penalty is a squared l2 penalty.
        # testing using different kernels, C and degrees.
        for kernel in kernels:
            if kernel == "linear":
                c = clist[0]
                for degree in range(1, 2):
                    res = self.svmPipeline(
                        ndata_train,
                        ndata_test,
                        labels_train,
                        labels_test,
                        # goodData=goodData,
                        degree=degree,
                        kernel=kernel,
                        C=c,
                        # coefs=coefs,
                    )
                    if self.verbose:
                        print(
                            "Result for degree {}, kernel {}, C = {}: {}".format(
                                degree, kernel, (c * 100 // 10) / 10, res
                            )
                        )
                    allResults.append([name, res, kernel, c])

            else:
                for c in clist:
                    for gamma in ["auto"]:
                        res = self.svmPipeline(
                            ndata_train,
                            ndata_test,
                            labels_train,
                            labels_test,
                            # goodData=goodData,
                            degree=degree,
                            kernel=kernel,
                            gamma=gamma,
                            C=c,
                        )
                        if self.verbose:
                            print(
                                "Result for gamma {}, kernel {}, C = {}: {}".format(
                                    gamma, kernel, (c * 100 // 10) / 10, res[0]
                                )
                            )
                        allResults.append([name, res, kernel, c])

        # coefs = np.reshape(coefs, [128, -1])
        # hyperParams = []
        # hyperParams.append([kernels, clist])
        if name not in self.featCombos:
            self.featCombos.append(name)
        self.hyperParams = [kernels, clist]
        return np.array(allResults, dtype=object)

    def getTestInfo(self):
        return self.hyperParams, self.featCombos

    def testSuiteAda(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,
        kernels=["linear", "rbf", "sigmoid"],
    ):

        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        allResults = []
        if self.quickTest:
            clist = [2.5]
        else:
            clist = np.linspace(0.5, 5, 5)

        c = clist[0]
        kernels = ["adaBoost"]
        kernel = "adaBoost"

        from sklearn.ensemble import AdaBoostClassifier

        ada = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
        )

        ada.fit(ndata_train, labels_train)
        predictions = ada.predict(ndata_test)

        correct = np.zeros(labels_test.shape)
        correctamount = 0
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        res = correctamount / labels_test.shape[0]  # , coefs

        allResults.append([name, res, kernel, c])
        # allResults.append([name, res, kernel, c])

        if name not in self.featCombos:
            self.featCombos.append(name)
        self.hyperParams = [kernels, clist]
        return np.array(allResults, dtype=object)

    def svmPipelineForest(
        self,
        ndata_train,
        ndata_test,
        labels_train,
        labels_test,
        kernel="linear",
        degree=3,
        gamma="auto",
        C=1,
    ):
        # coefs=None,
        # goodData,
        """
        Pipeline using SVM

        Args:
            data_train (np.array): Training data for SVM pipeline
            data_test (np.array): Test data for SVM pipeline
            labels_train (np.array): Training labels for SVM pipeline
            labels_test (np.array): Test labels for SVM pipeline
            kernel (str, optional): What kernel the SVM pipeline should use. Defaults to "linear".
            degree (int, optional): Degree of SVM pipeline. Defaults to 3.
            gamma (str, optional): Gamma of SVM pipeline. Defaults to "auto".
            C (int, optional): Learning coeffecient for SVM Pipeline. Defaults to 1.shape
            coefs (_type_, optional): When SelectKBest is used, these are its coefficients
            . Defaults to None.

        Returns:
            _type_: _description_
        """
        # if coefs is None:
        #     coefs = np.zeros([1, ndata_train.shape[1]])

        # from sklearn import multioutput as multiO, create new class for multiOutput

        clf = make_pipeline(
            SVC(  # anova_filter/#,
                gamma=gamma,
                kernel=kernel,
                degree=degree,
                verbose=False,
                C=C,
                cache_size=1800,
                tol=self.tol,
            ),
        )

        clf.fit(ndata_train, labels_train)
        predictions = clf.predict(ndata_test)

        correct = np.zeros(labels_test.shape)
        correctamount = 0
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        return correctamount / labels_test.shape[0]  # , coefs

    def testSuiteForest(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,
        kernels=["linear", "rbf", "sigmoid"],
    ):

        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        from sklearn.ensemble import RandomForestClassifier

        # from sklearn.neural_network import MLPClassifier

        # mlp = MLPClassifier(hidden_layer_sizes=(2000,1000), solver="lbfgs", activation="relu")

        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(ndata_train, labels_train)
        predictions = rfc.predict(ndata_test)

        correct = np.zeros(labels_test.shape)
        correctamount = 0
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        res = correctamount / labels_test.shape[0]
        allResults = []
        allResults.append([name, res, "gini", 1])
        allResults.append([name, res, "gini", 1])

        return np.array(allResults, dtype=object), ["gini", "1"]

#     def testSuiteMLP(
#         self,
#         data_train,
#         data_test,
#         labels_train,
#         labels_test,
#         name,
#         kernels=["linear", "rbf", "sigmoid"],
#     ):
#         scaler = StandardScaler()
#         scaler = scaler.fit(data_train)

#         # Shape of trials and features, should be
#         ndata_train = scaler.transform(data_train)
#         ndata_test = scaler.transform(data_test)
#         # ndata_train = np.reshape(ndata_train, [ndata_train.shape[0], 1, 1, -1])
#         # ndata_test = np.reshape(ndata_test, [ndata_test.shape[0], 1, 1, -1])

#         device = "cuda"
#         # from torch.utils.data import Dataset
#         torch_train = []
#         for trialData, trialLabel in zip(ndata_train, labels_train):
#             torch_trainData = torch.FloatTensor(trialData)
#             t = torch.LongTensor(1)
#             t[0] = int(trialLabel)
#             torch_train.append([torch_trainData, t])
#         print(labels_train)
#         torch_test = []
#         for trialData, trialLabel in zip(ndata_test, labels_test):
#             torch_testData = torch.FloatTensor(trialData)
#             t = torch.LongTensor(1)
#             t[0] = int(trialLabel)
#             torch_test.append([torch_testData, t])

#         print(f"size of training data {len(torch_train)}")
#         print(f"size of testing data {len(torch_test)}")
#         print(torch_train[0][0].shape)
#         print(torch_train[0][1].shape)
#         print(torch_train[0][1])
#         net = Net(featureSize=ndata_train.shape[-1])
#         net.to(device)
#         self.train_model(net=net, epochs=100, train_data=torch_train, device=device,
#                          batchSize=10, trainSize=len(torch_train), featureSize=ndata_train.shape[-1])

#         acc = self.test_model(
#             net=net, device=device, test_data=torch_test, featureSize=ndata_train.shape[-1])
#         print(acc)
#         allResults = []
#         allResults.append([name, acc, "MLP", 2.5])
#         allResults.append([name, acc, "MLP", 2.5])
#         return np.array(allResults, dtype=object)

#     def train_model(self, net, train_data, device, epochs, batchSize=10, trainSize=610, featureSize=50 * 50):
#         optimizer = optim.Adam(
#             net.parameters(), lr=0.0001, weight_decay=0.00001)
#         loss_function = nn.CrossEntropyLoss()

#         for epoch in tqdm(range(epochs)):
#             for i in (range(0, trainSize, batchSize)):
#                 if batchSize + i > trainSize:
#                     continue
#                 batch = train_data[i:i + batchSize]
#                 batch_x = torch.cuda.FloatTensor(batchSize, 1, featureSize)
#                 batch_y = torch.cuda.LongTensor(batchSize, 1)

#                 for k in range(batchSize):
#                     batch_x[k] = batch[k][0]
#                     batch_y[k] = batch[k][1]
#                 batch_x.to(device)
#                 batch_y.to(device)
#                 net.zero_grad()
#                 outputs = net(batch_x.view(-1, 1, featureSize))
#                 batch_y = batch_y.view(batchSize)
#                 loss = F.nll_loss(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#             print(f"epoch : {epoch}  loss : {loss}")

#     def test_model(self, net, device, test_data, featureSize=50 * 50):
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for data in tqdm(test_data):
#                 x = torch.FloatTensor(data[0])
#                 y = torch.LongTensor(data[1])

#                 x = x.view(-1, 1, featureSize)
#                 x = x.to(device)
#                 output = net(x)
#                 output = output.view(2)
#                 if (max(output[0], output[1]) == output[0]):
#                     index = 0
#                 else:
#                     index = 1
#                 if index == y[0]:
#                     correct += 1
#                 total += 1
#             return round(correct / total, 5)


# class Net(nn.Module, ):
#     def __init__(self, featureSize=50 * 50):
#         super().__init__()
#         # self.conv1 = nn.Conv2d(1, 32, 2)
#         # self.conv2 = nn.Conv2d(32, 64, 2)
#         # self.conv3 = nn.Conv2d(64, 128, 2)
#         # nn.Conv2d(1,32,5,)
#         # nn.Conv1d(1,32,1,1)
#         # nn.Conv1d
#         self.lin1 = nn.LazyLinear(1028)
#         self.lin2 = nn.LazyLinear(512)
#         self.lin3 = nn.LazyLinear(256)
#         # self.conv1 = nn.Conv1d(1, 32, 5, 2)
#         self.conv2 = nn.Conv1d(32, 64, 5, 2)
#         self.conv3 = nn.Conv1d(64, 128, 5, 2)
#         x = torch.rand(1, featureSize).view(-1, 1, featureSize)
#         self.linear_in = None
#         self.convs(x)

#         self.fc1 = nn.Linear(self.linear_in, 512)
#         self.fc2 = nn.Linear(512, 2)

#     def convs(self, x):
#         # x = F.relu(self.conv1(x))
#         # x = F.relu(self.lin1(x))
#         # x = F.relu(self.conv2(x))
#         # x = F.relu(self.conv3(x))
#         x = F.relu(self.lin1(x))
#         x = F.relu(self.lin2(x))
#         x = F.relu(self.lin3(x))
#         if self.linear_in is None:
#             self.linear_in = x[0].shape[0] * x[0].shape[1]  # * x[0].shape[2]
#         else:
#             return x

#     def forward(self, x):
#         x = self.convs(x)
#         x = x.view(-1, self.linear_in)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)