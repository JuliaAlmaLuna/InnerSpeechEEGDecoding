import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection
import tensorflow as tf
from tensorflow.keras import layers  # ,regularizers

# import tensorflow_hub as hub  # type:ignore

import matplotlib.pyplot as plt

# from sklearn.feature_selection import f_classif  # SelectKBest ,


class SvmMets:
    def __init__(
        self,
        signAll=False,
        signSolo=True,
        tol=0.001,
        significanceThreshold=0.1,
        verbose=True,
        quickTest=False,
    ):
        print("new SvmMets")
        self.signAll = signAll
        self.signSolo = signSolo
        self.significanceThreshold = significanceThreshold
        self.verbose = verbose
        self.tol = tol
        self.quickTest = quickTest

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

    def plotHeatMaps(self, plotData):
        # Plots heatmaps, used for covariance features.
        # This function does not need to be in this class
        plt.figure()
        plt.imshow(plotData, cmap="hot", interpolation="nearest")
        plt.show()

    from functools import lru_cache

    # @lru_cache(10)
    def anovaSolo(self, ndata_train, labels_train):

        # from sklearn.feature_selection import VarianceThreshold

        # vartresh = VarianceThreshold()
        # ndata_train = vartresh.fit_transform(ndata_train)
        f_statistic, p_values = feature_selection.f_classif(
            ndata_train, labels_train)
        p_values[
            p_values > self.significanceThreshold
        ] = 0  # Use sklearn selectpercentile instead?
        p_values[p_values != 0] = (1 - p_values[p_values != 0]) ** 2
        goodData2 = f_statistic * p_values
        return goodData2

    # @lru_cache(10)
    # TODO: BEFORE SPLIT
    def onlySignData(
        self, ndata_train, ndata_test, goodData=None, goodData2=None,
    ):
        # coefs=None
        if self.signAll and self.signSolo:
            if ndata_train[:, [goodData != 0][0] + [goodData2 != 0][0]].shape[1] < 3:
                return 0.25
            ndata_train = ndata_train[:, [
                goodData != 0][0] + [goodData2 != 0][0]]
            ndata_test = ndata_test[:, [goodData != 0]
                                    [0] + [goodData2 != 0][0]]

        elif self.signAll:
            if ndata_train[:, np.where(goodData != 0)[0]].shape[1] < 3:
                return 0.25
            ndata_train = ndata_train[:, np.where(goodData != 0)[0]]
            ndata_test = ndata_test[:, np.where(goodData != 0)[0]]

        elif self.signSolo:
            if ndata_train[:, np.where(goodData2 != 0)[0]].shape[1] < 3:
                return 0.25
            ndata_train = ndata_train[:, np.where(goodData2 != 0)[0]]
            ndata_test = ndata_test[:, np.where(goodData2 != 0)[0]]

        return ndata_train, ndata_test

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
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        return correctamount / labels_test.shape[0]  # , coefs

    def testSuite(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,

        kernels=["linear", "rbf", "sigmoid"],
    ):
        # goodData,
        # coefs = np.zeros([1, data_train.shape[1]])  # * data_train.shape[2]

        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        # goodData2 = None Doing it earlier
        # if self.signSolo:
        #     goodData2 = self.anovaSolo(ndata_train, labels_train)
        # if self.onlySign:
        #     ndata_train, ndata_test = self.onlySignData(
        #         ndata_train=ndata_train,
        #         ndata_test=ndata_test,
        #         goodData=goodData,
        #         goodData2=goodData2,
        #         coefs=coefs,
        #     )

        allResults = []
        if self.quickTest:
            clist = [2.5]
        else:
            clist = np.linspace(0.5, 5, 5)
        # testing using different kernels, C and degrees.
        for kernel in kernels:
            if kernel == "linear":
                for C in [0.5]:

                    for degree in range(1, 2):
                        res = self.svmPipeline(
                            ndata_train,
                            ndata_test,
                            labels_train,
                            labels_test,
                            # goodData=goodData,
                            degree=degree,
                            kernel=kernel,
                            C=C,
                            # coefs=coefs,
                        )
                        if self.verbose:
                            print(
                                "Result for degree {}, kernel {}, C = {}: {}".format(
                                    degree, kernel, (C * 100 // 10) / 10, res
                                )
                            )
                        allResults.append([name, res, kernel, C])

            else:
                for C in clist:

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
                            C=C,
                        )
                        if self.verbose:
                            print(
                                "Result for gamma {}, kernel {}, C = {}: {}".format(
                                    gamma, kernel, (C * 100 // 10) / 10, res[0]
                                )
                            )
                        allResults.append([name, res, kernel, C])

        # coefs = np.reshape(coefs, [128, -1])
        return np.array(allResults, dtype=object)

    def testSuiteNN(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,

    ):
        # goodData,
        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        goodData2 = None
        if self.signSolo:
            goodData2 = self.anovaSolo(ndata_train, labels_train)
        if self.onlySign:
            ndata_train, ndata_test = self.onlySignData(
                ndata_train=ndata_train,
                ndata_test=ndata_test,

                goodData2=goodData2,
            )
        # goodData=goodData,
        allResults = []
        ndata_train = np.reshape(
            ndata_train, [ndata_train.shape[0], ndata_train.shape[1], 1]
        )
        ndata_test = np.reshape(
            ndata_test, [ndata_test.shape[0], ndata_test.shape[1], 1]
        )
        # Below here, do NN stuff instead
        print(ndata_train.shape)
        print(ndata_test.shape)
        print(ndata_train.shape[1])
        eeg_model = tf.keras.Sequential(
            [
                # layers.Flatten(input_shape=(1, ndata_train.shape[1], 1)),
                layers.Dense(
                    input_shape=(ndata_train.shape[1], 1),
                    activation="relu",
                    units=20,
                ),
                layers.Dense(
                    activation="relu",
                    units=20,
                ),
                layers.Flatten(),
                layers.Dense(units=1, activation="softmax"),
            ]
        )

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        eeg_model.build()
        eeg_model.summary()
        eeg_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        # Training NN
        outputs = eeg_model.fit(
            ndata_train,
            labels_train,
            validation_split=0.2,
            callbacks=[callback],
            epochs=100,
            verbose=True,
        )

        # Printing results
        print("Results")
        eeg_model.evaluate(ndata_test, labels_test)
        # result = eeg_model.predict(data_test_send)

        # Plotting training and validation results.
        val_loss = outputs.history["val_loss"]
        loss = outputs.history["loss"]
        val_acc = outputs.history["val_accuracy"]
        acc = outputs.history["accuracy"]
        allResults.append([name, acc, "juliakernel", 2])

        plt.plot(loss, "r", label="Training loss")
        plt.plot(val_loss, "b", label="Validation loss")
        # plt.title(
        #     "loss {} for reg val {} , dropout val {} ,layersize val {} and act {}".format(
        #         round(eval[0], 2), reg, drp, lz, act
        #     )
        # )
        plt.legend()
        plt.figure()
        plt.pause(0.1)

        val_acc = outputs.history["val_accuracy"]
        acc = outputs.history["accuracy"]
        plt.plot(acc, "r", label="Training acc")
        plt.plot(val_acc, "b", label="Validation acc")
        # plt.title(
        #     "acc {}  for reg val {} , dropout val {} , layersize val {} and act {}".format(
        #         round(eval[1], 2), reg, drp, lz, act
        #     )
        # )
        plt.legend()
        plt.figure()
        plt.pause(0.1)
        # Need to return a allResults lists with objects that are each [name, res[0], kernel, C] equivalent

        # return np.array(allResults, dtype = object)

    # # For ensemble svm model use
    # def testSuite2(self, data_train, data_test, labels_train, labels_test, name):

    #     allResults = []

    #     # testing using different kernels, C and degrees.
    #     coefs = np.zeros([1, data_train.shape[1] * data_train.shape[2]])
    #     if self.quickTest:
    #         clist = [2.5]
    #     else:
    #         clist = np.linspace(0.5, 5, 5)
    #     for kernel in ["linear"]:
    #         if kernel == "linear":
    #             for C in [0.5]:

    #                 for degree in range(1, 2):
    #                     res, coefs = self.svmEnsemble(
    #                         data_train,
    #                         data_test,
    #                         labels_train,
    #                         labels_test,
    #                         degree=degree,
    #                         kernel=kernel,
    #                         C=C,
    #                         coefs=coefs,
    #                     )
    #                     print(
    #                         "Result for degree {}, kernel {}, C = {}: {}".format(
    #                             degree, kernel, (C * 100 // 10) / 10, res
    #                         )
    #                     )
    #                     allResults.append([name, res, kernel, C])

    #         else:
    #             for C in clist:

    #                 for gamma in ["auto"]:

    #                     res = self.svmPipeline(
    #                         data_train,
    #                         data_test,
    #                         labels_train,
    #                         labels_test,
    #                         degree=degree,
    #                         kernel=kernel,
    #                         gamma=gamma,
    #                         C=C,
    #                     )
    #                     print(
    #                         "Result for gamma {}, kernel {}, C = {}: {}".format(
    #                             gamma, kernel, (C * 100 // 10) / 10, res[0]
    #                         )
    #                     )
    #                     allResults.append([name, res[0], kernel, C])

    #     coefs = np.reshape(coefs, [128, -1])
    #     return np.array(allResults, dtype=object)

    # def svmSVCReturn(
    #     self,
    #     data_train,
    #     data_test,
    #     labels_train,
    #     labels_test,
    #     kernel="linear",
    #     degree=3,
    #     gamma="auto",
    #     C=1,
    #     coefs=None,
    # ):

    #     # scaler = StandardScaler()
    #     # scaler = scaler.fit(np.reshape(data_train, [data_train.shape[0], -1]))
    #     # ndata_train = scaler.transform(
    #     #     np.reshape(data_train, [data_train.shape[0], -1])
    #     # )

    #     svc = SVC(  # anova_filter/#,
    #         gamma=gamma,
    #         kernel=kernel,
    #         degree=degree,
    #         verbose=False,
    #         C=C,
    #         cache_size=1800,
    #         probability=True,
    #     )
    #     # svc.fit(np.reshape(data_train, [data_train.shape[0], -1]), labels_train)
    #     # svc.fit(ndata_train, labels_train)
    #     # predictions = clf.predict(ndata_test)
    #     return svc

    # def svmEnsemble(
    #     self,
    #     data_train,
    #     data_test,
    #     labels_train,
    #     labels_test,
    #     kernel="linear",
    #     degree=3,
    #     gamma="auto",
    #     C=1,
    #     coefs=None,
    # ):
    #     estimatorList = []
    #     for kernel in ["linear", "rbf", "sigmoid"]:
    #         if kernel == "linear":
    #             for C in [0.5]:

    #                 for degree in range(1, 2):
    #                     estimatorList.append(
    #                         [
    #                             f"{kernel}{C}",
    #                             self.svmSVCReturn(
    #                                 data_train,
    #                                 data_test,
    #                                 labels_train,
    #                                 labels_test,
    #                                 degree=degree,
    #                                 kernel=kernel,
    #                                 C=C,
    #                                 coefs=coefs,
    #                             ),
    #                         ]
    #                     )

    #                     # print(
    #                     #     "Result for degree {}, kernel {}, C = {}: {}".format(
    #                     #         degree, kernel, (C * 100 // 10) / 10, res
    #                     #     )
    #                     # )
    #                     # allResults.append([name, res, kernel, C])

    #         else:
    #             for C in np.linspace(0.5, 5, 5):
    #                 for gamma in ["auto"]:
    #                     estimatorList.append(
    #                         [
    #                             f"{kernel}{C}",
    #                             self.svmSVCReturn(
    #                                 data_train,
    #                                 data_test,
    #                                 labels_train,
    #                                 labels_test,
    #                                 degree=degree,
    #                                 kernel=kernel,
    #                                 gamma=gamma,
    #                                 C=C,
    #                             ),
    #                         ]
    #                     )
    #                     # print(
    #                     #     "Result for gamma {}, kernel {}, C = {}: {}".format(
    #                     #         gamma, kernel, (C * 100 // 10) / 10, res[0]
    #                     #     )
    #                     # )
    #                     # allResults.append([name, res[0], kernel, C])

    #     # StandardScaler() done inside return SVC
    #     scaler = StandardScaler()
    #     scaler = scaler.fit(np.reshape(data_train, [data_train.shape[0], -1]))
    #     ndata_train = scaler.transform(
    #         np.reshape(data_train, [data_train.shape[0], -1])
    #     )
    #     ndata_test = scaler.transform(np.reshape(
    #         data_test, [data_test.shape[0], -1]))

    #     print(len(estimatorList))
    #     clf = VotingClassifier(estimatorList, n_jobs=-2, voting="soft")
    #     f_statistic, p_values = feature_selection.f_classif(
    #         ndata_train, labels_train)
    #     p_values[p_values > 0.1] = 0  # Use sklearn selectpercentile instead?

    #     goodData = f_statistic * p_values
    #     if ndata_train[:, np.where(goodData != 0)[0]].shape[1] < 3:
    #         return 0.25, coefs
    #     clf.fit(ndata_train[:, np.where(goodData != 0)[0]], labels_train)

    #     predictions = clf.predict(ndata_test[:, np.where(goodData != 0)[0]])

    #     # clf.fit(ndata_train, labels_train)

    #     # predictions = clf.predict(ndata_test)

    #     correct = np.zeros(labels_test.shape)
    #     correctamount = 0
    #     for nr, pred in enumerate(predictions, 0):
    #         if pred == labels_test[nr]:
    #             correct[nr] = 1
    #             correctamount += 1

    #     coefs = np.reshape(coefs, [128, -1])  # I donno why this is still here
    #     return correctamount / labels_test.shape[0], coefs
