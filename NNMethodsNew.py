import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers
# import tensorflow_hub as hub  # type:ignore
# from sklearn.feature_selection import f_classif  # SelectKBest ,


class NNMetsNew:
    def __init__(
        self,
        signAll=False,
        signSolo=True,
        tol=0.001,
        significanceThreshold=0.1,
        verbose=True,
        quickTest=False
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
    def onlySignData(
        self, ndata_train, ndata_test, goodData=None, goodData2=None, coefs=None
    ):

        if self.signAll and self.signSolo:
            if ndata_train[:, [goodData != 0][0] + [goodData2 != 0][0]].shape[1] < 3:
                return 0.25, coefs
            ndata_train = ndata_train[:, [
                goodData != 0][0] + [goodData2 != 0][0]]
            ndata_test = ndata_test[:, [goodData != 0]
                                    [0] + [goodData2 != 0][0]]

        elif self.signAll:
            if ndata_train[:, np.where(goodData != 0)[0]].shape[1] < 3:
                return 0.25, coefs
            ndata_train = ndata_train[:, np.where(goodData != 0)[0]]
            ndata_test = ndata_test[:, np.where(goodData != 0)[0]]

        elif self.signSolo:
            if ndata_train[:, np.where(goodData2 != 0)[0]].shape[1] < 3:
                return 0.25, coefs
            ndata_train = ndata_train[:, np.where(goodData2 != 0)[0]]
            ndata_test = ndata_test[:, np.where(goodData2 != 0)[0]]

        return ndata_train, ndata_test

    def makeModel(self,
                  ndata
                  ):
        z = 20
        # x = 1
        # y = 1
        eeg_model = tf.keras.Sequential(
            [
                # input_shape=(
                #     ndata.shape[1], 1)
                layers.Flatten(input_shape=(
                    ndata.shape[1], 1)),
                layers.Dense(activation="relu", units=1500),
                layers.Dropout(0.1),
                layers.Dense(activation="relu", units=400,
                             kernel_regularizer=regularizers.L1L2(
                                 l1=z * 1e-5, l2=z * 1e-4),
                             bias_regularizer=regularizers.L2(z * 1e-4),
                             activity_regularizer=regularizers.L2(z * 1e-5),),
                layers.Dropout(0.1),
                layers.Dense(activation="relu", units=100,
                             kernel_regularizer=regularizers.L1L2(
                                 l1=z * 1e-5, l2=z * 1e-4),
                             bias_regularizer=regularizers.L2(z * 1e-4),
                             activity_regularizer=regularizers.L2(z * 1e-5)),
                layers.Flatten(),
                layers.Dropout(0.1),
                layers.Dense(activation="relu", units=10,
                             kernel_regularizer=regularizers.L1L2(
                                 l1=z * 1e-5, l2=z * 1e-4),
                             bias_regularizer=regularizers.L2(z * 1e-4),
                             activity_regularizer=regularizers.L2(z * 1e-5)),
                layers.Dropout(0.1),
                layers.Flatten(),
                layers.Dense(units=4, activation="softmax"),  # labels shape
            ]
        )
        return eeg_model

    def testSuiteNN(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        name,

    ):

        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        # goodData2 = None
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

        ndata_trainSend = np.reshape(
            ndata_train, [ndata_train.shape[0], ndata_train.shape[1], 1])
        ndata_testSend = np.reshape(
            ndata_test, [ndata_test.shape[0], ndata_test.shape[1], 1])
        # Reshaping data to fit with neural net.

        eeg_model = self.makeModel(ndata=ndata_trainSend)
        eeg_model.build()
        eeg_model.summary()

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        )

        eeg_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        print(ndata_trainSend.shape)

        # Training NN
        outputs = eeg_model.fit(
            ndata_trainSend,
            labels_train,
            validation_split=0.2,
            callbacks=[callback],
            epochs=1000,
            verbose=False,
        )

        # Printing results
        print("Results")
        eval = eeg_model.evaluate(ndata_testSend, labels_test)

        # Plotting training and validation results.
        val_loss = outputs.history["val_loss"]
        loss = outputs.history["loss"]

        # if saveBest is True:
        #     if (np.amin(val_loss) + eval[0]) < best_loss:
        #         best_loss = np.amin(val_loss) + eval[0]
        #         eeg_model.save("saved_model/best_model{}".format(specificSubject))

        plt.plot(loss, "r", label="Training loss")
        plt.plot(val_loss, "b", label="Validation loss")
        plt.title(
            "loss {} for reg val {} , dropout val {} ,layersize val {} and act {}".format(
                round(eval[0], 2), 1, 1, 1, 1
            )
        )
        plt.legend()
        plt.figure()
        plt.pause(0.1)

        val_acc = outputs.history["val_accuracy"]
        acc = outputs.history["accuracy"]
        plt.plot(acc, "r", label="Training acc")
        plt.plot(val_acc, "b", label="Validation acc")
        plt.title(
            "acc {}  for reg val {} , dropout val {} , layersize val {} and act {}".format(
                round(eval[1], 2), 1, 1, 1, 1
            )
        )
        plt.legend()
        plt.figure()
        plt.pause(0.1)
        allResults = [val_acc, acc, name]
        return np.array(allResults, dtype=object)
