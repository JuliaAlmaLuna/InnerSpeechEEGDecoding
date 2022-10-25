import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# from sklearn.feature_selection import f_classif  # SelectKBest ,


class SvmMets:
    def __init__(self):
        print("new SvmMets")

        """
        This class handles SVM pipeline testing.
        Right now it is very janky!
        """

    def svmPipeline(
        self,
        data_train,
        data_test,
        labels_train,
        labels_test,
        kernel="linear",
        degree=3,
        gamma="auto",
        C=1,
        coefs=None,
    ):
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
        if coefs is None:
            coefs = np.zeros([1, data_train.shape[1] * data_train.shape[2]])
        # anova_filter = SelectKBest(f_classif, k=10)

        clf = make_pipeline(
            StandardScaler(),
            SVC(  # anova_filter/#,
                gamma=gamma,
                kernel=kernel,
                degree=degree,
                verbose=False,
                C=C,
                cache_size=1800,
            ),
        )

        clf.fit(np.reshape(data_train, [data_train.shape[0], -1]), labels_train)
        predictions = clf.predict(np.reshape(data_test, [data_test.shape[0], -1]))

        # if selectKBest is used
        if kernel == "linear":
            # print(classification_report(labels_test, predictions))
            coefs = coefs + clf[:-1].inverse_transform(clf[-1].coef_)
        # print(clf[:-1].inverse_transform(clf[-1].coef_).shape)

        correct = np.zeros(labels_test.shape)
        correctamount = 0
        for nr, pred in enumerate(predictions, 0):
            if pred == labels_test[nr]:
                correct[nr] = 1
                correctamount += 1

        return correctamount / labels_test.shape[0], coefs

    def testSuite(self, data_train, data_test, labels_train, labels_test, name):

        allResults = []

        # testing using different kernels, C and degrees.
        coefs = np.zeros([1, data_train.shape[1] * data_train.shape[2]])

        for kernel in ["linear", "rbf", "sigmoid"]:
            if kernel == "linear":
                for C in [0.5]:

                    for degree in range(1, 2):
                        res, coefs = self.svmPipeline(
                            data_train,
                            data_test,
                            labels_train,
                            labels_test,
                            degree=degree,
                            kernel=kernel,
                            C=C,
                            coefs=coefs,
                        )
                        print(
                            "Result for degree {}, kernel {}, C = {}: {}".format(
                                degree, kernel, (C * 100 // 10) / 10, res
                            )
                        )
                        allResults.append([name, res, kernel, C])

            else:
                for C in np.linspace(0.5, 5, 5):

                    for gamma in ["auto"]:

                        res = self.svmPipeline(
                            data_train,
                            data_test,
                            labels_train,
                            labels_test,
                            degree=degree,
                            kernel=kernel,
                            gamma=gamma,
                            C=C,
                        )
                        print(
                            "Result for gamma {}, kernel {}, C = {}: {}".format(
                                gamma, kernel, (C * 100 // 10) / 10, res[0]
                            )
                        )
                        allResults.append([name, res[0], kernel, C])

        coefs = np.reshape(coefs, [128, -1])
        return np.array(allResults, dtype=object)
