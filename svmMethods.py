import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
        scaler = StandardScaler()
        scaler = scaler.fit(data_train)

        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)

        allResults = []
        if self.quickTest:
            clist = [2.5]
        else:
            clist = [0.1, 0.5, 1.2, 2.5, 5, 10]
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
        allResults.append([name, res, kernel, c])

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

    def testSuiteMLP(
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

        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(
            hidden_layer_sizes=(20, 12, 6, 3),
            solver="lbfgs",
            activation="relu",
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(ndata_train, labels_train)
        predictions = mlp.predict(ndata_test)

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


""
