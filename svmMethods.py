import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
# import cv2
from tqdm import tqdm
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

        # Shape of trials and features, should be
        ndata_train = scaler.transform(data_train)
        ndata_test = scaler.transform(data_test)
        # ndata_train = np.reshape(ndata_train, [ndata_train.shape[0], 1, 1, -1])
        # ndata_test = np.reshape(ndata_test, [ndata_test.shape[0], 1, 1, -1])

        device = "cuda"
        # from torch.utils.data import Dataset
        torch_train = []
        for trialData, trialLabel in zip(ndata_train, labels_train):
            torch_trainData = torch.FloatTensor(trialData)
            t = torch.LongTensor(1)
            t[0] = int(trialLabel)
            torch_train.append([torch_trainData, t])
        print(labels_train)
        torch_test = []
        for trialData, trialLabel in zip(ndata_test, labels_test):
            torch_testData = torch.FloatTensor(trialData)
            t = torch.LongTensor(1)
            t[0] = int(trialLabel)
            torch_test.append([torch_testData, t])

        print(f"size of training data {len(torch_train)}")
        print(f"size of testing data {len(torch_test)}")
        print(torch_train[0][0].shape)
        print(torch_train[0][1].shape)
        print(torch_train[0][1])
        net = Net(featureSize=ndata_train.shape[-1])
        net.to(device)
        self.train_model(net=net, epochs=100, train_data=torch_train, device=device,
                         batchSize=10, trainSize=len(torch_train), featureSize=ndata_train.shape[-1])

        acc = self.test_model(
            net=net, device=device, test_data=torch_test, featureSize=ndata_train.shape[-1])
        print(acc)
        allResults = []
        allResults.append([name, acc, "MLP", 2.5])
        allResults.append([name, acc, "MLP", 2.5])
        return np.array(allResults, dtype=object)

    def train_model(self, net, train_data, device, epochs, batchSize=10, trainSize=610, featureSize=50 * 50):
        optimizer = optim.Adam(
            net.parameters(), lr=0.0001, weight_decay=0.00001)
        loss_function = nn.CrossEntropyLoss()

        for epoch in tqdm(range(epochs)):
            for i in (range(0, trainSize, batchSize)):
                if batchSize + i > trainSize:
                    continue
                batch = train_data[i:i + batchSize]
                batch_x = torch.cuda.FloatTensor(batchSize, 1, featureSize)
                batch_y = torch.cuda.LongTensor(batchSize, 1)

                for k in range(batchSize):
                    batch_x[k] = batch[k][0]
                    batch_y[k] = batch[k][1]
                batch_x.to(device)
                batch_y.to(device)
                net.zero_grad()
                outputs = net(batch_x.view(-1, 1, featureSize))
                batch_y = batch_y.view(batchSize)
                loss = F.nll_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"epoch : {epoch}  loss : {loss}")

    def test_model(self, net, device, test_data, featureSize=50 * 50):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_data):
                x = torch.FloatTensor(data[0])
                y = torch.LongTensor(data[1])

                x = x.view(-1, 1, featureSize)
                x = x.to(device)
                output = net(x)
                output = output.view(2)
                if (max(output[0], output[1]) == output[0]):
                    index = 0
                else:
                    index = 1
                if index == y[0]:
                    correct += 1
                total += 1
            return round(correct / total, 5)


class Net(nn.Module, ):
    def __init__(self, featureSize=50 * 50):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 32, 2)
        # self.conv2 = nn.Conv2d(32, 64, 2)
        # self.conv3 = nn.Conv2d(64, 128, 2)
        # nn.Conv2d(1,32,5,)
        # nn.Conv1d(1,32,1,1)
        # nn.Conv1d
        self.lin1 = nn.LazyLinear(1028)
        self.lin2 = nn.LazyLinear(512)
        self.lin3 = nn.LazyLinear(256)
        # self.conv1 = nn.Conv1d(1, 32, 5, 2)
        self.conv2 = nn.Conv1d(32, 64, 5, 2)
        self.conv3 = nn.Conv1d(64, 128, 5, 2)
        x = torch.rand(1, featureSize).view(-1, 1, featureSize)
        self.linear_in = None
        self.convs(x)

        self.fc1 = nn.Linear(self.linear_in, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        if self.linear_in is None:
            self.linear_in = x[0].shape[0] * x[0].shape[1]  # * x[0].shape[2]
        else:
            return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.linear_in)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
