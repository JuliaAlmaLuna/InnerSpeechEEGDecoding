import dataLoader as dl
from copy import deepcopy as dp
from feature_extraction import featureEClass
import numpy as np


class baseLineCorrection(featureEClass):
    def __init__(
        self,
        subject,
        chunkAmount,
        session=2,  # Unsure if needed, here that is
        paradigmName="baseline",
        globalSignificance="0.5",  # Doesn't matter
        sampling_rate=256,
        featureFolder="SavedBaselineFeatures",
        chunk=False,
    ):
        print("The feature Class created next will be a baseline Feature Class")
        super().__init__(
            subject=subject,
            featureFolder=featureFolder,
            paradigmName=paradigmName,
            globalSignificance=globalSignificance,
            chunk=chunk,
            chunkAmount=chunkAmount,
        )
        self.session = session
        self.baseLineFeatures = []
        self.baseLineData = None
        self.avgBaselineFeatureList = None
        self.baseLineLabels = None
        self.sampling_rate = sampling_rate
        self.correctedFeaturesList = None
        self.chunk = chunk

    def loadBaselineData(self):

        # root_dir = "eeg-imagined-speech-nieto"
        # np.random.seed(23)

        # mne.set_log_level(verbose="warning")  # to avoid info at terminal
        # warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        # warnings.filterwarnings(action="ignore", category=FutureWarning)

        # # Sampling rate
        # fs = sampling_rate
        # # datatype = datatype2
        # # Subject number
        # N_S = subject_nr

        bdata, blabels = dl.load_data(
            "baseline",
            sampling_rate=self.sampling_rate,
            subject_nr=self.subject,
            t_start=0,
            t_end=15,
            verbose=True,
        )

        self.baseLineData = bdata[:, :128]
        self.baseLineLabels = blabels
        # self.data = bdata[:, :128]
        # self.labels = blabels

    def getBaselineFeatures(
        self,
        trialSampleAmount,
        featureList=[
            False,  # FFT
            False,  # Welch
            False,  # Hilbert
            False,  # Powerbands
            False,  # FFT frequency buckets
            False,  # FFT Covariance
            True,  # Welch Covariance
            True,  # Hilbert Covariance
            False,  # Covariance on smoothed Data
            False,  # Covariance on smoothed Data 2
            False,  # Correlate1d
            # More to be added
            # Add a feature that tells you if it is inner, pronounced or visualized
            # Make it shape(128,1) with maybe 10 ones at specific points if it is
            # each one of these
            # Maybe also make one similar with 10 for each subject
        ],
    ):
        if self.baseLineData is None:
            raise Exception("Data for baseline needs to be loaded first")

        # trialSampleAmount = round((t_max - t_min) * self.sampling_rate)
        baselineBatches = self.baseLineData.shape[2] // trialSampleAmount
        print(f"Baseline batches : {baselineBatches}")
        baselineFeatureListList = []
        for x in range(baselineBatches):
            self.paradigmName = f"split{x+1}"
            self.data = self.getBaselineData()[
                :, :, x * trialSampleAmount : (x + 1) * trialSampleAmount
            ]
            super().getFeatures(self.subject, featureList=featureList)
            baselineFeatureListList.append(self.getFeatureList())
            # print(len(self.getFeatureList())) # 8

        justFeatureArraysList = []
        # iterates through amount of features creating empty lists
        for x in range(len(baselineFeatureListList[0])):
            justFeatureArraysList.append([])

        for baselineSplit in baselineFeatureListList:
            for featNr, feat in enumerate(baselineSplit):  # Feature
                justFeatureArraysList[featNr].append(feat[0])

        avgFEATURESlist = []
        for featURE, unAvgFeature in zip(justFeatureArraysList, self.getFeatureList()):
            tempArray = np.asarray(featURE)
            print(tempArray.shape)
            avgFeature = np.mean(tempArray, axis=0)
            # Here, if chunked And one of the features not CV, then avg all chunks as well
            if self.chunk:
                if "CV" not in unAvgFeature[1]:
                    avgFeature2 = np.reshape(
                        avgFeature,
                        [
                            self.chunkAmount,
                            avgFeature.shape[0],
                            avgFeature.shape[1],
                            -1,
                        ],
                    )
                    avgFeature3 = np.mean(avgFeature2, axis=0)
                    avgFeature = np.concatenate(
                        [avgFeature3, avgFeature3, avgFeature3], axis=2
                    )

            avgFEATURESlist.append(avgFeature)
            print(avgFeature.shape)

        self.avgBaselineFeatureList = self.getFeatureList()
        for featURE, avgBaselineFeature in zip(
            avgFEATURESlist, self.avgBaselineFeatureList
        ):
            avgBaselineFeature[0] = featURE
            print(featURE.shape)

        # self.avgBaselineFeatureList = avgFEATURESlist

    def getBaselineData(self):
        tempData = dp(self.baseLineData)
        return tempData

    def getAvgBaselineFeatureList(self):
        tempData = dp(self.avgBaselineFeatureList)
        return tempData

    def baselineCorrect(
        self, unCorrectedFeatureList, labelsAux, paradigmName2
    ):  # Correct before creating Features of real data
        # or after, or both. Probably correct FFT, Welch, and Hilbert after creating them
        # And no correction using covariance for now. Create new fClasses for corrected?
        bfeatures = self.getAvgBaselineFeatureList()
        correctedFeatureList = dp(unCorrectedFeatureList)
        for bfeature in bfeatures:
            featureName = bfeature[1]
            # if self.chunk:
            #     featureName = f"{featureName}cn{self.chunkAmount}"
            print(featureName)
            for ufeature, cfeature in zip(unCorrectedFeatureList, correctedFeatureList):
                if ufeature[1] == featureName:
                    corrFeature = []
                    for trial, labelAux in zip(ufeature[0], labelsAux):
                        session = labelAux[3]
                        corrTrial = trial - bfeature[0][session - 1]
                        corrFeature.append(corrTrial)
                    corrFeature = np.array(corrFeature)
                    # print(cfeature[0][0][0][0])
                    # print("ada")
                    cfeature[0] = corrFeature

                    # if self.chunk:
                    #     cfeature[1] = f"{cfeature[1]}cn{self.chunkAmount}"

                    cfeature[1] = f"{cfeature[1]}BC"
                    # print(cfeature[0][0][0][0])
                    self.featureFolder = "SavedFeatures"
                    self.paradigmName = f"{paradigmName2}"
                    self.saveFeatures(f"{cfeature[1]}", cfeature)
                    self.featureFolder = ("SavedBaselineFeatures",)
        self.correctedFeaturesList = correctedFeatureList

        return correctedFeatureList

        # THIS MIGHT! MIGHT! WORK NOW

        # Here, split unCorrectedFeatures into days. And move day dimension before
        # Feature dimension. Trials into days
        # It comes as features * [feature, featureName] * trials * ....
        # Make it days * features * [feature, featureName] * trials*....

        # Create empty list correctedFeatures
        # for day in unCorrectedFeatures:
        #   for feature in bfeatures ( The ones that should be corrected )
        #       remove bfeature from feature with same name
        #       append corrected feature to list of corrected Features

        # return correctedFeatureList


if __name__ == "__main__":

    # nr_of_datasets = 1
    # specificSubject = 1
    # data, labels, labelsAux = dl.load_multiple_datasets(
    #     nr_of_datasets=nr_of_datasets,
    #     sampling_rate=256,
    #     t_min=2,
    #     t_max=3,
    #     specificSubject=specificSubject,
    #     twoDLabels=False,
    # )
    # print(labelsAux)
    # print(labelsAux[:, 1])  # Class
    # print(labelsAux[:, 2])  # Cond
    # print(labelsAux[:, 3])  # Session
    b = baseLineCorrection(subject=1, session=2)
    b.loadBaselineData()
    b.getBaselineFeatures()
    featureList = b.getFeatureList()
    print(len(featureList))
    print(len(featureList[0]))
    print(featureList[0][0].shape)
    baseLineCorrection.plotHeatMaps(featureList[0][0][1])

    # avgFEATURESlist = list of average features. Still containing session as first.
    # newbaselineFeatureListList = []

    # baselineFeatureListList

    # for featureNr in range(len(baselineFeatureListList[0])):
    #     specFeatureArrayList = []
    #     # This one seems to iterate over Features
    #     for baselineFeatureTuple in zip(*baselineFeatureListList):
    #         # Feature ( or in this case, tuple of every baselineDatas features) \
    # baselineFeatureTuple = tuple of Features * [Feature Name] * trials ....
    #         # print(baselineFeatureTuple[0])
    #         # actually not a tuple
    #         print(baselineFeatureTuple[1][1])

    #         # (12,2)  # Each baselinesplit and feature/name
    #         print(np.asarray(baselineFeatureTuple).T.shape)
    #         print(np.asarray(np.asarray(baselineFeatureTuple).T[0]).shape)
    #         specFeatureArrayList.append(np.asarray(
    #             baselineFeatureTuple).T[0])  # 2, 12
    #         # (12,2)
    #         # print(testJulia[2].shape) # The two here is for the second batch it seems, reasonable
    #         # print(baselineFeatureTuple[0])
    #         tupleJustFeatures = baselineFeatureTuple[featureNr][0]
    #         # s pecFeatureArrayList
    #         print(featureNr)
    #         print(tupleJustFeatures.shape)
    #         justFeatureArrayArray[featureNr]
    #         justFeatureArray = np.asarray(tupleJustFeatures)
    #         justFeatureArraysList.append(justFeatureArray)
    #         # print(justFeatureArray.shape)

    #         # justFeatureArraysList.append(np.asarray(baselineFeatureTuple[featureNr][0]))
    # print(specFeatureArrayList[0].shape)

    # tmp = specFeatureArrayList
    # newJuliaArray = np.zeros([tmp[0].shape[0], np.asarray(tmp[0][0]).shape[0], np.asarray(
    #     tmp[0][0]).shape[1], np.asarray(tmp[0][0]).shape[2]])  # Last shape is not the same!
    # for i, featu in enumerate(specFeatureArrayList):
    #     newJuliaArray[i] = featu[0]
    #     print(featu[0])
    # print(newJuliaArray.shape)
    # # print(np.asarray_chkfinite(featu).shape)
    # # print(np.asarray(featu[0]).shape)

    # print(len(justFeatureArraysList))
    # baselineFeatureArray = np.asarray(justFeatureArraysList)
    # print(baselineFeatureArray.shape)

    # #     np.asarray(baselineFeatureTuple[0])
    # #     print(baselineFeatureArray[0].shape)
    # #     # Shape should be nrOfBaselines * trials * .....
    # #     avgBaselineFeature = np.average(baselineFeatureArray)
    # #     avgBaselineFeatureList[featureNr][0] = avgBaselineFeature

    # # self.avgBaselineFeatureList = avgBaselineFeatureList
