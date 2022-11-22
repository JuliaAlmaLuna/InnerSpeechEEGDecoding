import dataLoader as dl
from copy import deepcopy as dp
from feature_extractionClean import featureEClass
import numpy as np


class baseLineCorrection(featureEClass):
    def __init__(
        self,
        subject,
        chunkAmount,  # Unsure if needed, here that is
        paradigmName="baseline",
        globalSignificance="0.5",  # Doesn't matter
        sampling_rate=256,
        featureFolder="WorkingBaselineFeaturesNew",
        chunk=False,
    ):
        print(
            "The feature Class created next will be a baseline STACK OLD Feature Class"
        )
        super().__init__(
            subject=subject,
            featureFolder=featureFolder,
            paradigmName=paradigmName,
            globalSignificance=globalSignificance,
            chunk=chunk,
            chunkAmount=chunkAmount,
            uniqueThresh=0.8,
            stftSplit=8
        )
        self.baseLineFeatures = []
        self.baseLineData = None
        self.avgBaselineFeatureList = None
        self.baseLineLabels = None
        self.sampling_rate = sampling_rate
        self.correctedFeaturesList = None
        self.chunk = chunk

    def loadBaselineData(self):

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

    def getBaselineFeatures(
        self,
        trialSampleAmount,
        featureList,
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
                :, :, x * trialSampleAmount: (x + 1) * trialSampleAmount
            ]
            # print(featureEClass.plotHeatMaps(self.data[0]))
            super().getFeatures(featureList=featureList, verbose=True)

            baselineFeatureListList.append(
                self.getFeatureList(),

            )

            # print(len(self.getFeatureList())) # 8
        print(len(baselineFeatureListList))
        justFeatureArraysList = []
        # iterates through amount of features creating empty lists
        for x in range(len(baselineFeatureListList[0])):
            justFeatureArraysList.append([])

        for baselineSplit in baselineFeatureListList:
            for featNr, feat in enumerate(baselineSplit):  # Feature
                justFeatureArraysList[featNr].append(feat[0])
        print(len(justFeatureArraysList))
        # print(np.array(justFeatureArraysList).shape)
        avgFEATURESlist = []
        # print("JULIANO")
        # print(justFeatureArraysList)
        # print("JULIANO")
        # print(self.getFeatureList())

        for featURE, unAvgFeature in zip(justFeatureArraysList, self.getFeatureList()):
            tempArray = np.asarray(featURE)
            # print(tempArray.shape)
            print(tempArray.shape)
            print("JULIA")
            avgFeature = np.mean(tempArray, axis=0)
            print(avgFeature.shape)
            # Here, if chunked And one of the features not CV, then avg all chunks as well
            if self.chunk:
                if "CV" not in unAvgFeature[1]:
                    avgFeature2 = np.reshape(
                        avgFeature,
                        [
                            avgFeature.shape[0],
                            avgFeature.shape[1],
                            self.chunkAmount,
                            -1,  # Now Session * channels * chunkAmount * onlySpecificDataPerchunk
                        ],
                    )
                    # Now session * channels * averagedSpecificDataPerChunk
                    avgFeature3 = np.mean(avgFeature2, axis=2)
                    avgFeature = np.tile(avgFeature3, reps=[
                                         1, 1, self.chunkAmount])
                    # np.repeat(
                    #     avgFeature3, repeats=self.chunkAmount, axis=0)
                    # np.concatenate(
                    #     [avgFeature3, avgFeature3, avgFeature3], axis=2
                    # )

            avgFEATURESlist.append(avgFeature)

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

    # Todo: Make it so that all are saved. And loaded based on names in featureList to be corrected.
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
            # print(bfeature.shape)

            # if self.chunk:
            #     featureName = f"{featureName}cn{self.chunkAmount}"
            # print(featureName)
            for ufeature, cfeature in zip(unCorrectedFeatureList, correctedFeatureList):
                # print(ufeature[0].shape)
                # print(cfeature[0].shape)
                sameDaySameLabel = 0
                samdDayDiffLabel = 0
                if ufeature[1] == featureName:
                    corrFeature = []
                    print(labelsAux)
                    # tempLabelsAux = np.roll(labelsAux, 1)
                    # print(np.sort(tempLabelsAux, axis=0))
                    # labelsAux[::, labelsAux[0,].argsort()[::-1]]
                    sameDaySameLabel = 0
                    samdDayDiffLabel = 0
                    for trial, labelAux in zip(ufeature[0], labelsAux):
                        session = labelAux[3]
                        # print(labelAux)
                        # print(labelAux)
                        # print(session)
                        if labelAux[1] == labelAux[3]:
                            sameDaySameLabel = sameDaySameLabel + 1
                        else:
                            samdDayDiffLabel = samdDayDiffLabel + 1
                        corrTrial = trial - bfeature[0][session - 1]

                        # featureEClass.plotHeatMaps(corrTrial)
                        # print(f"SUBJECT ABOVE {self.subject}")
                        # print(labelAux)
                        corrFeature.append(corrTrial)
                    # print(sameDaySameLabel)
                    # print(samdDayDiffLabel)
                    # print(samdDayDiffLabel / (samdDayDiffLabel + sameDaySameLabel))
                    corrFeature = np.array(corrFeature)

                    # featureEClass.plotHeatMaps(corrFeature[0])
                    # print(f"SUBJECT ABOVE {self.subject}")
                    # print(labelAux)
                    # print(cfeature[0][0][0][0])
                    # print("ada")
                    cfeature[0] = corrFeature

                    # if self.chunk:
                    #     cfeature[1] = f"{cfeature[1]}cn{self.chunkAmount}"

                    cfeature[1] = f"{cfeature[1]}_BC"
                    # print(cfeature[0][0][0][0])
                    print(paradigmName2)
                    print(self.paradigmName)
                    print(cfeature[1])
                    print(f"{cfeature[1]}_BC")
                    self.featureFolder = "SavedFeaturesNew"
                    self.paradigmName = f"{paradigmName2}"
                    self.saveFeatures(f"{cfeature[1]}", cfeature)
                    self.featureFolder = ("WorkingBaselineFeaturesNew",)
        self.correctedFeaturesList = correctedFeatureList

        return correctedFeatureList


if __name__ == "__main__":
    b = baseLineCorrection(subject=1, session=2)
    b.loadBaselineData()
    b.getBaselineFeatures()
    featureList = b.getFeatureList()
    print(len(featureList))
    print(len(featureList[0]))
    print(featureList[0][0].shape)
    baseLineCorrection.plotHeatMaps(featureList[0][0][1])
