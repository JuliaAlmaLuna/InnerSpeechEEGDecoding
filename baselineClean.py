import dataLoader as dl
from copy import deepcopy as dp
from feature_extraction import featureEClass
import numpy as np


class baseLineCorrection(featureEClass):
    def __init__(
        self,
        subject,
        chunkAmount,
        session,  # Unsure if needed, here that is
        paradigmName="baseline",
        globalSignificance="0.5",  # Doesn't matter
        sampling_rate=256,
        featureFolder="WorkingBaselineFeatures",
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
                            avgFeature.shape[0],
                            avgFeature.shape[1],
                            self.chunkAmount,
                            -1,  # Now Session * channels * chunkAmount * onlySpecificDataPerchunk
                        ],
                    )
                    # Now session * channels * averagedSpecificDataPerChunk
                    avgFeature3 = np.mean(avgFeature2, axis=2)
                    avgFeature = np.tile(avgFeature3, reps=[1, 1, self.chunkAmount])
                    # np.repeat(
                    #     avgFeature3, repeats=self.chunkAmount, axis=0)
                    # np.concatenate(
                    #     [avgFeature3, avgFeature3, avgFeature3], axis=2
                    # )
            print("Heyyo")
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
                    print(paradigmName2)
                    print(self.paradigmName)
                    print(cfeature[1])
                    print(f"{cfeature[1]}BC")
                    self.featureFolder = "SavedFeatures"
                    self.paradigmName = f"{paradigmName2}"
                    self.saveFeatures(f"{cfeature[1]}", cfeature)
                    self.featureFolder = ("WorkingBaselineFeatures",)
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
