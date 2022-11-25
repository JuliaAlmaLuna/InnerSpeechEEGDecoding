import dataLoader as dl
from copy import deepcopy as dp
from feature_extractionClean import featureEClass
import numpy as np


class baseLineCorrection(featureEClass):
    def __init__(
        self,
        subject,
        chunkAmount,  # Unsure if needed, here that is
        saveFolderName,
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
            saveFolderName=saveFolderName,
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
            "eeg",
            sampling_rate=self.sampling_rate,
            subject_nr=self.subject,
            t_start=4,
            t_end=4.5,
            verbose=True,
        )

        self.baseLineData = bdata  # [:, :128]
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
        self, unCorrectedFeatureList, baseLineFeatureList, paradigmName2, saveFolderName,
    ):  # Correct before creating Features of real data
        # or after, or both. Probably correct FFT, Welch, and Hilbert after creating them
        # And no correction using covariance for now. Create new fClasses for corrected?

        bfeatures = baseLineFeatureList
        correctedFeatureList = dp(unCorrectedFeatureList)
        for bfeature in bfeatures:
            featureName = bfeature[1]
            for ufeature, cfeature in zip(unCorrectedFeatureList, correctedFeatureList):
                if ufeature[1] == featureName:
                    corrFeature = []
                    bfeature[0] = np.roll(bfeature[0], shift=0, axis=0)
                    # tempval = 0
                    for trial, btrial in zip(ufeature[0], bfeature[0]):
                        corrTrial = trial - btrial

                        corrFeature.append(corrTrial)
                        # if tempval % 12 == 0:
                        #     featureEClass.plotHeatMaps(trial)
                        #     featureEClass.plotHeatMaps(btrial)
                        #     featureEClass.plotHeatMaps(corrTrial)
                        # tempval = tempval + 1

                    # featureEClass.plotHeatMaps(trial)
                    # featureEClass.plotHeatMaps(btrial)
                    # featureEClass.plotHeatMaps(corrTrial)
                    corrFeature = np.array(corrFeature)
                    # import time
                    # time.sleep(5)
                    cfeature[0] = corrFeature

                    cfeature[1] = f"{cfeature[1]}_BC"

                    print(paradigmName2)
                    print(self.paradigmName)
                    print(cfeature[1])
                    print(f"{cfeature[1]}_BC")
                    self.featureFolder = "SavedFeaturesNew"
                    self.paradigmName = f"{paradigmName2}"
                    oldFolderName = self.saveFolderName
                    self.saveFolderName = saveFolderName
                    self.saveFeatures(f"{cfeature[1]}", cfeature)
                    self.saveFolderName = oldFolderName
                    self.featureFolder = ("WorkingBaselineFeaturesNew",)
        self.correctedFeaturesList = correctedFeatureList
        correctedFeatureList = None
        if self.correctedFeaturesList is None:
            raise Exception("Nogood")
        return self.correctedFeaturesList


if __name__ == "__main__":
    b = baseLineCorrection(subject=1, session=2)
    b.loadBaselineData()
    b.getBaselineFeatures()
    featureList = b.getFeatureList()
    print(len(featureList))
    print(len(featureList[0]))
    print(featureList[0][0].shape)
    baseLineCorrection.plotHeatMaps(featureList[0][0][1])
