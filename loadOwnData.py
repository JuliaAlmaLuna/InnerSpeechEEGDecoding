
# from dask import dataframe as df
import pandas as pd
import glob
import numpy as np


def loadOwnData(words, dataPath="SadAngryHappyDisgusted/", t_start=0, t_end=4):
    import os
    # Get path of marker and ExG
    pathExG = glob.glob(f"{os.getcwd()}/MyDataset/{dataPath}*_ExG*")
    pathMarkers = glob.glob(f"{os.getcwd()}/MyDataset/{dataPath}*_Marker*")

    print(f"{os.getcwd()}/{dataPath}*_ExG*")
    print(words.values())
    # Create arrays and dataframe from them

    exgData = pd.DataFrame(pd.read_csv(pathExG[0]))
    markerData = pd.DataFrame(pd.read_csv(pathMarkers[0]))

    for pE, pM, in zip(pathExG[1:], pathMarkers[1:]):

        exgData = pd.concat([exgData, pd.DataFrame(pd.read_csv(pE))])
        markerData = pd.concat([markerData, pd.DataFrame(pd.read_csv(pM))])

    return exgData, markerData


def find_nearest(array, value, amount=10):
    value = value
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def createEpochs2(words, eegData, markerData, sampling_rate=250, lowestT=2500):
    allTrials = []
    allTrialsLabels = []
    numpyEEG = eegData.to_numpy()
    numpyEEG = preProcessDataAverage(numpyEEG)

    for marker in words.values():

        tStart = markerData.loc[markerData["Code"]
                                == f"sw_{marker}", "TimeStamp"]
        # Check for bad trials
        badTrialT = markerData.loc[markerData["Code"]
                                   == f"sw_{99}", "TimeStamp"]
        tStart = tStart - 6
        tEnd = tStart + 10
        tStart = np.array(tStart)
        tEnd = np.array(tEnd)

        trials = []
        labels = []
        for start, end in zip(tStart, tEnd):
            for t in badTrialT:
                if t > (start - 6) and t < (end - 6):
                    print("SKIPPED BAD TRIAL. CHECK THAT SHAPE IS OKAY!")
                    continue
            startVal, startInd = find_nearest(numpyEEG[:, 0], start)
            endtVal, endInd = find_nearest(numpyEEG[:, 0], end)
            trialData = np.array(
                numpyEEG[startInd:endInd, 1:][:sampling_rate * 10, :])
            trialData = np.swapaxes(trialData, 0, 1)
            trials.append(trialData)
            # print(trialData.shape)

            labels.append(marker)
        allTrials.extend(trials)
        allTrialsLabels.extend(labels)

    return allTrials, allTrialsLabels


def add_ref(data_in):

    # Create empty array to store new data
    data_out = np.empty((data_in.shape[0] + 1, data_in.shape[1]))

    # Add reference channel
    data_out[-1, :] = np.mean(data_in[1:], axis=0)

    # Add the rest of the data
    data_out[:-1, :] = data_in

    return data_out


def preProcessData(eegArray, bpass=[1, 100], notchFreq=50):
    from scipy import signal as sg
    fs = 250
    fo = 50
    q = 20
    # Makes channels second dim. First channel is timestep
    eegArray = np.swapaxes(eegArray, 0, 1)
    # eegArray = add_ref(eegArray)

    # Baselining with other ear.

    # Here, do (eegArray[-1, :] / 2 ) instead
    eegArray[1:- 1, :] = eegArray[1:- 1, :] - eegArray[-1, :]

    newChannel1 = eegArray[1, :] - eegArray[2, :]  # diff between temples
    # Replacing other ear channel with diff channel
    eegArray[- 1, :] = newChannel1

    eegArray2 = np.copy(eegArray)
    # print(eegArray.shape)
    # Filter the data with notch and a bandpass
    for ci, chan in enumerate(eegArray[1:], 1):
        # print(ci)
        sos = sg.butter(N=1, Wn=[1, 80],
                        btype="bandpass", fs=250, output="sos")
        c, d = sg.iirnotch(fo, q, fs)

        ctree = sg.filtfilt(c, d, chan)
        ctree2 = sg.sosfilt(sos, ctree)

        eegArray2[ci, :] = ctree2

    eegArray2 = np.swapaxes(eegArray2, 0, 1)
    eegArray = None
    return eegArray2


def average_rereferencing(EEG_data):
    num_channels = EEG_data.shape[0]
    avg_ref = np.mean(EEG_data, axis=0)
    for i in range(num_channels):
        EEG_data[i, :] -= avg_ref
    return EEG_data


def preProcessDataAverage(eegArray, bpass=[1, 100], notchFreq=50):
    from scipy import signal as sg
    fs = 250
    fo = 50
    q = 20
    # Makes channels second dim. First channel is timestep
    eegArray = np.swapaxes(eegArray, 0, 1)
    # eegArray = add_ref(eegArray)
    # Baselining with other ear.
    eegArray[1:- 1, :] = eegArray[1:- 1, :] - (eegArray[-1, :] / 2)
    # 0 channel is time. 7 is other ear
    avg_ref = np.mean(eegArray[1:-1, :], axis=0)
    for i in range(1, 7):
        eegArray[i, :] -= avg_ref
    newChannel1 = eegArray[1, :] - eegArray[2, :]  # diff between temples
    # Replacing other ear channel with diff channel
    eegArray[- 1, :] = newChannel1

    eegArray2 = np.copy(eegArray)
    # print(eegArray.shape)
    # Filter the data with notch and a bandpass
    for ci, chan in enumerate(eegArray[1:], 1):
        # print(ci)
        sos = sg.butter(N=1, Wn=[1, 80],
                        btype="bandpass", fs=250, output="sos")
        c, d = sg.iirnotch(fo, q, fs)

        ctree = sg.filtfilt(c, d, chan)
        ctree2 = sg.sosfilt(sos, ctree)

        eegArray2[ci, :] = ctree2

    eegArray2 = np.swapaxes(eegArray2, 0, 1)
    eegArray = None
    return eegArray2

# When using the upDownLeftRight dataset


def preProcessData2(eegArray, bpass=[1, 100], notchFreq=50):
    from scipy import signal as sg
    fs = 250
    fo = 50
    q = 20
    # Makes channels second dim. First channel is timestep
    eegArray = np.swapaxes(eegArray, 0, 1)
    # eegArray = add_ref(eegArray)
    # Baselining with other ear.
    # eegArray[1:- 1, :] = eegArray[1:- 1, :] - eegArray[-1, :]

    avg_ref = np.mean(eegArray[1:, :], axis=0)  # Using avg ref
    for i in range(1, 8):
        eegArray[i, :] -= avg_ref

    # diff between temples
    newChannel1 = eegArray[1, :] - eegArray[2, :]
    addedChannel = np.empty((eegArray.shape[0] + 1, eegArray.shape[1]))
    # Add diffChannel
    addedChannel[-1, :] = newChannel1
    # Add the rest of the data
    addedChannel[:-1, :] = eegArray
    eegArray = addedChannel

    # diff between forehead and two temples
    newChannel2 = eegArray[3, :] - \
        ((eegArray[1, :] / 2) + (eegArray[2, :] / 2))
    addedChannel2 = np.empty((eegArray.shape[0] + 1, eegArray.shape[1]))
    # Add diffChannel
    addedChannel2[-1, :] = newChannel2
    # Add the rest of the data
    addedChannel2[:-1, :] = eegArray
    eegArray = addedChannel2

    eegArray2 = np.copy(eegArray)
    # print(eegArray.shape)
    # Filter the data with notch and a bandpass
    for ci, chan in enumerate(eegArray[1:], 1):
        # print(ci)
        sos = sg.butter(N=1, Wn=[1, 80],
                        btype="bandpass", fs=250, output="sos")
        c, d = sg.iirnotch(fo, q, fs)

        ctree = sg.filtfilt(c, d, chan)
        ctree2 = sg.sosfilt(sos, ctree)

        eegArray2[ci, :] = ctree2

    eegArray2 = np.swapaxes(eegArray2, 0, 1)
    eegArray = None
    return eegArray2
