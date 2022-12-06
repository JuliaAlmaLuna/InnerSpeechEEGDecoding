
# from dask import dataframe as df
import pandas as pd
import glob
import numpy as np


# Separate EEG data into epochs using markers.
def createEpochs(eeg, markers, words, t_start=0, t_end=4):
    epochs = []
    labels = []
    eStart = 0
    eEnd = 0
    curMark = 0
    samplingRate = 250
    print(words)
    for marker in markers:

        if marker[1] == 7 or marker[1] == 8:
            curMark = marker[1]
            eStart = round(np.where(eeg[:, 0] > marker[0])[
                           0][0] + t_start * samplingRate)
            eEnd = round(eStart + ((t_end - t_start) * samplingRate))
        # Untill next marker
        # elif eStart != 0:
        #     eEnd = np.where(eeg[:,0]>marker[0])[0][0]

        if eStart != 0 and eEnd != 0:

            if eEnd > len(eeg):
                print("Epoch not finished")
            else:
                labels.append(int(curMark))
                epochs.append(np.array(eeg[eStart:eEnd, 1:], dtype=float))
            eStart = 0
            eEnd = 0

    nlabel = np.array(labels)
    narray = np.array(epochs, dtype=float)
    print(f"labels shape: {nlabel.shape}")
    print(f"epochs shape: {nlabel.shape}")
    return narray, nlabel


def average_rereferencing(EEG_data):
    num_channels = EEG_data.shape[0]
    avg_ref = np.mean(EEG_data, axis=0)
    for i in range(num_channels):
        EEG_data[i, :] -= avg_ref
    return EEG_data


def add_ref(data_in):

    # Create empty array to store new data
    data_out = np.empty((data_in.shape[0] + 1, data_in.shape[1]))

    # Add reference channel
    data_out[0, :] = np.mean(data_in, axis=0)

    # Add the rest of the data
    data_out[1:, :] = data_in

    return data_out


def loadOwnData(words, dataPath="SadAngryHappyDisgusted/", t_start=0, t_end=4):
    import os
    # Get path of marker and ExG
    pathExG = glob.glob(f"{os.getcwd()}/MyDataset/{dataPath}*_ExG*")
    pathMarkers = glob.glob(f"{os.getcwd()}/MyDataset/{dataPath}*_Marker*")
    # print(pathExG)
    print(f"{os.getcwd()}/{dataPath}*_ExG*")
    print(words.values())
    # Create arrays and dataframe from them

    exgData = pd.DataFrame(pd.read_csv(pathExG[0]))  # .to_numpy()
    # print(exgData.shape)
    # exgData = np.swapaxes(exgData, 0, 1)
    # exgData[1:, :] = preProcessData(exgData[1:, :])
    # print(exgData)
    # exgData = pd.DataFrame(exgData,)
    markerData = pd.DataFrame(pd.read_csv(pathMarkers[0]))
    for pE, pM, in zip(pathExG[1:], pathMarkers[1:]):
        # print(type(exgData))
        exgData = pd.concat([exgData, pd.DataFrame(pd.read_csv(pE))])
        markerData = pd.concat([markerData, pd.DataFrame(pd.read_csv(pM))])

    return exgData, markerData


def find_nearest(array, value, amount=10):
    # array = np.asarray(array)
    # print(array)
    value = value
    # print(np.abs(array - value))
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return array[idx], idx


def createEpochs2(words, eegData, markerData, sampling_rate=250, lowestT=2500):
    allTrials = []
    allTrialsLabels = []
    numpyEEG = eegData.to_numpy()
    # numpyEEG = np.swapaxes(numpyEEG, 0, 1)
    numpyEEG = preProcessData(numpyEEG)

    for marker in words.values():
        # print(markerData)
        # markerData = markerData.loc[markerData["TimeStamp"]
        #                             > lowestT]
        tStart = markerData.loc[markerData["Code"]
                                == f"sw_{marker}", "TimeStamp"]
        tStart = tStart - 6
        tEnd = tStart + 10
        tStart = np.array(tStart)
        tEnd = np.array(tEnd)

        # print(eegData)
        # print(eegData.to_numpy())
        print(eegData.to_numpy()[:, 0])

        trials = []
        labels = []
        for start, end in zip(tStart, tEnd):
            # print("looking")
            # print(start)
            # print(end)
            # print(numpyEEG[:, 0])
            # print(f"Found: {find_nearest(numpyEEG[:, 0], start)}")
            startVal, startInd = find_nearest(numpyEEG[:, 0], start)
            endtVal, endInd = find_nearest(numpyEEG[:, 0], end)
            trialData = np.array(
                numpyEEG[startInd:endInd, 1:][:sampling_rate * 10, :])
            trialData = np.swapaxes(trialData, 0, 1)
            trials.append(trialData)
            print(trialData.shape)

            labels.append(marker)
        allTrials.extend(trials)
        allTrialsLabels.extend(labels)

        print(marker)
        print(tStart)
        print(tEnd)
        print(tEnd - tStart)
        print(tStart.shape)
        print(type(tStart))
    return allTrials, allTrialsLabels
    # pass


def preProcessData(eegArray, bpass=[1, 100], notchFreq=50):
    from scipy import signal as sg
    fs = 250
    fo = 50
    q = 20
    eegArray = np.swapaxes(eegArray, 0, 1)  # Makes channels second dim.
    # Baselining with other ear.

    eegArray[1:- 1, :] = eegArray[1:- 1, :] - eegArray[-1, :]
    # eegArray = eegArray[:- 1, :]
    newChannel1 = eegArray[1, :] - eegArray[2, :]  # diff between temples
    eegArray[- 1, :] = newChannel1
    # eegArray = np.rollaxis(eegArray, 0, 1)
    # eegArray = np.swapaxes(eegArray, 0, 1)
    eegArray2 = np.copy(eegArray)
    print(eegArray.shape)
    print(eegArray)
    # Do a notch filter on data
    for ci, chan in enumerate(eegArray[1:], 1):
        print(ci)
        sos = sg.butter(N=1, Wn=[1, 80],
                        btype="bandpass", fs=250, output="sos")

        # a,b = sg.iirfilter(N=1, Wn=[1,30], fs=fs)
        c, d = sg.iirnotch(fo, q, fs)
        # ctwo = sg.lfilter(a,b, chan)
        # ctree = sg.lfilter(c,d,chan)
        ctree = sg.filtfilt(c, d, chan)
        ctree2 = sg.sosfilt(sos, ctree)
        # print(ctree2.shape)
        # print(ctree.shape)

        eegArray2[ci, :] = ctree2

    eegArray2 = np.swapaxes(eegArray2, 0, 1)
    return eegArray2
