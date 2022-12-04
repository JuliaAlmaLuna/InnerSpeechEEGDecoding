
# from dask import dataframe as df
from pandas import DataFrame as df
import pandas as pd
import glob
import numpy as np

import re


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


def loadOwnData(words, dataPath="SadAngryHappyDisgusted/", t_start=0, t_end=4):
    import os
    # Get path of marker and ExG
    pathExG = glob.glob(f"{os.getcwd()}/MyDataset/{dataPath}*_ExG*")
    pathMarkers = glob.glob(f"{os.getcwd()}/MyDataset/{dataPath}*_Marker*")
    # print(pathExG)
    print(f"{os.getcwd()}/{dataPath}*_ExG*")
    print(words.values())
    # Create arrays and dataframe from them
    exgData = pd.DataFrame(pd.read_csv(pathExG[0]))
    markerData = pd.DataFrame(pd.read_csv(pathMarkers[0]))
    for pE, pM, in zip(pathExG[1:], pathMarkers[1:]):
        # print(type(exgData))
        exgData = pd.concat([exgData, pd.DataFrame(pd.read_csv(pE))])
        markerData = pd.concat([markerData, pd.DataFrame(pd.read_csv(pM))])

    return exgData, markerData


def createEpochs2(words, eegData, markerData, sampling_rate = 256):
    
    for marker in words.values():
        
    
    pass
