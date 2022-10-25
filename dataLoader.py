import mne

import warnings
import numpy as np

# from google.colab import drive, files
from Inner_Speech_Dataset.Python_Processing.Data_extractions import (
    Extract_data_from_subject,
)
from Inner_Speech_Dataset.Python_Processing.Data_processing import (
    Select_time_window,
    Transform_for_classificator,
)


np.random.seed(23)

mne.set_log_level(verbose="warning")  # to avoid info at terminal
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


def load_data(
    datatype, sampling_rate=256, subject_nr=1, t_start=0, t_end=5, verbose=True
):

    root_dir = "eeg-imagined-speech-nieto"
    np.random.seed(23)

    mne.set_log_level(verbose="warning")  # to avoid info at terminal
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Sampling rate
    fs = sampling_rate
    # datatype = datatype2
    # Subject number
    N_S = subject_nr  # [1 to 10 d]

    # Load all trials for a sigle subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
    if verbose is True:
        print("Data shape: [trials x channels x samples]")
        print(X.shape)  # Trials, channels, samples

        print("Labels shape")
        print(Y.shape)  # Time stamp, class , condition, session
    # Classes :  0 = UP, 1 = DOWN, 2 = RIGHT, 3 = LEFT
    # Conditions : 0 = Pronounced, 1 = Inner, 2 = Visualized

    # Conditions to compared
    # Conditions = [["Inner"],["Inner"], ["Pronounced"], ["Pronounced"]]
    Conditions = [["Inner"], ["Inner"]]

    # The class for the above condition
    Classes = [["Up"], ["Down"]]
    # Classes    = [  ["Up"] ,["Down"], ["Up"] ,["Down"]]
    # Classes    = [  ["Up"] ,["Down"], ["Right"], ["Left"] ]

    # Transform data and keep only the trials of interest
    if datatype != "baseline":
        X, Y = Transform_for_classificator(X, Y, Classes, Conditions)
    if verbose is True:
        print("Final data shape")
        print(X.shape)

        print("Final labels shape")

        print(Y.shape)
    print("Up is {} and Down is {}".format(np.unique(Y)[0], np.unique(Y)[1]))
    return X, Y


def load_multiple_datasets(
    nr_of_datasets=1,
    datatype="EEG",
    sampling_rate=64,
    t_min=0,
    t_max=4.5,
    twoDLabels=True,
    specificSubject=1,
):
    # Minus nr 3
    if nr_of_datasets > 9:
        nr_of_datasets = 9

    datax, labelsx = load_data(
        datatype="EEG",
        subject_nr=specificSubject,
        verbose=True,
        sampling_rate=sampling_rate,
        t_start=t_min,
        t_end=t_max,
    )
    # datax = np.concatenate([datax[0:datax.shape[1]//2], np.zeros([datax.shape[0], 1, datax.shape[2]]),
    # datax[datax.shape[1]//2:]], axis=1)
    # datax[:,(datax.shape[1]//2)+1, (datax.shape[2]//nr_of_datasets)*0:(datax.shape[2]//nr_of_datasets)*1 ] = 1

    # for x in range(2,nr_of_datasets+1):
    for x in range(2, nr_of_datasets + 1):
        print("runninghere")
        if x == 3:
            continue
        data1, labels1 = load_data(
            datatype="EEG",
            subject_nr=x,
            verbose=False,
            sampling_rate=sampling_rate,
            t_start=t_min,
            t_end=t_max,
        )
        # data1 = np.concatenate([data1, np.zeros([data1.shape[0], nr_of_datasets, data1.shape[2]])], axis=1)
        # data1[:,data1.shape[1]-x, : ] = 1
        datax = np.concatenate([datax, data1], axis=0)
        labelsx = np.concatenate([labelsx, labels1], axis=0)

    data = datax
    labels1d = labelsx
    if twoDLabels is True:
        labels = np.zeros([labels1d.shape[0], 2])
        for row, label in enumerate(labels1d, 0):
            if label == 0:
                labels[row, 0] = 1
            if label == 1:
                labels[row, 1] = 1
    else:
        labels = labels1d
    return data, labels


def get_channelnames():
    return [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
        "A20",
        "A21",
        "A22",
        "A23",
        "A24",
        "A25",
        "A26",
        "A27",
        "A28",
        "A29",
        "A30",
        "A31",
        "A32",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
        "B15",
        "B16",
        "B17",
        "B18",
        "B19",
        "B20",
        "B21",
        "B22",
        "B23",
        "B24",
        "B25",
        "B26",
        "B27",
        "B28",
        "B29",
        "B30",
        "B31",
        "B32",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31",
        "C32",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "D11",
        "D12",
        "D13",
        "D14",
        "D15",
        "D16",
        "D17",
        "D18",
        "D19",
        "D20",
        "D21",
        "D22",
        "D23",
        "D24",
        "D25",
        "D26",
        "D27",
        "D28",
        "D29",
        "D30",
        "D31",
        "D32",
    ]
