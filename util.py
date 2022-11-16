import numpy as np
import dataLoader as dl


# from Inner_Speech_Dataset.Plotting.ERPs import
from Inner_Speech_Dataset.Python_Processing.Data_processing import (
    Calculate_power_windowed,
)

# Frequencies

from scipy.fft import rfft  # , ifft


# Separate into equal 5 buckets
def sepFreqIndexBuckets(freqs2, nr_of_buckets=5):

    bucket_size_amp = np.sum(freqs2) / nr_of_buckets
    # print(bucket_size_amp)

    buckets = np.zeros([nr_of_buckets, 2])
    bucket = []
    cur_buck_size = 0

    b = 0
    c = 0
    for index, freqs in enumerate(freqs2, 0):
        cur_buck_size += freqs
        bucket.append(index)
        if cur_buck_size > bucket_size_amp:
            buckets[b] = [0 + c, c + len(bucket)]
            b += 1
            c += len(bucket)
            # print(len(bucket))
            bucket = []
            cur_buck_size = 0

    buckets[b] = [0 + c, c + len(bucket)]
    # print(len(bucket))

    return buckets


def createFreqBuckets(data, nr_of_buckets=5):

    nr_of_buckets = nr_of_buckets
    buckets = np.zeros([nr_of_buckets, 2])
    for trial in data:
        for channel in trial:
            buckets += sepFreqIndexBuckets(
                abs(rfft(channel))[: (channel.shape[0] // 2)], nr_of_buckets
            )

    buckets = buckets / (data.shape[0] * data.shape[1])

    return np.int32(buckets)


def data_into_freq_buckets(data, nr_of_buckets, buckets):

    freqAmps = np.zeros([data.shape[0], data.shape[1], nr_of_buckets])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            for b in range(nr_of_buckets):
                ff_c = abs(rfft(channel)) * 1000
                freqAmps[tr_nr, ch_nr, b] = np.sum(
                    ff_c[int(buckets[b, 0]) : int(buckets[b, 1])]
                )
    return freqAmps


def data_into_freq_array(data):

    freqarray = np.zeros([data.shape[0], data.shape[1], data.shape[2] // 2])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            ff_c = abs(rfft(channel))[: (channel.shape[0] // 2)]
            freqarray[tr_nr, ch_nr, :] = ff_c
    return freqarray


def fftCovariance(data):
    channelXE = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            e = np.mean(channel)
            channelXE[tr_nr, ch_nr, :] = channel - e

    channelCV = []
    for tr_nr, trial in enumerate(channelXE):
        channelCV.append(np.cov(trial))

    return channelCV


# Same results basically


def fftCorrelation(data):
    channelXE = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            e = np.mean(channel)
            channelXE[tr_nr, ch_nr, :] = channel - e

    channelCV = []
    for tr_nr, trial in enumerate(channelXE):
        matrix = []
        for ch_nr, channel in enumerate(trial):
            matrixRow = []
            for ch_nr2, channel2 in enumerate(trial):
                matrixRow.append(np.correlate(channel, channel2))
            matrix.append(np.array(matrixRow))
        channelCV.append(np.squeeze(np.array(matrix)))
    print(np.array(channelCV).shape)

    return channelCV


# def fftCovarianceRoll(data, roll = 5):
#     channelXE = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
#     channelXER = np.zeros([data.shape[0], data.shape[1], data.shape[2]])

#     #Unrolled
#     for tr_nr, trial in enumerate(data):
#         for ch_nr, channel in enumerate(trial):
#             e = np.mean(channel)
#             channelXE[tr_nr, ch_nr, :] = channel - e
#     #Rolled
#     channelXER = np.roll(channelXE, roll, 2)

#     channelCV = []
#     for tr_nr, trial in enumerate(channelXER):
#         matrix = []
#         for ch_nr, channel in enumerate(trial):
#             matrixRow = []
#             for ch_nr2, channel2 in enumerate(channelXE[tr_nr]):
#                 matrixRow.append(np.cov(channelXER[tr_nr, ch_nr], channelXE[tr_nr, ch_nr2]))
#             matrix.append(np.array(matrixRow)[:,0,1])
#             #print(np.array(matrixRow)[:,0,1].shape)
#         channelCV.append(np.array(matrix))
#         #print(np.array(matrix).shape)

#     return channelCV


def fftData(data):
    fftData = np.zeros([data.shape[0], data.shape[1], data.shape[2] // 2])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            fftData[tr_nr, ch_nr, :] = abs(rfft(channel))[: (channel.shape[0] // 2)]
    return fftData


# TODO: Split into real and imaginary instead. Send back both
def fftData2(data):
    fftDataAbs = np.zeros([data.shape[0], data.shape[1], data.shape[2] // 2])
    fftDataAngle = np.zeros([data.shape[0], data.shape[1], data.shape[2] // 2])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            # print(channel.shape[0])
            # print(channel.shape[0] // 2)
            fftDataAbs[tr_nr, ch_nr, :] = abs(rfft(channel))[: (channel.shape[0] // 2)]
            # print("julia1")
            # print(fftDataAbs)
            # print("julia2")
            fftDataAngle[tr_nr, ch_nr, :] = np.angle(rfft(channel))[
                : (channel.shape[0] // 2)
            ]

            # print(fftDataAngle)
            # print("julia3")
    return fftDataAbs, fftDataAngle


def shortTimefftData(data, windowLength, nperseg):
    from scipy.signal import stft

    sfftData = np.zeros([data.shape[0], data.shape[1], data.shape[2] // 2])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            sfftData[tr_nr, ch_nr, :] = stft(
                channel, noverlap=True, window=windowLength, nperseg=nperseg
            )
            # abs(rfft(channel))[: (channel.shape[0] // 2)]
    return abs(sfftData), sfftData.real, sfftData.imag


# use complex(data, dataC.imag) to put them back together
def ifftData(data, angle):
    import cmath

    # Compute fft inluding complex from orig data, the complex/phase part of this one needs to be kept.
    # Then add this complex part to fftDataBC, then do iFFT
    # from scipy import irfft
    from scipy.fft import irfft

    ifftData = np.zeros([data.shape[0], data.shape[1], data.shape[2] * 2])
    for tr_nr, (trial, trialA) in enumerate(zip(data, angle)):
        for ch_nr, (channelAbs, channelAngle) in enumerate(zip(trial, trialA)):
            channel = []
            for abs, angle in zip(channelAbs, channelAngle):
                channel.append(cmath.rect(abs, angle))

            channel = np.array(channel)

            # channel = cmath.rect(channelAbs, channelAngle)
            # test = irfft(channel)
            # print(test.shape)
            ifftData[tr_nr, ch_nr, :] = irfft(channel, n=data.shape[2] * 2)
    return ifftData


def welchData(data, nperseg, fs=256):
    from scipy.signal import welch

    if nperseg < fs:
        arSize = nperseg // 2
    else:
        arSize = fs // 2
    welchData = np.zeros([data.shape[0], data.shape[1], arSize])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):

            welchData[tr_nr, ch_nr, :] = welch(channel, fs=fs, nperseg=nperseg)[1][
                0:arSize
            ]
    return welchData


def welchData3(data, nperseg, fs=256):
    from scipy.signal import welch

    # could just be
    welchData = welch(data, fs=fs, window="blackman")

    return welchData


def arrToDict(arr):
    dict = {}
    for row in arr:
        dict[row[0]] = row[1]

    return dict


def get_channelNames():
    ch_names = np.array(dl.get_channelnames())
    nr = np.arange(ch_names.shape[0])
    ch_names = np.array([ch_names, nr]).T
    ch_names = arrToDict(ch_names)
    return ch_names


def only_spec_channel_data(data, picks):

    channel_names_string = picks_from_channels(picks)
    ch_names = get_channelNames()
    channel_nr = []
    for name in channel_names_string:
        channel_nr.append(int(ch_names[name]))
        # print(ch_names[name])

    channel_nr = np.array(channel_nr)

    # print(channel_nr)
    # data = np.swapaxes(data, 0, 1)
    # labels = np.swapaxes(labels, 0, 1)
    # for channelnrs in channels:
    data2 = np.delete(data, np.delete(np.arange(128), channel_nr), axis=1)
    return data2


def get_power_array(split_data, samplingRate, trialSplit=1, t_min=0, t_max=0.99):

    # trialSplit = 16
    sR = samplingRate  # samplingRate = 32
    data_power = np.zeros([split_data.shape[0], split_data.shape[1], trialSplit, 2])
    for t, trial in enumerate(split_data, 0):
        for c, channel in enumerate(trial, 0):
            for x in range(trialSplit):
                data_power[t, c, x, :] = Calculate_power_windowed(
                    channel,
                    fc=sR,
                    window_len=1 / 8,
                    window_step=1 / 8,
                    t_min=t_min * (1 / trialSplit),
                    t_max=t_max * (1 / trialSplit),
                )

    # m_power , std_power
    # print(data_power.shape)
    return data_power


# Create Frequency buckets using either,
# equal amp splits, equal band width splits or manual


def getFreqBuckets(data, nr_of_buckets=15):
    # Equal amp splits
    buckets = createFreqBuckets(data[:, :128, :], nr_of_buckets)

    # Equal band width splits
    # buckets = np.reshape(np.linspace(0,80,nr_of_buckets*2),[nr_of_buckets, -1])

    # Manual
    # buckets = np.array([[0,3],[4,8],[9,15],[16,34],[35,45],[45,80]])

    nr_of_buckets = buckets.shape[0]
    print("buckets")
    print(buckets)
    return buckets


def splitData(data, labels, split, seed=None):

    if seed is not None:
        np.random.seed(seed)

    order = np.arange(labels.shape[0])
    np.random.shuffle(order)

    temp_data = np.zeros(data.shape)
    temp_labels = np.zeros(labels.shape)

    for x in range(labels.shape[0]):
        i = order[x]

        temp_data[x] = data[i]
        temp_labels[x] = labels[i]

    dataT = temp_data
    labelsT = temp_labels

    data_train, data_test = np.split(
        dataT, indices_or_sections=[int(labelsT.shape[0] * split)], axis=0
    )
    labels_train, labels_test = np.split(
        labelsT, indices_or_sections=[int(labelsT.shape[0] * split)], axis=0
    )
    print(labels_train.shape)
    print(data_train.shape)
    print(data_test.shape)
    return data_train, data_test, labels_train, labels_test


def picks_from_channels(channels):
    """
    Parameters
    ----------
    channels : str
        Name of the channel or regions.

    Returns
    -------
    picks : list
        List of picks that corresponds with the channels.

    """
    if channels == "A":
        picks = [
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
        ]
    elif channels == "B":
        picks = [
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
        ]
    elif channels == "C":
        picks = [
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
        ]
    elif channels == "D":
        picks = [
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

    elif channels == "OCC_L" or channels == "OL":
        picks = ["A10", "A8", "D30", "A9"]
    elif channels == "OCC_Z" or channels == "OZ":
        picks = ["A22", "A23", "A24", "A15", "A28"]
    elif channels == "OCC_R" or channels == "OR":
        picks = ["B12", "B5", "B6", "B7"]

    elif channels == "FRONT_L" or channels == "FL":
        picks = ["D6", "D5", "C32", "C31"]
    elif channels == "FRONT_Z" or channels == "FZ":
        picks = ["C18", "C20", "C27", "C14"]
    elif channels == "FRONT_R" or channels == "FR":
        picks = ["C9", "C6", "C10", "C5"]

    elif channels == "C_L" or channels == "CL":
        picks = ["D26", "D21", "D10", "D19"]
    elif channels == "C_Z" or channels == "CZ":
        picks = ["D15", "A1", "B1", "A2"]
    elif channels == "C_R" or channels == "CR":
        picks = ["B16", "B24", "B29", "B22"]

    elif channels == "P_Z" or channels == "PZ":
        picks = ["A4", "A19", "A20", "A32", "A5"]

    elif channels == "OP_Z" or channels == "OPZ":
        picks = ["A17", "A30", "A20", "A21", "A22"]

    elif channels == "all" or channels == "All":
        picks = "all"
    else:
        picks = []
        raise Exception("Invalid channels name")

    return picks


def channelNumbersFromGroupName(groupName):
    cn = get_channelNames()
    channelNrArray = []
    picks = picks_from_channels(groupName)
    for pick in picks:
        channelNrArray.append(int(cn[pick]))
    return np.array(channelNrArray)
