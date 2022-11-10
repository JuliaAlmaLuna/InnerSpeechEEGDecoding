import numpy as np
import dataLoader as dl


# from Inner_Speech_Dataset.Plotting.ERPs import
from Inner_Speech_Dataset.Python_Processing.Data_processing import (
    Calculate_power_windowed,
)
from Inner_Speech_Dataset.Python_Processing.Utilitys import picks_from_channels


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
    fftDataC = np.zeros([data.shape[0], data.shape[1], data.shape[2] // 2])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            fftDataC[tr_nr, ch_nr, :] = rfft(channel)[: (channel.shape[0] // 2)]
    return fftDataC.real, fftDataC.imag


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
def ifftData(data, dataC):
    # Compute fft inluding complex from orig data, the complex/phase part of this one needs to be kept.
    # Then add this complex part to fftDataBC, then do iFFT
    from scipy import irfft

    ifftData = np.zeros([data.shape[0], data.shape[1], data.shape[2] * 2])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            ifftData[tr_nr, ch_nr, :] = irfft(channel)
            abs(rfft(channel))[: (channel.shape[0] // 2)]
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


def welchData2(data, nperseg, fs=256):
    from scipy.signal import welch

    if nperseg < fs:
        arSize = nperseg // 2
    else:
        arSize = fs // 2
    welchData = np.zeros([data.shape[0], data.shape[1], arSize])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            welchData[tr_nr, ch_nr, :] = welch(
                channel, fs=fs, nperseg=nperseg, scaling="spectrum"
            )[1][0:arSize]
    return welchData


# Channel name array


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
