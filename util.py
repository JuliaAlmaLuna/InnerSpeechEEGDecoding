import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import dataLoader as dl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from Inner_Speech_Dataset.Plotting.ERPs import 
from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Calculate_power_windowed
from Inner_Speech_Dataset.Python_Processing.Utilitys import picks_from_channels
from Inner_Speech_Dataset.Python_Processing.Data_processing import Average_in_frec

#Frequencies

from scipy.fft import rfft, ifft, fftshift, fftfreq


#Separate into equal 5 buckets
def sepFreqIndexBuckets(freqs2, nr_of_buckets = 5): 
     
    bucket_size_amp = np.sum(freqs2)/nr_of_buckets
    #print(bucket_size_amp)
    
    buckets = np.zeros([nr_of_buckets, 2])
    bucket = []
    cur_buck_size = 0
    
    b = 0
    c = 0
    for index, freqs in enumerate(freqs2,0):
        cur_buck_size += freqs
        bucket.append(index)
        if cur_buck_size > bucket_size_amp:
            buckets[b] = [0 + c , c + len(bucket)]
            b += 1
            c += len(bucket)
            #print(len(bucket))
            bucket = []
            cur_buck_size = 0
            
    
    buckets[b] = [0 + c , c + len(bucket)]
    #print(len(bucket)) 
       
    return buckets  




def createFreqBuckets(data, nr_of_buckets = 5):


    nr_of_buckets = nr_of_buckets
    buckets = np.zeros([nr_of_buckets, 2])
    for trial in data:
        for channel in trial:
            buckets += sepFreqIndexBuckets(abs(rfft(channel))[:(channel.shape[0]//2)], nr_of_buckets)
            
    buckets = buckets/(data.shape[0]*data.shape[1])
    
    
    return np.int32(buckets)


def data_into_freq_buckets(data, nr_of_buckets, buckets):

    freqAmps = np.zeros([data.shape[0], data.shape[1], nr_of_buckets])
    for tr_nr, trial in enumerate(data):
        for ch_nr, channel in enumerate(trial):
            for b in range(nr_of_buckets):
                ff_c = abs(rfft(channel))*1000
                freqAmps[tr_nr, ch_nr, b] = np.sum(ff_c[int(buckets[b, 0]):int(buckets[b,1])])
    return freqAmps


#Channel name array

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

def only_spec_channel_data(data , picks):
    
    channel_names_string = picks_from_channels(picks)
    ch_names = get_channelNames()
    channel_nr = []
    for name in  channel_names_string:
        channel_nr.append(int(ch_names[name]))
        #print(ch_names[name])

    channel_nr = np.array(channel_nr)
    
    #print(channel_nr)
    #data = np.swapaxes(data, 0, 1)
    #labels = np.swapaxes(labels, 0, 1)
    #for channelnrs in channels:
    data2 = np.delete(data, np.delete(np.arange(128), channel_nr) , axis=1)
    return data2


def get_power_array(split_data , samplingRate, trialSplit = 1, t_min = 0, t_max = 0.99):

    #trialSplit = 16
    sR = samplingRate #samplingRate = 32
    data_power = np.zeros([split_data.shape[0], split_data.shape[1], trialSplit, 2])
    for t, trial in enumerate(split_data,0):
        for c, channel in enumerate(trial,0):
            for x in range(trialSplit):
                data_power[t, c, x, : ] = Calculate_power_windowed(channel, fc=sR, window_len=1/8, window_step=1/8, t_min=t_min*(1/trialSplit), t_max=t_max*(1/trialSplit))


    #m_power , std_power
    #print(data_power.shape)
    return data_power
    


