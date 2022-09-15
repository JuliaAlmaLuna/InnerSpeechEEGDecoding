import os
import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

#from google.colab import drive, files

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time


np.random.seed(23)

mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning )


def load_data(datatype, sampling_rate=256, subject_nr=3, t_start=0, t_end=5, verbose=True):
    
    root_dir = "eeg-imagined-speech-nieto"
    np.random.seed(23)
  
    mne.set_log_level(verbose='warning') #to avoid info at terminal
    warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
    warnings.filterwarnings(action = "ignore", category = FutureWarning )
   
    # Sampling rate
    fs = sampling_rate
    #datatype = datatype2
    # Subject number
    N_S = subject_nr   #[1 to 10]
   

    # Load all trials for a sigle subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)
    if verbose == True:
        print("Data shape: [trials x channels x samples]")
        print(X.shape) # Trials, channels, samples

        print("Labels shape")
        print(Y.shape) # Time stamp, class , condition, session
    #Classes :  0 = UP, 1 = DOWN, 2 = RIGHT, 3 = LEFT
    #Conditions : 0 = Pronounced, 1 = Inner, 2 = Visualized
    
    # Conditions to compared
    Conditions = [["Inner"],["Inner"]]
    # The class for the above condition
    Classes    = [  ["Up"] ,["Down"] ]

    # Transform data and keep only the trials of interest
    if datatype!= "baseline":
        X , Y =  Transform_for_classificator(X, Y, Classes, Conditions)
    if verbose == True:
        print("Final data shape")
        print(X.shape)

        print("Final labels shape")
        
        print(Y.shape)
    print("Up is {} and Down is {}".format(np.unique(Y)[0], np.unique(Y)[1]))
    return X, Y