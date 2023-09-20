import numpy as np
from keras.utils import to_categorical

from scipy import signal
from scipy.signal import butter, lfilter, savgol_filter


def load_data(data_folder = '/content/gdrive/MyDrive/C247_NNDL_Project/data/', CNN_greyscale = False, one_hot_ylabel = False, only_subject_1 = False):
  '''
  Inputs:
  data_folder: Which folder is the dataset in
  CNN_greyscale: Expands dimensions to grey scale images for use in CNN. Defaults to False
  one_hot_ylabel: If true converts output to one hot vectors where each label is (1,4). Defaults to False
  only_subject_1: If true only returns data for subject 1. Does not return the person data. Defaults to False.
  '''
  person_train_valid = np.load(data_folder + "person_train_valid.npy")
  X_train_valid = np.load(data_folder + "X_train_valid.npy")
  y_train_valid = np.load(data_folder + "y_train_valid.npy")

  X_test = np.load(data_folder + "X_test.npy")
  y_test = np.load(data_folder + "y_test.npy")
  person_test = np.load(data_folder + "person_test.npy")

  y_train_valid = y_train_valid - np.min(y_train_valid)
  y_test = y_test - np.min(y_test)

  X_train_valid = np.transpose(X_train_valid, (0, 2, 1))
  X_test = np.transpose(X_test, (0, 2, 1))

  if CNN_greyscale:
    X_train_valid = np.expand_dims(X_train_valid, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

  if one_hot_ylabel:
    y_train_valid = to_categorical(y_train_valid)
    y_test = to_categorical(y_test)
  
  if only_subject_1:
    sub_1_train_valid_idx = np.where(person_train_valid.ravel() == 0)
    sub_1_test_idx = np.where(person_test.ravel() == 0)
    
    X_train_valid = X_train_valid[sub_1_train_valid_idx]
    y_train_valid = y_train_valid[sub_1_train_valid_idx]

    X_test = X_test[sub_1_test_idx]
    y_test = y_test[sub_1_test_idx]
  
  data = {
        'X_train_valid':X_train_valid,
        'y_train_valid':y_train_valid,
        'X_test':X_test,
        'y_test':y_test,
      }

  if(not only_subject_1):
    data['person_train_valid'] = person_train_valid
    data['person_test'] = person_test

  for key in data.keys():
    print('{}: {}'.format(key, data[key].shape))

  return data

def butter_bandpass(lowcut, highcut, fs, order):
  return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

def band_filter_eeg(data, lowcut, highcut, is_dimexpanded, fs = 250, order=5):
  '''
  data: the data to filer
  lowcut: lower cutoff for band filter
  highcut: upper cutoff for band filter
  is_dimexpanded: boolean for if the data is dimension expanded, ie has shape (samples, 1000, 22, 1)
  fs: sampling frequency. Is 250 for this data
  order: the order of the butterworth filter
  '''

  if is_dimexpanded:
    data = data.squeeze()

  filtered_data = np.zeros(data.shape)

  samples, _ ,channels = data.shape

  for s in range(samples):
    for c in range(channels):
      filtered_data[s,:,c] = butter_bandpass_filter(data[s,:,c], lowcut, highcut, fs, order)

  if is_dimexpanded:
    filtered_data = np.expand_dims(filtered_data, axis=-1)

  return filtered_data

def smooth_savgol_filter(data, window_length = 10, polyorder = 3):
  '''Applies a Savgol filter to the signals
  data: the data to apply it to
  window_length: the length of window, defaults to 10
  pylorder: the degre of polynomial fit, defaults to 3
  '''
  smoothened_data = savgol_filter(data, window_length, polyorder, axis = 1)
  return smoothened_data

def preprocess(X, y, sub_sample=2, noise=True, noise_var = 0.5, filter=True, max_pool=True, avg_pool=True):
  X = X[:,0:(500+500%sub_sample),:]
  X_aug = X[:, 0::sub_sample,:]
  y_aug = y

  for i in range(sub_sample-1):
    X_sub = X[:, i::sub_sample,:]
    X_aug = np.vstack((X_aug, X_sub))
    y_aug = np.hstack((y_aug, y))

  if max_pool:
    X_aug = np.vstack((X_aug, np.max(X.reshape(X.shape[0], -1, sub_sample, X.shape[2]), axis=2)))
    y_aug = np.hstack((y_aug, y))

  if avg_pool:
    X_aug = np.vstack((X_aug, np.mean(X.reshape(X.shape[0], -1, sub_sample, X.shape[2]), axis=2)))
    y_aug = np.hstack((y_aug, y))

  if noise:
    X_aug = X_aug + np.random.normal(0.0, noise_var, X_aug.shape)

  return X_aug, y_aug



