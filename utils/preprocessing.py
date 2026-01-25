import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from tqdm import tqdm
import more_itertools as mit
import copy
import time
import random
import numpy as np
from sklearn import preprocessing
import scipy.io as sio
from sklearn import linear_model
from scipy import interpolate
import csv
random.seed(2)

class Data_preprocess:

    def files_load(self, file_name):
        train_data = np.loadtxt(file_name)
        return train_data

    #transform and delete unused sensors (normalization)
    def normalization_data(self, data_train, data_test, unused_sensors):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_train[:, 2:] = min_max_scaler.fit_transform(data_train[:, 2:])
        data_test[:, 2:] = min_max_scaler.transform(data_test[:, 2:])
        data_train_normalized = data_train
        data_test_normalized = data_test
        data_train_normalized = np.delete(data_train_normalized, unused_sensors, axis=1)
        data_test_normalized = np.delete(data_test_normalized, unused_sensors, axis=1)

        return data_train_normalized, data_test_normalized     

    def sliding_window_processed (self, train_data, test_data, rul_data, window_size, RUL_max): 
        trainX = []
        trainY = []
        testX = []
        testY = []
        testY_bu = []
        trainY_bu = [] 
        testLen = []
        #Training set sliding time window processing     
        for i in range(1, int(np.max(train_data[:, 0])) + 1):  
            ind = np.where(train_data[:, 0] == i)  
            ind = ind[0]
            data_temp = train_data[ind, :] 
            for j in range(len(data_temp) - window_size + 1):  
                trainX.append(data_temp[j:j + window_size, 2:].tolist())
                train_RUL = len(data_temp) - window_size - j  
                train_bu = RUL_max - train_RUL
                if train_RUL > RUL_max:
                    train_RUL = RUL_max
                    train_bu = 0.0
                trainY.append(train_RUL)
                trainY_bu.append(train_bu)
        trainX = np.array(trainX)
        trainY = np.array(trainY)/RUL_max
            
        #Test set sliding time window processing
        for i in range(1, int(np.max(test_data[:, 0])) + 1): 
            ind = np.where(test_data[:, 0] == i)
            ind = ind[0]
            testLen.append(float(len(ind)))
            data_temp = test_data[ind, :] 
            testY_bu.append(data_temp[-1, 1])
            if len(data_temp) < window_size:  
                data_temp_a = []
                for myi in range(data_temp.shape[1]):
                    x1 = np.linspace(0, window_size - 1, len(data_temp))
                    x_new = np.linspace(0, window_size - 1, window_size)
                    tck = interpolate.splrep(x1, data_temp[:, myi])
                    a = interpolate.splev(x_new, tck)
                    data_temp_a.append(a.tolist())
                data_temp_a = np.array(data_temp_a)
                data_temp = data_temp_a.T
                data_temp = data_temp[:, 2:]
            else:
                data_temp = data_temp[-window_size:, 2:]  

            data_temp = np.reshape(data_temp, (1, data_temp.shape[0], data_temp.shape[1])) 
            
            if i == 1:
                testX = data_temp
            else:
                testX = np.concatenate((testX, data_temp), axis=0)
            if rul_data[i - 1] > RUL_max:
                testY.append(RUL_max)
                #testY_bu.append(0.0)
            else:
                testY.append(rul_data[i - 1])

        testX = np.array(testX)
        testY = np.array(testY)/RUL_max

        return trainX, trainY, testX, testY  

    def save_data (self, file_name, data):
        data = np.array(data)
        sio.savemat(file_name, {"data": data})

class processing:
    def __init__(self, file_name_train_data_to_load, file_name_test_data_to_load, file_name_rul_data_to_load,
                 window_size, unused_sensors, rul_max, feasize):
        self.file_name_train_data_to_load = file_name_train_data_to_load
        self.file_name_test_data_to_load = file_name_test_data_to_load
        self.file_name_rul_data_to_load = file_name_rul_data_to_load
        self.window_size = window_size
        self.unused_sensors = unused_sensors
        self.rul_max = rul_max
        self.feasize = feasize

        process = Data_preprocess()
        train_data = process.files_load(self.file_name_train_data_to_load)
        test_data = process.files_load(self.file_name_test_data_to_load)
        rul_data = process.files_load(self.file_name_rul_data_to_load)
        train_data_nor, test_data_nor = process.normalization_data(train_data, test_data, self.unused_sensors)
        self.trainX, self.trainY, self.testX, self.testY = process.sliding_window_processed (train_data_nor, test_data_nor, rul_data, self.window_size, self.rul_max)
        self.trainY = self.trainY.reshape((1, self.trainY.shape[0]))
        self.testY = self.testY.reshape((1, self.testY.shape[0]))

    def get_data(self):        
        return self.trainX, self.trainY, self.testX, self.testY