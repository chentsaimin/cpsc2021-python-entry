#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import keras
import os
import scipy.io as sio
import sys
import wfdb

from scipy import stats
from os import listdir
from tensorflow.python.client import device_lib
from keras.models import Sequential, load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers import UpSampling1D, Bidirectional, LeakyReLU, Dense, Dropout, Input, Convolution1D, GRU, Activation, Concatenate, Layer
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import regularizers, initializers, constraints
from keras import backend as K
from tqdm import tqdm
from scipy.signal import find_peaks
# from keras.utils import multi_gpu_model
from utils import qrs_detect, comp_cosEn, save_dict


fs_ = 200
random_seed = 42
num_classes = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #Use multi-gpu
window_size = 160

#model structure
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs): 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform') 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer) 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint) 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint) 
            self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint) 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W) 
        if self.bias:
            uit += self.b 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u) 
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def MetforNet121(length, num_channels, num_classes):
    main_input = Input(shape=(length,num_channels), dtype='float32')

    x = Convolution1D(12, 3, padding='same')(main_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 48, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    cnnout = Dropout(0.2)(x)
    x = Bidirectional(GRU(12, recurrent_activation='sigmoid', return_sequences=True, return_state=False, reset_after=True, use_bias=True))(cnnout)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes)(x)    
    main_output = Activation('sigmoid')(x)
    return Model(inputs=main_input, outputs=main_output)
sup_model = MetforNet121(window_size, 1, 1)
sup_model.load_weights('CPSC2021_MetforNet121_'+str(window_size))
print(sup_model.summary())

def windows_prediction(x_val_from_train, model_sup, filter_size=5200, channel=2, step=75):
    final_outputs_count = np.zeros((x_val_from_train.shape[0],x_val_from_train.shape[1]+(filter_size)*2,channel))
    final_outputs_sup = np.zeros((x_val_from_train.shape[0],x_val_from_train.shape[1]+(filter_size)*2,channel))
    x_val_from_train_temp = np.zeros((x_val_from_train.shape[0],x_val_from_train.shape[1]+(filter_size)*2,x_val_from_train.shape[2]))  
    x_val_from_train_temp[:,filter_size:-filter_size,:] = x_val_from_train.copy()
    for k in range(0,filter_size,step):
        x_val_from_train_temp2 = x_val_from_train_temp[:,k:k+x_val_from_train.shape[1],:].reshape(-1,filter_size,x_val_from_train.shape[2])
        
        x_val_from_train_norm = x_val_from_train_temp2.copy()
        mean = np.mean(x_val_from_train_temp2, axis=1)
        # std = np.std(x_val_from_train_temp2, axis=1)
        # std[std == 0] = np.finfo(float).eps
        for n in range(filter_size):
            x_val_from_train_norm[:,n,:] = x_val_from_train_norm[:,n,:] - mean
            # x_val_from_train_norm[:,n,:] /= std
        
        sup_answer = model_sup.predict(x_val_from_train_norm)
        sup_answer_temp = np.zeros([x_val_from_train_temp2.shape[0], x_val_from_train_temp2.shape[1], channel])
        for m in range(x_val_from_train_temp2.shape[1]):
            sup_answer_temp[:,m,:] += sup_answer 
        final_outputs_count[:,k:k+x_val_from_train.shape[1],:] += 1
        sup_answer_new = sup_answer_temp.reshape(-1,x_val_from_train.shape[1],channel)        
        final_outputs_sup[:,k:k+x_val_from_train.shape[1],:] += sup_answer_new[:,:,:]
    return_outputs = np.nan_to_num(final_outputs_sup[:,filter_size:-filter_size,:]/final_outputs_count[:,filter_size:-filter_size,:])
    return return_outputs

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def ngrams_rr(data, length):
    grams = []
    for i in range(0, length-12, 12):
        grams.append(data[i: i+12])
    return grams

def challenge_entry(sample_path):
    """
    This is a baseline method.
    """

    ECG, _, _ = load_data(sample_path)
    end_points = []

    post_size = window_size*4
    period = len(ECG)
    temp = np.zeros((1,((period//window_size)+1)*window_size,1))
    pred = np.zeros((1,((period//window_size)+1)*window_size,1))
    for i in range(ECG.shape[1]):
        temp[0,-period:,0] = ECG[:,i]
        pred += windows_prediction(temp, sup_model, filter_size=window_size, channel=1, step=10)
    pred /= ECG.shape[1]  
    prediction = np.round(pred[0,-period:,0], 0)

    previous = 0
    for i in range(period):
        if previous == 0 and prediction[i] == 1:
            end_points.append([i])
            previous = 1
        elif previous == 1 and prediction[i] == 0 :
            end_points[-1].append(i-1)
            previous = 0
    if previous == 1 :
       end_points[-1].append(period-1)

    if len(end_points) >= 2:
        end_points_temp = []
        continuous_start = end_points[0][0]
        for i in range(len(end_points)-1):
            if end_points[i+1][0] - end_points[i][-1] > post_size*2:
                end_points_temp.append([continuous_start, end_points[i][-1]])
                continuous_start = end_points[i+1][0]
        end_points_temp.append([continuous_start, end_points[-1][-1]])
        end_points = end_points_temp   

    end_points_temp = []
    for i in range(len(end_points)):
        if end_points[i][-1] - end_points[i][0] > post_size/2:
            end_points_temp.append(end_points[i])
    end_points = end_points_temp   

    if len(end_points) >= 1:
        if end_points[0][0] < post_size:
            end_points[0][0] = 0
        if period-1-end_points[-1][-1] < post_size:
            end_points[-1][-1] = period-1


    upperThreshold = 0.95
    lowerThreshold = 0.05 
    temp_P = np.zeros(prediction.shape)
    for i in range(len(end_points)):
        temp_P[end_points[i][0]:end_points[i][-1]+1] = 1
    AFPercentage = np.sum(temp_P)/len(temp_P)
    print(AFPercentage) 

    # if len(end_points) >= 2:
    #     if AFPercentage >= upperThreshold and end_points[0][0] == 0 and end_points[-1][-1] == period-1:
    #             end_points = [[0, period-1]]  
        
    if AFPercentage >= upperThreshold:
        end_points = [[0, len(prediction)-1]]
    if AFPercentage < lowerThreshold:
        end_points = []
    print(end_points)        
    pred_dcit = {'predict_endpoints': end_points}
    
    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

