# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:14:18 2020

@author: yyd81
"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import matplotlib.pyplot as py
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import scipy.stats as st
from validation_AUC import  make_Confusion_AUC, auc_avg
from train_data import data_load2
from collections import Counter
import pandas as pd 
import seaborn as sb 


file_directory = 'C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/2020/average/training/'
file_ID = os.listdir(file_directory)




val_directory = 'C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/2020/average/validation/'
val_ID = os.listdir(val_directory)
 

Tr_data1, y = data_load2(file_directory , file_ID ) 
x_tr1 = np.delete(Tr_data1, [1,5,6,10], axis =1)
sc = MinMaxScaler()
Tr_data = sc.fit_transform(x_tr1)

def Drug_Model():
    
    Model = tf.keras.models.Sequential()
    Model.add(tf.keras.layers.InputLayer(input_shape = (9, )))
    Model.add(tf.keras.layers.Dense(units = 5, kernel_initializer='normal', activation = 'relu'))
    Model.add(tf.keras.layers.Dense(3,activation ='Relu'))
    Model.summary()
    
    return Model



def Model_Train(file_directory, sc, file_ID, epochs, batch_size, n_split):

    tr, train_y = data_load2(file_directory , file_ID )
    tr1 = np.delete(tr, [1,5,6,10], axis =1)
    train = sc.transform(tr1)
               

    cv = KFold(n_split,shuffle = True)
    for i,(train_index, test_index) in enumerate(cv.split(train)):
        X_train, X_test, y_train, y_test = train_test_split(train, train_y,random_state =100, test_size=0.1)

     
        Model_path = 'C:/Users/CML_Ye/'
        
        if not os.path.exists(Model_path):
            os.mkdir(Model_path)
            
        save_path = Model_path+str(i)+'-{epoch:02d}-{loss:.4f}-{accuracy:.4f}--{val_loss:.4f}-{val_accuracy:.4f}-Model.hdf5'
        checkpoint = ModelCheckpoint(filepath = save_path, monitor = 'val_accuracy', save_best_only = True)
        callback_list = [checkpoint]
        
        Model = Drug_Model()
        Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
        hist = Model.fit(X_train,y_train, epochs = epochs,shuffle =True, batch_size = batch_size, validation_data = (X_test,y_test), callbacks = callback_list, verbose = 1)
        
        print('################################'+ str(i) + '#######################################')
        fig, loss_ax = py.subplots(figsize = (10,7))
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'r', linestyle ='--', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'b', linestyle ='--', label='val loss')

        acc_ax.plot(hist.history['accuracy'], 'r', label='train acc')
        acc_ax.plot(hist.history['val_accuracy'], 'b', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        py.savefig(Model_path+str(i)+'.png', dpi=300)
        py.show()
        
    return Model



train_model = Model_Train(file_directory, sc = sc, file_ID = file_ID, epochs = 300, batch_size =20, n_split = 10)


#####################################################################################################################
############################################# Model Test ############################################################


Label = ['low', 'inter', 'high']

model = tf.keras.models.load_model('C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/Model/0-28-0.4832-0.7579-valid-0.4472-0.7900-Model.hdf5')
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

AUC_a ,f, acc = make_Confusion_AUC(val_directory, val_ID, sc, model, Label, dataset = 10, graph =1)
low,inter,high = auc_avg(AUC_a)


def data_histogram(label, name):
         
    print('max: {: .3f}'.format(np.max(label)))
    print('median: {: .3f}'.format(np.median(label)))
    print('min: {: .3f}'.format(np.min(label)))
     
 
    py.figure(figsize=(8,8), dpi =300)
    n, bins, patches = py.hist(label, bins=10, linewidth=6, rwidth = 0.9)
    py.grid(axis='y', alpha=0.75)
    py.xticks(fontsize = 17)
    py.yticks(fontsize = 17)
    py.xlabel('AUC_Value', fontsize = 20)
    py.ylabel('Frequency', fontsize = 20)
    py.title(name, fontsize = 20)

# 각 위험군 별로 그래프 출력
frequency_high = data_histogram(label = high, name = 'High')                                  
frequency_inter = data_histogram(label = inter, name = 'Inter')
frequency_low = data_histogram(label = low, name = 'Low')
