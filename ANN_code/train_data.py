# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:31:12 2020

@author: Abebe
"""
import pandas as pd
import numpy as np 

def drug_List(drug_name):
    
    drug_list_file = r'C:/Users/CML_Ye/cmax1-4/drug_list.csv'
    df =pd.read_csv(drug_list_file, delimiter = ',')
    y_label = df.values
    
    i = 0
    while drug_name != y_label[i,0]:
        i = i+1
        # print(drug_name)
    return y_label[i, 1]
# ' low' = 2, 'inter' =1, 'high' = 0

def set_Label(data, risk_level):
    
    if risk_level == 'low':
        Label = np.zeros(len(data))
    elif risk_level == 'inter':
        Label = np.ones(len(data))
    else:
        Label = np.ones(len(data))*2
        
    return Label


def data_load2(file_directory, file_ID):
    all_data =[]
    drug_lisk = []
    
    for i in range(len(file_ID)):
        a = file_ID[i].split('.')
        data = pd.read_csv(file_directory+file_ID[i], delimiter = ',',header = None)
        data = data.values[:,:]
        all_data.append(np.stack(data))
        # print(len(data))
        # print(file_ID[i])
        
        drug_label = set_Label(data,drug_List(a[0]))
        drug_lisk.append(np.vstack(drug_label))
        
    X = np.vstack(all_data)
    y = np.vstack(drug_lisk)
    
    return X, y
