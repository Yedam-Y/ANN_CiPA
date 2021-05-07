# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:32:03 2020

@author: Abebe
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from keras.utils import to_categorical
from train_data import drug_List, set_Label
import matplotlib as plt
import matplotlib.pyplot as py


def val_data(file_directory, file_ID):
    all_data =[]
    drug_lisk = []
    
    for i in range(len(file_ID)):
        a = file_ID[i].split('.')
        data = pd.read_csv(file_directory+file_ID[i], delimiter = ',',header = None)
        data = data.sample()
        all_data.append(data)
        
        drug_label = set_Label(data,drug_List(a[0]))
        drug_lisk.append(np.vstack(drug_label))
        
    return np.vstack(all_data), np.vstack(drug_lisk)

def make_Confusion_AUC(val_directory, val_ID, sc, model, Label, dataset, graph):
   
    fpr = dict()
    tpr = dict()
    auc_temp = dict()
    auc_value = []
    roc_auc = dict()

    f_score =[]
    acc = []
    
    for number in range(dataset):
        test, test_y = val_data(val_directory,val_ID )
        test = np.delete(test, [1,5,6,10], axis =1)

        test1 =sc.transform(test)
        y_pred2 = model.predict(test1)
        y_pred = np.argmax(y_pred2, axis =1)
        
        Y_te_cate = to_categorical(test_y)
        y_pred_cate = y_pred2
        
        f_temp = f1_score(test_y, y_pred, average= None)
        f_score.append(f_temp)
        
        accuracy = accuracy_score(test_y, y_pred)
        acc.append(accuracy)
        for i in range(len(Label)):
            fpr[i], tpr[i], _ = roc_curve(Y_te_cate[:,i], y_pred_cate[:,i])
            auc_temp =auc(fpr[i],tpr[i])
            auc_value.append(auc_temp)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        if graph == 1:
            color = plt.cm.rainbow(np.linspace(0, 1, len(Label)))
            lw = 2 # line width
            py.figure(figsize = (12, 10))
            py.rc('axes', labelsize = 30)
            py.rc('xtick', labelsize = 25)
            py.rc('ytick', labelsize = 25)    
            for i, c, in zip(range(len(Label)), color):
                py.plot(fpr[i], tpr[i], c=c, lw=lw, label = 'area = %0.2f' %roc_auc[i] + ' of %s' %Label[i])
            py.plot([0, 1], [0, 1], color = 'gray', lw=lw, linestyle ='--')
            py.xlim([0.0, 1.0])
            
            py.ylim([0.0, 1.05])
            py.xlabel('False Positive Rate')
            py.ylabel('True Positive Rate')
            py.legend(loc='Lower right')
            py.show()
            
            
    return np.stack(auc_value), np.stack(f_score), np.stack(acc)

def extract_index(AUC_a):
    
    avg = []
    avg1 =[]
    avg2 =[]

    for i,v in enumerate(AUC_a):
        if (i+3) % 3 ==0:
            avg.append(i)
        elif (i+3) % 3 ==1:
            avg1.append(i)
        elif (i+3) % 3 ==2:
            avg2.append(i)
    return np.stack(avg),np.stack(avg1),np.stack(avg2)

def auc_avg(AUC_a):
    low_auc =[]
    inter_auc =[]
    high_auc =[]
    
    # avg,avg1= extract_index(AUC_a)
    avg,avg1,avg2 = extract_index(AUC_a)

    for i,v in enumerate(avg):
        temp = AUC_a[v]
        low_auc.append(temp)
        
    for i,v in enumerate(avg1):
        temp = AUC_a[v]
        inter_auc.append(temp)  
        
    for i,v in enumerate(avg2):
        temp = AUC_a[v]
        high_auc.append(temp)
        
    return np.stack(low_auc,axis =0), np.stack(inter_auc,axis =0), np.stack(high_auc,axis =0)
