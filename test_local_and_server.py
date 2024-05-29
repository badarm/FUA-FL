import random
from imblearn.over_sampling import SMOTENC
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


import torch
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

def test_client_model(X_test, Y_test,  model,pp_Group,npp_Group,  sa_index, fairness_notion, ytest_potential, s_client_list, sensitive_feature):
    
    cce = tf.keras.losses.BinaryCrossentropy()
    logits = model.predict(X_test)
    preidect = np.around(logits)
    preidect = np.nan_to_num(preidect)
    Y_test = np.nan_to_num(Y_test)
    conf = (confusion_matrix(Y_test,preidect)) 
    TN = conf[0][0]
    FP = conf[0][1]
    FN = conf[1][0]
    TP = conf[1][1]
    sensitivity = TP/(TP+FN) 
    specificity = TN/(FP+TN)
        
    BalanceACC = (sensitivity+specificity)/2
    if fairness_notion == 'stat_parity':
        disc_score = find_statistical_parity_score(X_test,Y_test,preidect, pp_Group, npp_Group,  sa_index)
    elif fairness_notion == 'ate':
        disc_score = find_ate(ytest_potential, preidect, s_client_list, pp_Group)
    else:
        disc_score = find_eqop_score(X_test,Y_test,preidect, pp_Group, npp_Group,  sa_index)
    
    assigned_positive_labels = 0
    total_positive_labels = 0
    
    
    unique, counts = np.unique(preidect, return_counts=True)
    count_ap_dict = dict(zip(unique, counts))
    assigned_positive_labels = count_ap_dict.get(1,0)
    
    unique, counts = np.unique(Y_test, return_counts=True)
    count_tp_dict = dict(zip(unique, counts))
    total_positive_labels = count_tp_dict.get(1,0)
    
    return disc_score, assigned_positive_labels, total_positive_labels, BalanceACC

def test_model(X_test,Y_test, pp_Group, npp_Group,  sa_index,  model, comm_round, Y_test_potential, s_list, sensitive_feature):
    cce = tf.keras.losses.BinaryCrossentropy()
    logits = model.predict(X_test)
    preidect = np.around(logits)
    preidect = np.nan_to_num(preidect)
    Y_test = np.nan_to_num(Y_test)
    stat_parity = find_statistical_parity_score(X_test,Y_test,preidect, pp_Group, npp_Group,  sa_index)
    eqop = find_eqop_score(X_test,Y_test,preidect, pp_Group, npp_Group,  sa_index)    
    ate = find_ate(Y_test_potential, preidect, s_list, pp_Group)
    
    conf = (confusion_matrix(Y_test,preidect))   
    loss = cce(Y_test, preidect)
    acc = accuracy_score(preidect,Y_test)
    
    print('comm_round: {} | global_acc: {} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss,conf,stat_parity, eqop, ate

