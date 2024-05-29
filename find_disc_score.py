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
from test_local_and_server import *
from find_potential_outcomes_utilities import *

def find_ate(y_potential, predictions, protected_attr, pp_Group):

        tp_protected = 0.
        tn_protected = 0.
        fp_protected = 0.
        fn_protected = 0.

        tp_non_protected = 0.
        tn_non_protected = 0.
        fp_non_protected = 0.
        fn_non_protected = 0.
        
        saValue = pp_Group
        for idx in range(len(protected_attr)):
            # protrcted population
            if protected_attr[idx] == saValue:

                # correctly classified
                if y_potential[idx] == predictions[idx]:
                    if y_potential[idx] == 1:
                        tp_protected += 1.
                    else:
                        tn_protected += 1.
      
                else:
                    if y_potential[idx] == 1:
                        fn_protected += 1.
                    else:
                        fp_protected += 1.

            else:

                # correctly classified
                if y_potential[idx] == predictions[idx]:
                    if y_potential[idx] == 1:
                        tp_non_protected += 1.
                    else:
                        tn_non_protected += 1.
                # misclassified
                else:
                    if y_potential[idx] == 1:
                        fn_non_protected += 1.
                    else:
                        fp_non_protected += 1.

        if((tp_protected + fn_protected)==0):
            tpr_protected = 0
        else:
            tpr_protected = tp_protected / (tp_protected + fn_protected)
        #tnr_protected = tn_protected / (tn_protected + fp_protected)

        if((tp_non_protected + fn_non_protected) == 0):
            tpr_non_protected=0
        else:
            tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
        #tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)


        eqop = tpr_non_protected - tpr_protected
        return eqop
def find_statistical_parity_score(data,labels,predictions,pp_Group,npp_Group, sa_index):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    
    saIndex = sa_index
    saValue = pp_Group
    
    for idx, val in enumerate(data):
        # protected population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.
           
            
        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.
            
    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot
    return stat_par
    
def find_eqop_score(data,labels,predictions, pp_Group,npp_Group, sa_index):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    saIndex = sa_index
    saValue = pp_Group
    for idx, val in enumerate(data):
        # protrcted population
        if val[saIndex] == saValue:
            
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                #else:
                #    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                #else:
                #    fp_protected += 1.

        else:
            
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                #else:
                #    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                #else:
                #    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    #tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    #tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    
    eqop = tpr_non_protected - tpr_protected
    return eqop
