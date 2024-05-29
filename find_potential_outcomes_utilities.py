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

def find_potential_outcomes(H, observed_output, sensitive_feature): #H is client_data

    G = H.copy(deep=True)
    G.reset_index(inplace=True)
    G = G.apply(pd.to_numeric)
    #Z is samples without output
    #now make output y as dataframe and connect it with the samples


    psm = PsmPy(G, treatment=sensitive_feature, indx='index', exclude = [])
    psm.logistic_ps(balance = True)
    psm.predicted_data
    psm.knn_matched(matcher='propensity_score', replacement=True, caliper=None)
    
    psm.matched_ids
    psm.predicted_data['propensity_logit']
    ps = psm.predicted_data['propensity_logit']
    #print(prop_logit)



    caliper = np.std(ps) * 0.5
    #print(f'caliper (radius) is: {caliper:.4f}')

    n_neighbors = 2

    # setup knn
    knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)
    
    potential_output = find_potential_outcomes_ordered(psm, observed_output, knn, sensitive_feature)
    return potential_output

def find_potential_outcomes_ordered(psm, observed_output, knn, sensitive_feature):
    female = psm.predicted_data[psm.predicted_data[sensitive_feature] == 1]
    male = psm.predicted_data[psm.predicted_data[sensitive_feature] == 0]

    knn.fit(male[['propensity_logit']])
    _, neighbor_indexes_female = knn.kneighbors(female[['propensity_logit']])

    knn.fit(female[['propensity_logit']])
    _, neighbor_indexes_male = knn.kneighbors(male[['propensity_logit']])

    # Create dictionaries with keys as original indexes and values as potential outcomes
    potential_output_female_dict = {female.index[i]: observed_output[idx] for i, idx in enumerate(neighbor_indexes_female[:, 0])}
    potential_output_male_dict = {male.index[i]: observed_output[idx] for i, idx in enumerate(neighbor_indexes_male[:, 0])}

    # Combine the dictionaries
    potential_output_dict = {**potential_output_female_dict, **potential_output_male_dict}

    # Create a list of potential outcomes, maintaining the original order in the dataset
    potential_output_ordered = [potential_output_dict[i] for i in sorted(potential_output_dict.keys())]

    return potential_output_ordered
