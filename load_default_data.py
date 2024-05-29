# import urllib2
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf

# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)

def load_default():
    FEATURES_CLASSIFICATION = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]  # features to be used for classification
    CONT_VARIABLES = ["AGE","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y"  # the decision variable
    SENSITIVE_ATTRS = ["SEX"]
    CAT_VARIABLES = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    CAT_VARIABLES_INDICES = [0,2,3,5,6,7,8,9,10]
    
    INPUT_FILE = "./datasets/default.csv"

    df = pd.read_csv(INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    #y[y == "yes"] = 1
    #y[y == 'no'] = -1
    y = np.array([int(k) for k in y])

    X = np.array([]).reshape(len(y), 0)  # empty array with num rows same as num examples, will hstack the features to it
    
    x_control = defaultdict(list)
    i=0
    feature_names = []
    
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
            

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            
            lb = preprocessing.LabelEncoder()
            lb.fit(vals)
            vals = lb.transform(vals)
            vals = np.reshape(vals, (len(y), -1))
            '''
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
            
            if attr == 'SEX':
                print(lb.classes_)
                print(lb.transform(lb.classes_))
            '''
            #if attr == 'job':
            #    print(lb.classes_)
            #    print(lb.transform(lb.classes_))
            
           
            
            
        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            

        # add to learnable features
        X = np.hstack((X, vals))
        
        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    
    x_control = dict(x_control)
    
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
    
    feature_names.append('target')
    # '0' is 'female'
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, 1, x_control, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS


#####



def process_loaded_default_data(dataFrame):
    FEATURES_CLASSIFICATION = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]  # features to be used for classification
    CONT_VARIABLES = ["AGE","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y"  # the decision variable
    SENSITIVE_ATTRS = ["SEX"]
    CAT_VARIABLES = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    CAT_VARIABLES_INDICES = [0,2,3,5,6,7,8,9,10]
    
    INPUT_FILE = "./datasets/default.csv"

    df = pd.read_csv(INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    #y[y == "yes"] = 1
    #y[y == 'no'] = -1
    y = np.array([int(k) for k in y])

    X = np.array([]).reshape(len(y), 0)  # empty array with num rows same as num examples, will hstack the features to it
    
    x_control = defaultdict(list)
    i=0
    feature_names = []
    
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
            

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelEncoder()
            lb.fit(vals)
            vals = lb.transform(vals)
            vals = np.reshape(vals, (len(y), -1))
            '''
            if attr == 'SEX':
                print(lb.classes_)
                print(lb.transform(lb.classes_))
            '''
            #if attr == 'job':
            #    print(lb.classes_)
            #    print(lb.transform(lb.classes_))
            
           
            
            
        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals
            

        # add to learnable features
        X = np.hstack((X, vals))
        
        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    
    x_control = dict(x_control)
    
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
    
    feature_names.append('target')
    # '0' is 'female'
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS

def load_default_attr():

    
    INPUT_FILE = "./datasets/default.csv"

    df = pd.read_csv(INPUT_FILE)
    ranges = [0,30,40,50,100]
    
    mask = (df['AGE'] > 0) & (df['AGE'] < 30)
    df1 = df[mask]
    mask = (df['AGE'] > 29) & (df['AGE'] < 40)
    df2 = df[mask]
    mask = (df['AGE'] > 39) & (df['AGE'] < 80)
    df3 = df[mask]
    mask = (df['AGE'] > 49)
    df4 = df[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    #df4['y'] = pd.Series(np.where(df4.y.values == 'yes', 1, 0),df4.index)
    #print(df4.groupby('y').count())
    X1,y1, sa_index, p_Group, x_control, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS = process_loaded_default_data(df1)
    X2,y2, sa_index, p_Group, x_control, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS = process_loaded_default_data(df2)
    X3,y3, sa_index, p_Group, x_control, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS = process_loaded_default_data(df3)
    X4,y4, sa_index, p_Group, x_control, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS = process_loaded_default_data(df4)
    np_Group = 1
    return X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group, FEATURES_CLASSIFICATION, SENSITIVE_ATTRS