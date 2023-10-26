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


def load_adult():
    FEATURES_CLASSIFICATION = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                               "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"] #features to be used for classification
    CONT_VARIABLES = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y" # the decision variable
    SENSITIVE_ATTRS = ["sex"]
    
    COMPAS_INPUT_FILE = "./datasets/adult2.csv"

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df.dropna()
    # convert to np arra
    data = df.to_dict('list')

    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """
    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y==0] = -1
    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)
    
    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col
        
        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelEncoder()
            lb.fit(vals)
            vals = lb.transform(vals)
            vals = np.reshape(vals, (len(y), -1))
            #if attr =="sex": 
                #print(lb.classes_)
                #print(lb.transform(lb.classes_))
        
        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

    
    #print(SENSITIVE_ATTRS[0])
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, 1, x_control

def load_bank():
    FEATURES_CLASSIFICATION = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                               "day", "month", "duration", "campaign", "pdays", "previous",
                               "poutcome"]  # features to be used for classification
    CONT_VARIABLES = ["age", "balance", "day", "duration", "campaign", "pdays",
                      "previous"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y"  # the decision variable
    SENSITIVE_ATTRS = ["marital"]
    CAT_VARIABLES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "previous", "poutcome"]
    CAT_VARIABLES_INDICES = [1,2,3,4,6,7,8,10,14,15]
    # COMPAS_INPUT_FILE = "bank-full.csv"
    COMPAS_INPUT_FILE = "./datasets/bank-full.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == "yes"] = 1
    y[y == 'no'] = -1
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
    # '0' is the marital status = 'married'
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, 1, x_control

def load_law():
    FEATURES_CLASSIFICATION = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa","zgpa", "fulltime", "fam_inc", "sex", "race", "tier"]  # features to be used for classification
    CONT_VARIABLES = ["lsat", "ugpa", "zfygpa", "zgpa"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y"  # the decision variable
    SENSITIVE_ATTRS = ["sex"]
    CAT_VARIABLES = ["decile1b", "decile3", "fulltime", "fam_inc", "sex", "race", "tier"]
    CAT_VARIABLES_INDICES = [1,2,7,8,9,10,11]
    # COMPAS_INPUT_FILE = "bank-full.csv"
    INPUT_FILE = "./datasets/law.csv"

    df = pd.read_csv(INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    #print(np.unique(y))
    #y[y == 0] = 1
    #y[y==0] = -1
    #y[y == 1] = -1
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
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
           
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
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 1,0, x_control

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
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
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
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, 1, x_control


#####
def process_loaded_adult_data(dataFrame):
    FEATURES_CLASSIFICATION = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                               "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"] #features to be used for classification
    CONT_VARIABLES = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y" # the decision variable
    SENSITIVE_ATTRS = ["sex"]
    
    COMPAS_INPUT_FILE = "./datasets/adult2.csv"
    CAT_VARIABLES_INDICES = [1,2,3,4,6,7,8,10,14,15]
    # COMPAS_INPUT_FILE = "bank-full.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == "yes"] = 1
    y[y == 'no'] = 0
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
    # '0' is the marital status = 'married'
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control

def load_adult_attr():

    # COMPAS_INPUT_FILE = "bank-full.csv"
    COMPAS_INPUT_FILE = "./datasets/adult2.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)
    ranges = [0,30,40,50,100]
    
    mask = (df['age'] > 0) & (df['age'] < 30)
    df1 = df[mask]
    mask = (df['age'] > 29) & (df['age'] < 40)
    df2 = df[mask]
    mask = (df['age'] > 40)
    df3 = df[mask]
    
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    
    X1,y1, sa_index, p_Group, x_control = process_loaded_adult_data(df1)
    X2,y2, sa_index, p_Group, x_control = process_loaded_adult_data(df2)
    X3,y3, sa_index, p_Group, x_control = process_loaded_adult_data(df3)
    np_Group = 1
    return X1,X2,X3,y1,y2,y3, sa_index, p_Group, np_Group

def process_loaded_bank_data(dataFrame):
    FEATURES_CLASSIFICATION = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                               "day", "month", "duration", "campaign", "pdays", "previous",
                               "poutcome"]  # features to be used for classification
    CONT_VARIABLES = ["age", "balance", "day", "duration", "campaign", "pdays",
                      "previous"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y"  # the decision variable
    SENSITIVE_ATTRS = ["marital"]
    CAT_VARIABLES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "previous", "poutcome"]
    CAT_VARIABLES_INDICES = [1,2,3,4,6,7,8,10,14,15]
    # COMPAS_INPUT_FILE = "bank-full.csv"
    COMPAS_INPUT_FILE = "./datasets/bank-full.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == "yes"] = 1
    y[y == 'no'] = 0
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
    # '0' is the marital status = 'married'
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control

def load_bank_attr():

    # COMPAS_INPUT_FILE = "bank-full.csv"
    COMPAS_INPUT_FILE = "./datasets/bank-full.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)
    ranges = [0,30,40,50,100]
    
    mask = (df['age'] > 0) & (df['age'] < 30)
    df1 = df[mask]
    mask = (df['age'] > 29) & (df['age'] < 40)
    df2 = df[mask]
    mask = (df['age'] > 39) & (df['age'] < 80)
    df3 = df[mask]
    mask = (df['age'] > 49)
    df4 = df[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    #df4['y'] = pd.Series(np.where(df4.y.values == 'yes', 1, 0),df4.index)
    #print(df4.groupby('y').count())
    X1,y1, sa_index, p_Group, x_control = process_loaded_bank_data(df1)
    X2,y2, sa_index, p_Group, x_control = process_loaded_bank_data(df2)
    X3,y3, sa_index, p_Group, x_control = process_loaded_bank_data(df3)
    X4,y4, sa_index, p_Group, x_control = process_loaded_bank_data(df4)
    np_Group = 1
    return X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group

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
   
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control

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
    X1,y1, sa_index, p_Group, x_control = process_loaded_default_data(df1)
    X2,y2, sa_index, p_Group, x_control = process_loaded_default_data(df2)
    X3,y3, sa_index, p_Group, x_control = process_loaded_default_data(df3)
    X4,y4, sa_index, p_Group, x_control = process_loaded_default_data(df4)
    np_Group = 1
    return X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group



def process_loaded_law_data(dataFrame):
    FEATURES_CLASSIFICATION = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa","zgpa", "fulltime", "fam_inc", "sex", "race", "tier"]  # features to be used for classification
    CONT_VARIABLES = ["lsat", "ugpa", "zfygpa", "zgpa"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "y"  # the decision variable
    SENSITIVE_ATTRS = ["sex"]
    CAT_VARIABLES = ["decile1b", "decile3", "fulltime", "fam_inc", "sex", "race", "tier"]
    CAT_VARIABLES_INDICES = [1,2,7,8,9,10,11]
    # COMPAS_INPUT_FILE = "bank-full.csv"
    INPUT_FILE = "./datasets/law.csv"

    df = pd.read_csv(INPUT_FILE)
    
    # convert to np array
    data = df.to_dict('list')
    
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    #print(np.unique(y))
    #y[y == 0] = 1
    #y[y==0] = -1
    #y[y == 1] = -1
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
    
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 1, x_control


def load_law_attr():

    # COMPAS_INPUT_FILE = "bank-full.csv"
    COMPAS_INPUT_FILE = "./datasets/law.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)
    ranges = [0,30,40,50,100]
    mask = (df['fam_inc'] >=1) & (df['fam_inc'] <=2)
    df1 = df[mask]
    mask = (df['fam_inc'] ==3)
    df2 = df[mask]
    mask = (df['fam_inc'] >=4) & (df['fam_inc'] <=5)
    df3 = df[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    #df4['y'] = pd.Series(np.where(df4.y.values == 'yes', 1, 0),df4.index)
    #print(df4.groupby('y').count())
    X1,y1, sa_index, p_Group, x_control = process_loaded_law_data(df1)
    X2,y2, sa_index, p_Group, x_control = process_loaded_law_data(df2)
    X3,y3, sa_index, p_Group, x_control = process_loaded_law_data(df3)
    
    np_Group = 0
    return X1,X2,X3,y1,y2,y3, sa_index, p_Group, np_Group

def create_clients(image_list, label_list, num_clients, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 
def batch_data(data_shard, bs=30):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def create_clients_attr(Xtr1, Ytr1,Xtr2,Ytr2,Xtr3,Ytr3,num_clients,initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''
    clients = {}
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    data = list(zip(Xtr1, Ytr1))
    clients.update({client_names[0] :data})
    data = list(zip(Xtr2, Ytr2))
    clients.update({client_names[1] :data})
    data = list(zip(Xtr3, Ytr3))
    clients.update({client_names[2] :data})
    #data = list(zip(Xtr4, Ytr4))
    #clients.update({client_names[3] :data})

    return clients
def load_dataset(dataset_name, split, num_clients):
    if split == 'random':
        if dataset_name=='adult':
            X,y, sa_index, p_Group, np_Group, x_control= load_adult()
            Y = []
            for i in y:
                if (i == -1):
                    Y.append(0)
                else:
                    Y.append(1)
            Y = np.array(Y)
            
        elif dataset_name=='bank':
            X,y, sa_index, p_Group, np_Group, x_control= load_bank()
            Y = []
            for i in y:
                if (i == -1):
                    Y.append(0)
                else:
                    Y.append(1)
            Y = np.array(Y)
            
        elif dataset_name=='default':
            X,y, sa_index, p_Group, np_Group, x_control= load_default()
            Y = []
            for i in y:
                if (i == 0):
                    Y.append(0)
                else:
                    Y.append(1)
            Y = np.array(Y)
        elif dataset_name=='law':
            
            X,y, sa_index, p_Group, np_Group, x_control= load_law()
            Y = []
            for i in y:
                if (i == 0):
                    Y.append(1)
                else:
                    Y.append(0)
        
            Y = np.array(Y)
        else:
            print("dataset not supported, add the required pre-processing code for this dataset.")
            return
            
        
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
        Xtr = x_train
        Xte = x_test
        Ytr = y_train
        Yte = y_test
        
        clients_test_data = create_clients(Xte, Yte, num_clients=3, initial='client')
        clients = create_clients(Xtr, Ytr, num_clients=3, initial='client')
        clients_test_data_batched = dict()
        for (client_name, data) in clients_test_data.items():
            clients_test_data_batched[client_name] = batch_data(data)
    
        clients_batched = dict()
        for (client_name, data) in clients.items():
            clients_batched[client_name] = batch_data(data)
        #process and batch the test set  
        test_batched = tf.data.Dataset.from_tensor_slices((Xte, Yte)).batch(len(Yte))
        return clients_batched, clients_test_data_batched, test_batched, p_Group, np_Group, sa_index, Xtr
    else:
        if dataset_name=='adult':
            X1,X2,X3,y1,y2,y3, sa_index, p_Group, np_Group = load_adult_attr()
        elif dataset_name=='bank':
            X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group = load_bank_attr()
        elif dataset_name=='default':
            X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group = load_default_attr()
        elif dataset_name=='law':
            X1,X2,X3,y1,y2,y3, sa_index, p_Group, np_Group = load_law_attr()

        # prepare train data and test data of each client
        clients = {}
        client_data_testx = []
        client_data_testy = []
        client_names = ['{}_{}'.format('client', i+1) for i in range(4)]

        x_train, x_test, y_train, y_test = train_test_split(X1,y1,test_size=0.2)
        Xtr1 = x_train
        Xte1 = x_test
        Ytr1 = y_train
        Yte1 = y_test
        Xtr = x_train
        client_data_testx.append(Xte1)
        client_data_testy.append(Yte1)
        ####
        x_train, x_test, y_train, y_test = train_test_split(X2,y2,test_size=0.2)
        Xtr2 = x_train
        Xte2 = x_test
        Ytr2 = y_train
        Yte2 = y_test
        client_data_testx.append(Xte2)
        client_data_testy.append(Yte2)
        ####
        x_train, x_test, y_train, y_test = train_test_split(X3,y3,test_size=0.2)
        Xtr3 = x_train
        Xte3 = x_test
        Ytr3 = y_train
        Yte3 = y_test
        client_data_testx.append(Xte3)
        client_data_testy.append(Yte3)
        ####
        #concatnate teset data
        x_test_new = np.concatenate((client_data_testx[0], client_data_testx[1]), axis=0)
        x_test_new = np.concatenate((x_test_new, client_data_testx[2]), axis=0)
        y_test_new = np.concatenate((client_data_testy[0], client_data_testy[1]), axis=0)
        y_test_new = np.concatenate((y_test_new, client_data_testy[2]), axis=0)
        test_batched = tf.data.Dataset.from_tensor_slices((x_test_new, y_test_new)).batch(len(y_test_new))
        clients = create_clients_attr(Xtr1,Ytr1,Xtr2,Ytr2,Xtr3,Ytr3, num_clients=3, initial='client')
        #process and batch the training data for each client
        clients_batched = dict()
        for (client_name, data) in clients.items():
            clients_batched[client_name] = batch_data(data)
        return clients_batched, client_data_testx, client_data_testy, test_batched, p_Group, np_Group, sa_index, Xtr1
    
        

    
