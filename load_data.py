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
from load_adult_data import *
from load_bank_data import *
from load_law_data import *
from load_default_data import *
# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)


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
            X,y, sa_index, p_Group, np_Group, x_control, features, sensitive_feature= load_adult()
            Y = []
            for i in y:
                if (i == -1):
                    Y.append(0)
                else:
                    Y.append(1)
            Y = np.array(Y)
            
        elif dataset_name=='bank':
            X,y, sa_index, p_Group, np_Group, x_control, features, sensitive_feature= load_bank()
            Y = []
            for i in y:
                if (i == -1):
                    Y.append(0)
                else:
                    Y.append(1)
            Y = np.array(Y)
            
        elif dataset_name=='default':
            X,y, sa_index, p_Group, np_Group, x_control, features, sensitive_feature= load_default()
            Y = []
            for i in y:
                if (i == 0):
                    Y.append(0)
                else:
                    Y.append(1)
            Y = np.array(Y)
        elif dataset_name=='law':
            
            X,y, sa_index, p_Group, np_Group, x_control, features, sensitive_feature= load_law()
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
        return clients_batched, clients_test_data_batched, test_batched, p_Group, np_Group, sa_index, Xtr, features, sensitive_feature
    else:
        if dataset_name=='adult':
            X1,X2,X3,y1,y2,y3, sa_index, p_Group, np_Group, features, sensitive_feature = load_adult_attr()
        elif dataset_name=='bank':
            X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group, features, sensitive_feature = load_bank_attr()
        elif dataset_name=='default':
            X1,X2,X3,X4,y1,y2,y3,y4, sa_index, p_Group, np_Group, features, sensitive_feature = load_default_attr()
        elif dataset_name=='law':
            X1,X2,X3,y1,y2,y3, sa_index, p_Group, np_Group, features, sensitive_feature = load_law_attr()

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
        return clients_batched, client_data_testx, client_data_testy, test_batched, p_Group, np_Group, sa_index, Xtr1, features, sensitive_feature
    
        

    
