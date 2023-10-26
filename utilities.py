import random
from imblearn.over_sampling import SMOTENC
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
def k_nearest_neighbors(data, predict, k):
    distances = []
    count = 0
    for sample in data:
        euclidean_distance = np.linalg.norm(np.array(sample)-np.array(predict))
        distances.append([euclidean_distance,count])
        count+=1
    
    votes = [i[1] for i in sorted(distances)[:k]] ##votes is returning indexes of k random samples

    #vote_result = Counter(votes).most_common(9)[0][0]
    return votes

##algo 2:

def fair_kSMOTE_algo_2(dmajor,dminor,k,r, sa_index):
    S = []
    Ns =  int(r*(len(dmajor)))
    
    Nks = int(Ns / (k-1))
    difference = Ns-Nks*(k-1)
    
    rb = []
    #pick a random k sample from dmin and save them in rb
    dmin_rand = random.sample(dminor, k-1)   
    #for debugging
    sens_attr_vals = []
    i = 0
    
    #do algorithem (choose the nearest neighbor and linear interpolation)
    for xb in dmin_rand:
        N= k_nearest_neighbors(dminor,xb,k) #from minority-p
        
        #do linear interpolation
        Sxb = []
        
        if i==0:
            Nkss = Nks+difference
        else:
            Nkss = Nks
        
        i+=1
        for s in range(Nkss):
            
            
            j = 1 
            #randome k sample
            #j = random.randint(0, len(N))
            
            ##here nearst neghber insted of dminor
            #linear interpolation
            x_new = ((dminor[N[j]]-xb) * random.sample(range(0, 1), 1))
            j+=1
            
            while(j < len(N)):
                #here on random xb
                ind = N[j]
                x_new = x_new + ((dminor[ind]-xb) * random.sample(range(0, 1), 1))
                j += 1
                
            x_new = x_new / (len(N)-1) 
            Synthesized_instance = xb + x_new
            
            
            #for algo 3 when finding nearest neighbors from min_np and assigning the 
            #'p' sensitive value to all synthesized instances
            Synthesized_instance[sa_index] = xb[sa_index] 
            Sxb.append(Synthesized_instance)
        
            
            
            
            
        S.append(Sxb)
    
    
    
    return S



###algo 3, algo 3 x_synthesized as a mixture of nearest neighbors from dmin-p and dmin-np

def fair_kSMOTE(dmajor,dminor_wg,dminor,k,r, sa_index):
    S = []
    #Ns =  int(r*(len(dmajor) - len(dminor)))
    Ns =  int(r*(len(dmajor)))
    
    
    Nks = int(Ns / (k-1))
    difference = Ns-Nks*(k-1)
    #if r==-1:
    #    Nks = 1
    rb = []
    #pick a random k sample from dmin and save them in rb
    dmin_rand = random.sample(dminor, k-1)   
    #for debugging
    sens_attr_vals = []
    
    
    #do algorithem (choose the nearest neighbor and linear interpolation)
    i = 0
    
    for xb in dmin_rand:
        N= k_nearest_neighbors(dminor,xb,int(k/2)+1) #from minority-p
        N2= k_nearest_neighbors(dminor_wg,xb,int(k/2)) #from minority-np
    
        N3 = np.hstack((N, N2))
        if i==0:
            Nkss = Nks+difference
        else:
            Nkss = Nks
        
        i+=1
        #do linear interpolation
        Sxb = []
        
        for s in range(Nkss):
            
            j = 1  
            #randome k sample
            #j = random.randint(0, len(N))
            
            ##here nearst neghber insted of dminor
            #linear interpolation
            x_new = ((dminor[N[j]]-xb) * random.sample(range(0, 1), 1))
            j+=1
            
            while(j < len(N)):
                #here on random xb
                ind = N[j]
                
                x_new = x_new + ((dminor[ind]-xb) * random.sample(range(0, 1), 1))
                j += 1
            j = 0
            while(j < len(N2)):
                #here on random xb
                ind = N2[j]
                
                x_new = x_new + ((dminor_wg[ind]-xb) * random.sample(range(0, 1), 1))
                j += 1    
            x_new = x_new / (len(N3)-1) 
            Synthesized_instance = xb + x_new 
            
            
            #for algo 3 when finding nearest neighbors from min_np and assigning the 
            #'p' sensitive value to all synthesized instances
            Synthesized_instance[sa_index] = xb[sa_index] 
            Sxb.append(Synthesized_instance)
        
            
            
            
            
        S.append(Sxb)
    
    
   
    return S

def splitYtrain(Xtr,Ytr,minority_lable):
    #print(Ytr)
    dmaj_x = []
    dmin_x = []
    
    
    for i in range(len(Ytr)):
        if((Ytr[i]) == minority_lable):
            dmin_x.append(Xtr[i])
            
        else:
            dmaj_x.append(Xtr[i])
            
    
    return dmaj_x,dmin_x
def splitYtrain_sa_value(Xtr,Ytr,minority_lable,majority_label,pp_Group,npp_Group, sa_index): #splite Ytrain based on sensitive attribute value
    #print(Ytr)
    dmaj_p_x = []
    dmaj_np_x = []
    dmin_p_x = []
    dmin_np_x = []
    
    
    for i in range(len(Ytr)):
        if((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==pp_Group): #select minority instances with "protected" value 
            dmin_p_x.append(Xtr[i])
        elif((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==npp_Group): #select minority instances with "protected" value 
            dmin_np_x.append(Xtr[i])
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==pp_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_p_x.append(Xtr[i])
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==npp_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_np_x.append(Xtr[i])
    
    return dmin_p_x, dmin_np_x, dmaj_p_x, dmaj_np_x
def get_statistics(Xtr,Ytr,minority_lable,majority_label): #splite Ytrain based on sensitive attribute value
    #print(Ytr)
    dmaj_p_x =0
    dmaj_np_x = 0
    dmin_p_x = 0
    dmin_np_x = 0
    
    
    for i in range(len(Ytr)):
        if((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==p_Group): #select minority instances with "protected" value 
            dmin_p_x+=1
        elif((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==np_Group): #select minority instances with "protected" value 
            dmin_np_x+=1
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==p_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_p_x+=1
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==np_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_np_x+=1
    
    return dmin_p_x, dmin_np_x, dmaj_p_x, dmaj_np_x
def create_synth_data(clinet_traning_x, clinet_traning_y, minority_lable,majority_label,k,r,group,pp_group,npp_group, sa_index):
    
    
    
    #create two data set from traning data (one for maj (ex.0 class) and one for min(ex.1 class)) 
    #for simple federated learning
    dmaj_client,dmin_client = splitYtrain(clinet_traning_x,clinet_traning_y,minority_lable)
    
    #for fair federated learning
    
    
    dmin_p_x, dmin_np_x, dmaj_p_x, dmaj_np_x = splitYtrain_sa_value(clinet_traning_x,clinet_traning_y,minority_lable,majority_label, pp_group,npp_group, sa_index)
    
    group_names = ['dmin_p_x', 'dmin_np_x', 'dmaj_p_x', 'dmaj_np_x']
    group_label_dict = {'dmin_p_x':minority_lable, 'dmin_np_x': minority_lable, 'dmaj_p_x':  majority_label, 'dmaj_np_x': majority_label}
    group_dict = {'dmin_p_x':dmin_p_x, 'dmin_np_x': dmin_np_x, 'dmaj_p_x':  dmaj_p_x, 'dmaj_np_x': dmaj_np_x}
    
    group_lengths = [len(dmin_p_x),len(dmin_np_x), len(dmaj_p_x), len(dmaj_np_x)] 
    
    #group with maximum length
    max_length_group = group_lengths.index(max(group_lengths))  
    
    #group name which has maximum length
    max_group_name = group_names[max_length_group]
    
    #find the insances in the group with maximum length and store them in dmaj_x
    for key, value in group_dict.items():
        if key== max_group_name:
            dmaj_x = value
            break
    Xtr_new = []
    Ytr_new = []  
    
    
    
    ##Algo 3:
    
     ##Algo 3:
    if group =='min_p':
        dmaj_x = dmaj_p_x
        dmin_x = dmin_p_x
        
        #dmin_np = dmin_np_x
        x_syn = fair_kSMOTE(dmaj_x,dmin_np_x,dmin_x,k,r, sa_index)
        #x_syn = fair_kSMOTE_algo_2(dmaj_x,dmin_x,k,r)
        # add the created synthatic data to the traning data
        # here merrge old traning data with the new synthatic data
        new_label = minority_lable
        for j in x_syn:
            for s in j:
                Xtr_new.append(s)
                Ytr_new.append(new_label)
        
    elif group =='maj_np':
        dmaj_x = dmin_np_x
        dmin_x = dmaj_np_x
        #dmin_np = dmin_np_x
        x_syn = fair_kSMOTE_algo_2(dmaj_x,dmin_x,k,r, sa_index)
        
        
        # add the created synthatic data to the traning data
        # here merrge old traning data with the new synthatic data
        new_label = majority_label
        for j in x_syn:
            for s in j:
                Xtr_new.append(s)
                Ytr_new.append(new_label)
    
    
    
    return Xtr_new,Ytr_new
def k_nearest_neighbors_modified(data, predict, k,data_rand_index):
    distances = []
    count=0
    for sample in data:
        if count in data_rand_index:
            count+=1
        else:
            euclidean_distance = np.linalg.norm(np.array(sample)-np.array(predict))
            distances.append([euclidean_distance,count])
            count+=1
    votes = [i[1] for i in sorted(distances)[:k]] ##votes is returning indexes of k random samples
    #vote_result = Counter(votes).most_common(9)[0][0]
    return votes
def downsample_utility_function(clinet_traning_x, clinet_traning_y,data,data_index, k,r,label,reduction_amount, sa_index):
    S = []
    #pick a random k sample from data which we want to downsample
    d_rand = []
    d_rand_index = []
    
    #reduction_amount = int(r*len(data))  
    #print("reduction amount: %s" % reduction_amount)
    
    for i in range(reduction_amount):
        index = random.randint(0, (len(data)-1))
        #random.sample(range(0,len(data)),1)
        
        d_rand.append(data[index])
        d_rand_index.append(index)
    data_rand_index = list(d_rand_index)
    Sxb = []   
    sens_attr_vals = []
    k = 3 ##number of nearest neighbours
    indices_neighbours_removed = list(d_rand_index)
    #print(len(d_rand_index))
    #do algorithm (choose the nearest neighbor and linear interpolation)
    
    Nks=1
    S=[]
    #do algorithem (choose the nearest neighbor and linear interpolation)
    for xb in d_rand:
        #N= k_nearest_neighbors(data,xb,k)
        N= k_nearest_neighbors_modified(data,xb,k,data_rand_index)
        
        #do linear interpolation
        Sxb = []
        for s in range(Nks):
            j = 0 #j = 1
            #linear interpolation
            x_new = ((data[N[j]]-xb) * random.sample(range(0, 1), 1))
            indices_neighbours_removed.append(N[1])
            j+=1
            
            while(j < len(N)-1):
                #here on random xb
                ind = N[j]
                x_new = x_new + ((data[ind]-xb) * random.sample(range(0, 1), 1))
                j += 1
                
            x_new = x_new / (len(N)-1) 
            Synthesized_instance = xb + x_new 
            Synthesized_instance[sa_index] = xb[sa_index] 
            
            Sxb.append(Synthesized_instance)
            
            
            
            
        S.append(Sxb)
    #print(indices_neighbours_removed)
    
    
    for i in range(len(indices_neighbours_removed)):
        indx = indices_neighbours_removed[i]
        indices_neighbours_removed[i] = data_index[indx]
    indices = np.unique(indices_neighbours_removed)
    indices = list(indices)
    difference = (reduction_amount*2)-len(indices)
    
        
    #for i in range(0,difference-1):
    #    indices.pop(0)
    #print("reduced amount: %s" % len(indices))
    
    indices.sort()
    
    Xtr = []
    Ytr = []
    for k in clinet_traning_x:
        Xtr.append(k)
        
    for k in clinet_traning_y:
        Ytr.append(k)
    
    for i in range(len(indices)):
        indx = indices[i]-i
        Xtr.pop(indx)
        Ytr.pop(indx)
    
    for j in S:
            for s in j:
                Xtr.append(s)
                Ytr.append(label)
       
    Xtr = np.array(Xtr)
    Ytr = np.array(Ytr)
    return Xtr, Ytr
def splitYtrain_sa_value_index(Xtr,Ytr,minority_lable,majority_label,pp_Group,npp_Group, sa_index): #split data based on sensitive attribute value
    #print(Ytr)
    dmaj_p_x = []
    dmaj_p_index = [] ##index of maj_p instance in the main dataset
    
    dmaj_np_x = []
    dmaj_np_index = [] ##index of maj_np instance in the main dataset
    
    dmin_p_x = []
    dmin_p_index = [] ##index of min_p instance in the main dataset
    
    dmin_np_x = []
    dmin_np_index = [] ##index of min_np instance in the main dataset
    
    
    for i in range(len(Ytr)):
        if((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==pp_Group): #select minority instances with "protected" value 
            dmin_p_x.append(Xtr[i])
            dmin_p_index.append(i)
        elif((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==npp_Group): #select minority instances with "protected" value 
            dmin_np_x.append(Xtr[i])
            dmin_np_index.append(i)
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==pp_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_p_x.append(Xtr[i])
            dmaj_p_index.append(i)
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==npp_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_np_x.append(Xtr[i])
            dmaj_np_index.append(i)
    
    return dmin_p_x, dmin_np_x, dmaj_p_x, dmaj_np_x, dmin_p_index, dmin_np_index, dmaj_p_index, dmaj_np_index
def splitYtrain_sa_value_index(Xtr,Ytr,minority_lable,majority_label,pp_Group,npp_Group, sa_index): #split data based on sensitive attribute value
    #print(Ytr)
    dmaj_p_x = []
    dmaj_p_index = [] ##index of maj_p instance in the main dataset
    
    dmaj_np_x = []
    dmaj_np_index = [] ##index of maj_np instance in the main dataset
    
    dmin_p_x = []
    dmin_p_index = [] ##index of min_p instance in the main dataset
    
    dmin_np_x = []
    dmin_np_index = [] ##index of min_np instance in the main dataset
    
    
    for i in range(len(Ytr)):
        if((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==pp_Group): #select minority instances with "protected" value 
            dmin_p_x.append(Xtr[i])
            dmin_p_index.append(i)
        elif((Ytr[i]) == minority_lable and (Xtr[i][sa_index])==npp_Group): #select minority instances with "protected" value 
            dmin_np_x.append(Xtr[i])
            dmin_np_index.append(i)
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==pp_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_p_x.append(Xtr[i])
            dmaj_p_index.append(i)
        elif ((Ytr[i]) == majority_label and (Xtr[i][sa_index])==npp_Group): #select minority(positive class) instances with "non-protected" value
            dmaj_np_x.append(Xtr[i])
            dmaj_np_index.append(i)
    
    return dmin_p_x, dmin_np_x, dmaj_p_x, dmaj_np_x, dmin_p_index, dmin_np_index, dmaj_p_index, dmaj_np_index

def downsample(clinet_traning_x, clinet_traning_y, minority_lable,majority_label,k,r,group,pp_group,npp_group, sa_index):
    
    #group: maj_p, maj_np, min_p, min_np
    
    dmaj_client,dmin_client = splitYtrain(clinet_traning_x,clinet_traning_y,minority_lable)
   
    #for fair federated learning
    
    dmin_p_x, dmin_np_x, dmaj_p_x, dmaj_np_x, dmin_p_index, dmin_np_index, dmaj_p_index, dmaj_np_index = splitYtrain_sa_value_index(clinet_traning_x,clinet_traning_y,minority_lable,majority_label,pp_group,npp_group, sa_index)
    
    ##Algo 5:
    if group =='min_np':
        
        label = minority_lable
        data = dmin_np_x
        reduction_amount = int(r*len(data))
        #print("reduction amount: %s" % reduction_amount)
        x_small,y_small = downsample_utility_function(clinet_traning_x, clinet_traning_y,data,dmin_np_index, k,r,label,reduction_amount, sa_index)
        
    elif group =='maj_p':
        label = majority_label
        data = dmaj_p_x
        reduction_amount = int(r*len(data))
        x_small,y_small = downsample_utility_function(clinet_traning_x, clinet_traning_y,data,dmaj_p_index, k,r,label,reduction_amount, sa_index)
        
    return x_small,y_small



def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

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

from sklearn.metrics import f1_score
def test_client_model(X_test, Y_test,  model,pp_Group,npp_Group,  sa_index, fairness_notion):
    
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
    
    '''
    for i in range(len(preidect)):
        if preidect[i]==1:
            assigned_positive_labels+=1
        if Y_test[i]==1:
            total_positive_labels+=1
    '''    
    #eqop = find_eqop_score(X_test,Y_test,preidect, pp_Group,npp_Group) 
    #print('eqop: {}'.format(eqop))
    #return eqop, assigned_positive_labels, total_positive_labels, BalanceACC
    #print('stat_parity: {}'.format(disc_score))
    
    return disc_score, assigned_positive_labels, total_positive_labels, BalanceACC


def find_class_Weight(labels,majority_label,minority_label):
    unique, counts = np.unique(labels, return_counts=True)
    count_ap_dict = dict(zip(unique, counts))
    
    majority_class_weight = 1
    minority_class_weight = count_ap_dict.get(majority_label,0)/count_ap_dict.get(minority_label,1)
    class_weights={majority_label:1,minority_label:minority_class_weight}
    
    return class_weights

def test_model(X_test,Y_test, pp_Group, npp_Group,  sa_index,  model, comm_round):
    cce = tf.keras.losses.BinaryCrossentropy()
    logits = model.predict(X_test)
    preidect = np.around(logits)
    preidect = np.nan_to_num(preidect)
    Y_test = np.nan_to_num(Y_test)
    stat_parity = find_statistical_parity_score(X_test,Y_test,preidect, pp_Group, npp_Group,  sa_index)
    eqop = find_eqop_score(X_test,Y_test,preidect, pp_Group, npp_Group,  sa_index)    
        
    conf = (confusion_matrix(Y_test,preidect))   
    loss = cce(Y_test, preidect)
    acc = accuracy_score(preidect,Y_test)
    print('comm_round: {} | global_acc: {} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss,conf,stat_parity, eqop

