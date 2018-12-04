'''
Loads data, trains som.py
'''
import sys
from somtf import SOM
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import gen_plots, som_map, score
import sys
import matplotlib
import json

def js_r(filename):
   with open(filename) as f_in:
       return(json.load(f_in))

def create_dict(neurons, size):
    '''
    given a bmu, returns a severity level
    '''
    #count = {key: 0 for key in range(0, size[0] * size[1])}
    count = {}
    for i in neurons:
        count[str(i)] = 0

    for i in neurons:
        count[str(i)] += 1
    

    return count

def map_to_dict(neurons):
    severity  = {'Very High': 1.0,
                'High': 0.75,
                'Medium': 0.50,
                'Warning': 0.25,
                'Normal': 0}
    
    mapping = js_r('mapping.json')

    print([str(i) for i in neurons])

    levels = [severity[mapping[str(i)]] for i in neurons]

    return levels

def lookup(num):
    var_list = ['Bearing-In-Vib', 'Bearing-Out-Vib', 'Motor-In-Vib', 'Motor-Out-Vib']
    if num < 8: var = var_list[0]
    elif (num > 8) and (num < 16): var = var_list[1]
    elif (num > 16) and (num < 24): var = var_list[2]
    else: var = var_list[3]
        
    num = num % 8
    name_lookup = {'0': 'pk-to-pk', '1': 'rms', '2': 'kurtosis', '3': 'skew', '4': 'standard-deviation'}
    #name_lookup = {'0': 'entropy', '1': 'no-peaks', '2': 'highest-autocorr', '3': 'skew', '4': 'standard-deviation'}
    name_lookup = {k: v + '-' + var for (k, v) in name_lookup.items()}
    
    n=0
    for name in var_list:
        if name != var:
            name_lookup.update({str(n+5):'corr-' + var + '-' + name})
            n+=1
            
    return name_lookup[str(num)]

def get_training_data():
    train = np.loadtxt('../data/training-data-som.csv', delimiter = ',')
    df1 = np.loadtxt("../data/testing-data-som.csv", delimiter = ',')
    df2 = np.loadtxt("../data/testingtwo-data-som.csv", delimiter = ',')
    df3 = np.loadtxt("../data/testingthree-data-som.csv", delimiter = ',')
    df_failure = np.loadtxt("../data/failure-data-som.csv", delimiter = ',')
    return train, df1, df2, df3, df_failure

def get_training_data_north():
    train = np.loadtxt('../data/training-north-data-som.csv', delimiter = ',')
    df1 = np.loadtxt("../data/testing-north-data-som.csv", delimiter = ',')
    df2 = np.loadtxt("../data/testingtwo-north-data-som.csv", delimiter = ',')
    df3 = np.loadtxt("../data/testingthree-north-data-som.csv", delimiter = ',')
    df_failure = np.loadtxt("../data/failure-north-data-som.csv", delimiter = ',')
    return train, df1, df2, df3, df_failure

def select_vars(train, df1, df2, df3, df_failure, sel, just_corr):
    # standardize selected vars except for correlations
    if not just_corr:
        sel_not_corr = [n for n in sel if "corr" not in lookup(n)]
        sel_corr = [n for n in sel if "corr" in lookup(n)]
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        strain = sc.fit_transform(train[:,sel_not_corr])
        sdf1 = sc.transform(df1[:,sel_not_corr])
        sdf2 = sc.transform(df2[:,sel_not_corr])
        sdf3 = sc.transform(df3[:,sel_not_corr])
        sdf_failure = sc.transform(df_failure[:,sel_not_corr])

        train = np.concatenate((strain, train[:,sel]), axis=1)
        df1 = np.concatenate((sdf1, df1[:,sel]), axis=1)
        df2 = np.concatenate((sdf2, df2[:,sel]), axis=1)
        df3 = np.concatenate((sdf3, df3[:,sel]), axis=1)
        df_failure = np.concatenate((sdf_failure, df_failure[:,sel]), axis=1)

        return train, df1, df2, df3, df_failure
        
    if just_corr:
        train = train[:,sel]
        df1 = df1[:,sel]    
        df2 = df2[:,sel]    
        df3 = df3[:,sel]    
        df_failure = df_failure[:,sel] 
    
    return train, df1, df2, df3, df_failure

def main(train_on):
    
    train, df1, df2, df3, df_failure = get_training_data()
    #ntrain, ndf1, ndf2, ndf3, ndf_failure = get_training_data_north()
    #gen_plots(train, df1, df2, df3, df_failure)
    # model 1 [14,15,5]
    #sel = [29,24,22,21,30,20] # 27,4,5]
    sel = range(0, train.shape[1])
    print(train.shape[0])
    #sel = [14,15,5]
    just_corr = False
    train, df1, df2, df3, df_failure = select_vars(train, 
                                            df1, df2, df3, df_failure, sel, just_corr)
    #ntrain, ndf1, ndf2, ndf3, ndf_failure = select_vars(ntrain, 
    #                                        ndf1, ndf2, ndf3, ndf_failure, sel, just_corr)
    m = [8,8]
    som = SOM(m = m,dim =  train.shape[1], n_iterations=10000)

    if train_on == 'failure': som.train(df_failure)
    elif train_on == 'train': som.train(train)
    elif train_on == 'df1': som.train(df1)
    elif train_on == 'df2': som.train(df2)
    elif train_on == 'df3': som.train(df3)
    elif train_on == 'all': som.train(np.concatenate((train)))

    centroids = som.get_centroids()
    print("Training done...")

    lc = som_map(som, m, train, centroids)

    
    threshold = 30
    score(som, train, df1, df2, df3, df_failure, 'mean', lc, threshold, 1)

    #lc = som_map(som, m, train, centroids)
    #score(som, ntrain, ndf1, ndf2, ndf3, ndf_failure, 'mean', lc, threshold, 1)
    #score(som, train, df1, df2, df3, df_failure, 'std', lc, threshold, 2)



if __name__ == "__main__":
    main(sys.argv[1])
