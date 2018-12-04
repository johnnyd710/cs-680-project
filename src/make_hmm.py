'''
Loads data, trains som.py
'''
import sys
from somtf import SOM
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from plots import gen_plots, som_map, score
import sys

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

def select_vars(train, df1, df2, df3, df_failure, sel, just_corr):
    # standardize selected vars except for correlations
    if not just_corr:
        sel_not_corr = [n for n in sel if "corr" not in lookup(n)]
        sel_corr = [n for n in sel if "corr" in lookup(n)]
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        ntrain = sc.fit_transform(train[:,sel_not_corr])
        ndf1 = sc.transform(df1[:,sel_not_corr])
        ndf2 = sc.transform(df2[:,sel_not_corr])
        ndf3 = sc.transform(df3[:,sel_not_corr])
        ndf_failure = sc.transform(df_failure[:,sel_not_corr])

        train = np.concatenate((ntrain, train[:,sel]), axis=1)
        df1 = np.concatenate((ndf1, df1[:,sel]), axis=1)
        df2 = np.concatenate((ndf2, df2[:,sel]), axis=1)
        df3 = np.concatenate((ndf3, df3[:,sel]), axis=1)
        df_failure = np.concatenate((ndf_failure, df_failure[:,sel]), axis=1)

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
    #gen_plots(train, df1, df2, df3, df_failure)
    # model 1 [14,15,5]
    #sel = [29,24,22,21,30,20] # 27,4,5]
    sel = range(0, train.shape[1])
    just_corr = False
    train, df1, df2, df3, df_failure = select_vars(train, 
                                            df1, df2, df3, df_failure, sel, just_corr)
    m = [9,7]
    som = SOM(m = m,dim =  train.shape[1], n_iterations=1000)
    if train_on == 'failure': som.train(df_failure)
    elif train_on == 'train': som.train(train)
    elif train_on == 'df1': som.train(df1)
    elif train_on == 'df2': som.train(df2)
    elif train_on == 'df3': som.train(df3)

    centroids = som.get_centroids()
    print("Training done...")
    lc = som_map(som, m, train, centroids)
    threshold = 1
    score(som, train, df1, df2, df3, df_failure, 'mean', lc, threshold, 2)
    #score(som, train, df1, df2, df3, df_failure, 'std', lc, threshold, 2)

if __name__ == "__main__":
    main(sys.argv[1])