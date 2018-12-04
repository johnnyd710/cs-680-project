'''
Loads data, trains som.py
'''
import sys
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import gen_plots, som_map, score
import sys
import matplotlib
import json

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

def create_label(a, val):
    s = a.shape[0]
    if val == 0: b = np.zeros(s)
    else: b = np.ones(s)
    return b.reshape(1,-1)


def main(train_on):
    
    feb7, jan1, jan16, jan7, failure = get_training_data()
    #ntrain, ndf1, ndf2, ndf3, ndf_failure = get_training_data_north()
    gen_plots(feb7, jan1, jan16, jan7, failure)

    label_feb7=create_label(feb7, 0)    
    label_jan16=create_label(jan16, 0)    
    label_failure=create_label(failure, 1)    

    clf = LogisticRegression(random_state=0, solver='lbfgs', n_jobs=-1)
    if train_on == 'all':
        train = np.concatenate((feb7, jan16, failure))
        labels = np.concatenate((label_feb7, label_jan16, label_failure))
        clf.fit(train, labels)
    
    pred = clf.predict(feb7)
    print("feb 7th")
    print(clf.score(pred, label_feb7))
    plt.plot(pred)
    plt.show()

    pred = clf.predict(failure)
    print("failure")
    print(clf.score(pred, label_failure))
    plt.plot(pred)
    plt.show()

    pred = clf.predict(jan16)
    print("jan 16th")
    print(clf.score(pred, label_jan16))
    plt.plot(pred)
    plt.show()

    pred = clf.predict(feb7)
    print("test jan 1st")
    plt.plot(pred)
    plt.show()

    pred = clf.predict(jan7)
    print("test jan 7th")
    plt.plot(pred)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])