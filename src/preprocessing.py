'''
collection of functions to perform preprocessing on the data:
- separate data into 0.2 second chunks
- peak to peak
- standard deviation
- skewness
- kurtosis
- RMS
'''
import numpy as np
from scipy.stats import kurtosis, skew
import time
import pandas as pd
import sys

def fast_load(data_path, cols):
    print("Loading data ...")
    start = time.time()
    df = pd.read_csv(data_path, usecols=cols, skiprows=[0,2], sep='\t')
    end = time.time()
    print("Time elasped:", end - start)
    return df

def load(data_path, flag, cols):
    # load data
    print("Loading data...")
    start = time.time()
    if flag: 
        df = pd.read_csv(data_path, parse_dates = [0], date_parser = pd.to_datetime, 
        usecols = cols,
        index_col=0, skiprows=[0,2], sep='\t', nrows=50000)
        #import matplotlib.pyplot as plt
        #normalized_df=(df-df.mean())/df.std()
        #normalized_df.plot()
        #plt.show()

    else:
        df = pd.read_csv(data_path, parse_dates = [0], date_parser = pd.to_datetime, 
        usecols=cols,
        index_col=0, skiprows=[0,2], sep='\t')

    print("Starting on %s" % df.index[0])
    end = time.time()
    print("Time elasped:", end - start)
    return df

def get_peaks(y):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y)
    return len(peaks)

def pk_to_pk(y):
    max_pk = max(y)
    min_pk = min(y)
    return max_pk - min_pk

def rms(y):
    return np.sqrt(np.mean(y**2))

# Input a pandas series 
def ent(data):
    import scipy as sc
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    print(result)
    return result[round(len(result)/2,0):][100]

def transform(dfs, chunk_size = 200):
    '''
    takes in a list of pandas dataframes (dfs)
    and returns the starting time,
    and a new numpy matrix where is column a feature
    made from the above functions.
    '''
    chunk_size = round(len(dfs.index) / chunk_size)
    dfs = np.array_split(dfs, chunk_size)
    print("Getting pk-pk 0 (5), rms 1 (6), kurtosis 2 (7), skew 3 (8), std 4 (9), corr 5 6 7 (8 9 10)")
    num_of_functions = 8
    num_of_features = num_of_functions*len(list(dfs[0])) # num of functions * num of org. cols
    ans = np.zeros(shape=(len(dfs), num_of_features))
    i = 0; j=0 
    for chunk in dfs:
        for col in chunk:
            ans[i,j] = pk_to_pk(chunk[col])
            j+=1
            ans[i,j] = rms(chunk[col])
            j+=1
            ans[i,j] = kurtosis(chunk[col])
            j+=1
            ans[i,j] = skew(chunk[col])
            j+=1
            ans[i,j] = np.std(chunk[col])
            j+=1
            for z in chunk:
                if z != col:
                    ans[i,j] = np.corrcoef(chunk[col], chunk[z])[0,1]
                    j+=1
        i+=1
        j=0

    return ans[~np.isnan(ans).any(axis=1)], dfs[0].index[0]

#def scale(x):
#    sc = MinMaxScaler(feature_range = (0, 1))
#    return sc.fit_transform(x)

def shift_data(x, size = 500):
    '''
    two hundred size is one second 
    '''
    name = list(x)[0]
    for i in range(1, size):
        x[name + '-' + str(i)] = x[name].shift(i)

    x = x.iloc[size:]
    return x[~np.isnan(x).any(axis=1)]
