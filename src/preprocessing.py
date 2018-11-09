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

def pk_to_pk(y):
    max_pk = max(y)
    min_pk = min(y)
    return max_pk - min_pk

def rms(y):
    return np.sqrt(np.mean(y**2))

def transform(dfs, chunk_size = 200):
    '''
    takes in a list of pandas dataframes (dfs)
    and returns the starting time,
    and a new numpy matrix where is column a feature
    made from the above functions.
    '''
    dfs = np.array_split(dfs, chunk_size)
    print("size of chunk:", len(dfs[0].index))
    print("Getting pk-to-pk 0 (5), rms 1 (6), kurtosis 2 (7), skew 3 (8), std 4 (9) for each")
    num_of_functions = 5
    num_of_features = num_of_functions*len(list(dfs[0])) # num of functions * num of org. cols
    ans = np.zeros(shape=(len(dfs), num_of_features))
    i = 0; j=0 
    for chunk in dfs:
        for col in chunk:
            #ans[i,j] = pk_to_pk(chunk[col])
            #j+=1
            ans[i,j] = rms(chunk[col])
            j+=1
            #ans[i,j] = kurtosis(chunk[col])
            #j+=1
            #ans[i,j] = skew(chunk[col])
            #j+=1
            ans[i,j] = np.std(chunk[col])
            j+=1
        i+=1
        j=0

    return ans, dfs[0].index[0]

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

    return x.iloc[size:]

