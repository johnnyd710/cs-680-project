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
from sklearn.preprocessing import MinMaxScaler

def pk_to_pk(y):
    max_pk = max(y)
    min_pk = min(y)
    return max_pk - min_pk

def rms(y):
    return np.sqrt(np.mean(y**2))

def transform(dfs):
    '''
    takes in a list of pandas dataframes (dfs)
    and returns the starting time,
    and a new numpy matrix where is column a feature
    made from the above functions.
    '''
    num_of_functions = 5
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
        i+=1
        j=0

    sc = MinMaxScaler(feature_range = (0, 1))
    ans = sc.fit_transform(ans)
    return ans, dfs[0].index[0]
