import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from somtf import SOM
from plots import som_map
from make_som import select_vars
from preprocessing import fast_load
import matplotlib
import csv

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})


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


# start of main

datapath = '/home/johnny/code/arcelormittal-project/data/Furnace 1 1-1-17 0056 to 0400.txt'
rawfeb7 = fast_load(datapath, ['F1S F1SFIBV Overall', 
                            'F1S F1SFOBV Overall']) 

datapath = '/home/johnny/code/arcelormittal-project/data/Furnace 1 2-7-17 0056 to 1300.txt'
rawjan1 = fast_load(datapath, ['F1S F1SFIBV Overall', 
                            'F1S F1SFOBV Overall'])     

datapath = '/home/johnny/code/arcelormittal-project/data/Furnace 1 1-7-17 0000 to 1100.txt'
rawjan7 = fast_load(datapath, ['F1S F1SFIBV Overall', 
                            'F1S F1SFOBV Overall']) 

datapath = '/home/johnny/code/arcelormittal-project/data/Furnace 1 1730 to 2300.txt'
rawjan14 = fast_load(datapath, ['F1S F1SFIBV Overall', 
                            'F1S F1SFOBV Overall']) 

threshold = 1

plt.figure(figsize=(15,10))
rawjan14['F1S F1SFIBV Overall'].plot(color = 'black', label = 'Vibration Signal')
plt.plot([threshold]*len(rawjan14.index), linestyle='--', label = 'Threshold', color = 'grey')
plt.title('Raw Vibration Signal January 14th')
plt.legend()
plt.ylabel("g RMS")
plt.xlabel("Data Point")
plt.savefig('../figs/raw-jan14.png')
plt.close()

plt.figure(figsize=(15,10))
rawjan1['F1S F1SFIBV Overall'].plot(color = 'black', label = 'Vibration Signal')
plt.plot([threshold]*len(rawjan1.index), linestyle='--', label = 'Threshold', color = 'grey')
plt.title('Raw Vibration Signal January 1st')
plt.legend()
plt.ylabel("g RMS")
plt.xlabel("Data Point")
plt.savefig('../figs/raw-jan1.png')
plt.close()

plt.figure(figsize=(15,10))
rawjan7['F1S F1SFIBV Overall'].plot(color = 'black', label = 'Vibration Signal')
plt.plot([threshold]*len(rawjan7.index), linestyle='--', label = 'Threshold', color = 'grey')
plt.title('Raw Vibration Signal January 7th')
plt.legend()
plt.ylabel("g RMS")
plt.xlabel("Data Point")
plt.savefig('../figs/raw-jan7.png')
plt.close()

plt.figure(figsize=(15,10))
rawfeb7['F1S F1SFIBV Overall'].plot(color = 'black', label = 'Vibration Signal')
plt.plot([threshold]*len(rawfeb7.index), linestyle='--', label = 'Threshold', color = 'grey')
plt.title('Raw Vibration Signal February 7th')
plt.legend()
plt.ylabel("g RMS")
plt.xlabel("Data Point")
plt.savefig('../figs/raw-feb7.png')
plt.close()

data = []

tmp = []
tmp.append('jan-14th')
#s = rawjan14['F1S F1SFIBV Overall']
s = rawjan14['F1S F1SFIBV Overall'].rolling(window=2000).std().dropna()
time_above = np.sum(s > threshold) / len(rawjan14.index)
tmp.append(time_above)
time_above = np.sum(rawjan14['F1S F1SFOBV Overall'] > threshold) / len(rawjan14.index)
tmp.append(time_above)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)

tmp = []
tmp.append('feb-7th')
s = rawfeb7['F1S F1SFIBV Overall'].rolling(window=2000).std().dropna()
time_above = np.sum(s > threshold) / len(rawfeb7.index)
tmp.append(time_above)
time_above = np.sum(rawfeb7['F1S F1SFOBV Overall'] > threshold) / len(rawfeb7.index)
tmp.append(time_above)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)

tmp = []
tmp.append('jan-1st')
s = rawjan1['F1S F1SFIBV Overall'].rolling(window=2000).std().dropna()
time_above = np.sum(s > threshold) / len(rawjan1.index)
tmp.append(time_above)
time_above = np.sum(rawjan1['F1S F1SFOBV Overall'] > threshold) / len(rawjan1.index)
tmp.append(time_above)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)

tmp = []
tmp.append('jan-7th')
s = rawjan7['F1S F1SFIBV Overall'].rolling(window=2000).std().dropna()
time_above = np.sum(s > threshold) / len(rawjan7.index)
tmp.append(time_above)
time_above = np.sum(rawjan7['F1S F1SFOBV Overall'] > threshold) / len(rawjan7.index)
tmp.append(time_above)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)

feb7, jan1, jan16, jan7, jan14 = get_training_data()

sel = range(0, feb7.shape[1])
print(feb7.shape[0])
just_corr = False
feb7, jan1, jan16, jan7, jan14 = select_vars(feb7, 
                                        jan1, jan16, jan7, jan14, sel, just_corr)

m = [8,8]
som = SOM(m = m,dim =  feb7.shape[1], n_iterations=10000)
som.train(feb7)
centroids = som.get_centroids()


lc = som_map(som, m, feb7, centroids)
scores = pd.DataFrame(som.health_score(feb7, 3, lc, 1))
scores = np.array(scores.rolling(window=5).mean().dropna())
threshold = 35
tmp = []
tmp.append('feb-7th-health')
time_above = np.sum(scores > threshold) / len(scores)
tmp.append(time_above)
s = list(scores)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)


lc = som_map(som, m, feb7, centroids)
scores = pd.DataFrame(som.health_score(jan1, 3, lc, 1))
scores = np.array(scores.rolling(window=5).mean().dropna())
threshold = 35
tmp = []
tmp.append('jan-1st-health')
time_above = np.sum(scores > threshold) / len(scores)
tmp.append(time_above)
s = list(scores)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)


lc = som_map(som, m, feb7, centroids)
scores = pd.DataFrame(som.health_score(jan7, 3, lc, 1))
scores = np.array(scores.rolling(window=5).mean().dropna())
threshold = 35
tmp = []
tmp.append('jan-7th-health')
time_above = np.sum(scores > threshold) / len(scores)
tmp.append(time_above)
s = list(scores)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)


lc = som_map(som, m, feb7, centroids)
scores = pd.DataFrame(som.health_score(jan14, 3, lc, 1))
scores = np.array(scores.rolling(window=5).mean().dropna())
threshold = 35
tmp = []
tmp.append('jan-14th-health')
time_above = np.sum(scores > threshold) / len(scores)
tmp.append(time_above)
s = list(scores)
df = pd.DataFrame({'s':s})
df['next_s'] = df.s.shift(-1)
crossed = ((df.next_s > threshold) & (df.s <= threshold))
tmp.append(np.sum(crossed))
tmp.append(np.mean(s))
data.append(tmp)

with open('output.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)

csvFile.close()
