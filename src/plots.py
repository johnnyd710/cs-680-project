import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns
from preprocessing import load
import numpy as np

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

def score(som, train, df1, df2, df3, df_failure, method, lc, threshold=0.2, t=1):
    #from scipy.signal import savgol_filter
    import pandas as pd
    plt.style.use('ggplot')
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 18})

    scores = pd.DataFrame(som.health_score(train, 3, lc, t))
    x_axis = range(0, len(scores.rolling(window=5).mean().dropna()))
    if method == 'mean': plt.plot(x_axis, scores.rolling(window=5).mean().dropna(), label = 'trained', color = 'black')
    elif method == 'std': plt.plot(scores.rolling(window=20).std(), label = 'trained')
    plt.plot(x_axis, [threshold]*len(x_axis), linestyle='--', label = 'threshold', color = 'grey')
    plt.xlabel('Data Point')
    plt.ylabel('Health Score')
    plt.title('SOM Health Score February 7th')
    plt.legend()
    plt.savefig('../figs/health_score-feb7.png')
    plt.show()
    plt.figure(figsize=(15,10))

    scores = pd.DataFrame(som.health_score(df1, 3, lc, t))
    x_axis = range(0, len(scores.rolling(window=5).mean().dropna()))
    if method == 'mean': plt.plot(x_axis, scores.rolling(window=5).mean().dropna(), label = '1-1', color = 'black')
    elif method == 'std': plt.plot(scores.rolling(window=20).std(), label = '1-1')
    plt.plot(x_axis, [threshold]*len(x_axis), linestyle='--', label = 'threshold', color = 'grey')
    plt.xlabel('Data Point')
    plt.ylabel('Health Score')
    plt.title('SOM Health Score January 1st')
    plt.legend()
    plt.savefig('../figs/health_score-jan1.png')
    plt.figure(figsize=(15,10))

    scores = pd.DataFrame(som.health_score(df_failure, 3, lc, t))
    x_axis = range(0, len(scores.rolling(window=5).mean().dropna()))
    if method == 'mean': plt.plot(x_axis, scores.rolling(window=5).mean().dropna(), label = 'failure', color = 'black')
    elif method == 'std': plt.plot(scores.rolling(window=20).std(), label = 'failure')
    plt.plot(x_axis, [threshold]*len(x_axis), linestyle='--', label = 'threshold', color = 'grey')
    plt.xlabel('Data Point')
    plt.ylabel('Health Score')
    plt.title('SOM Health Score January 14th')
    plt.legend()
    plt.savefig('../figs/health_score-jan14.png')
    plt.figure(figsize=(15,10))

    scores = pd.DataFrame(som.health_score(df3, 3, lc, t))
    x_axis = range(0, len(scores.rolling(window=5).mean().dropna()))
    if method == 'mean': plt.plot(x_axis, scores.rolling(window=5).mean().dropna(), label = '1-7', color = 'black')
    elif method == 'std': plt.plot(scores.rolling(window=20).std(), label = '1-7')
    plt.plot(x_axis, [threshold]*len(x_axis), linestyle='--', label = 'threshold', color = 'grey')
    plt.xlabel('Data Point')
    plt.ylabel('Health Score')
    plt.title('SOM Health Score January 7th')
    plt.legend()
    plt.savefig('../figs/health_score-jan7.png')

def som_map(som, m, d, centroids):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    pca = PCA(n_components=3)
    weights = np.zeros([m[0]*m[1], d.shape[1]])
    i=0
    keys = []
    for key, val in centroids.items():
        keys.append(key) # need this for later
        weights[i] = val
        i+=1

    colours = pca.fit_transform(weights)
    sc = MinMaxScaler()
    colours = sc.fit_transform(colours)
    # Map inputs to their closest neurons
    count = {key: 0 for key in keys}
    mapped = som.map_vects(d)
    for bmu in mapped:
        count[str(bmu)] += 1

    list_count = [val for key, val in count.items()]
    from matplotlib import patches as patches

    # plot the rectangles
    fig = plt.figure(figsize=(10,10))
    # setup axes
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, m[0]))
    ax.set_ylim((0, m[1]))
    ax.set_title('Self-Organising Map after %d iterations' % 100)

    i=0
    for x in range(0, m[0]):
        for y in range(0, m[1]):
            ax.add_patch(patches.Rectangle((x, y), 1, 1,
                        facecolor=np.round(colours[i],2),
                        edgecolor='none'))
            ax.text(x+0.5, y+0.5, count[str(np.array([x, y]))], style='italic', color='white')
            i+=1

    plt.savefig('../figs/som-colour-map.png')

    plt.show()

    return list_count

def gen_plots(train, df1, df2, df3, df_failure):
    dfs = [train, df1, df2, df3, df_failure]
    dataset_labels = ['2-7', '1-1', '1-16', '1-7', '1-14 (failure)']

    for col in range(0, train.shape[1]):
        plt.figure(figsize=(15,10))
        sns.distplot(train[:,col], color = 'blue', kde=False, label='2-7')
        sns.distplot(df1[:,col], color = 'green', kde=False, label='1-1')
        sns.distplot(df2[:,col], color = 'purple', kde=False, label='1-16')
        sns.distplot(df3[:,col], color = 'yellow', kde=False, label='1-7')
        sns.distplot(df_failure[:,col], color = 'red', kde=False, label='unhealthy')
        plt.legend()
        plt.xlabel('Range')
        plt.ylabel('Values')
        plt.title(lookup(col) + str(col))
        plt.savefig('../figs/dist-'+lookup(col)+'.png')

        
    for col in range(0, train.shape[1]):
        mean = [np.mean(d[:,col]) for d in dfs]
        error = [np.std(d[:,col]) for d in dfs]
        fig, ax = plt.subplots(figsize=(15,10))
        ax.bar(np.arange(len(mean)), mean, yerr=error, align='center', alpha=0.5, 
                                    ecolor='black', capsize=25)
        ax.set_ylabel(col)
        ax.set_xticks(np.arange(len(mean)))
        ax.set_xticklabels(dataset_labels)
        ax.set_title('Range of Variable ' + lookup(col))
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('../figs/' + 'bar-' + lookup(col) + str(col) + '.png')
            
        #plt.show()
    

def dist(df1, df2, col):
    '''
    Plot distribution of variables for
    healthy vs unhealthy.
    df1 = healthy
    df2 = unhealthy
    '''
    sns.distplot(df1[:,col], color = 'blue', kde=False, label='healthy')
    sns.distplot(df2[:,col], color = 'red', kde=False, label='unhealthy')
    plt.legend()
    plt.title(col)
    plt.show()

def dist_before(df1, df2, col):
    '''
    Plot distribution of variables for
    healthy vs unhealthy.
    df1 = healthy
    df2 = unhealthy
    '''
    sns.distplot(df1[col], color = 'blue', kde=False, label='healthy')
    sns.distplot(df2[col], color = 'red', kde=False, label='unhealthy')
    plt.legend()
    plt.title(col)
    plt.show()

def plot_all_dist():
    '''
    Plot distribution of variables for all datasets
    '''
    dfs=[]
    variables = ['time','F1S F1SMIBV Overall', 'F1S F1SMOBV Overall']#'F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']
    colours = ['red', 'blue', 'green', 'orange', 'yellow', 'pink']
    dfs.append(load("../data/Furnace 1 1-1-17 0056 to 0400.txt", False, variables))
    dfs.append(load("../data/Furnace 1 1-7-17 0000 to 1100.txt", False, variables))
    dfs.append(load("../data/Furnace 1 1-16-17 0000 to 1100.txt", False, variables))
    dfs.append(load("../data/Furnace 1 2-7-17 0056 to 1300.txt", False, variables))
    dfs.append(load("../data/Furnace 1 1730 to 2300.txt", False, variables))
    variables = [var for var in variables if var != 'time']
    dataset_labels = ['1-1-17', '1-7-17', '1-16-17', 
                        '2-7-17', 'failure']
    #dfs = [scale(x) for x in dfs]

    import scipy.stats as stats

    for i, col in enumerate(variables):
        for n, dataset in enumerate(dfs):
            # remove outliers:
            #z = np.abs(stats.zscore(dataset))
            #dataset = dataset.loc[:, (dataset != dataset.iloc[0]).any()] 
            #dataset = dataset[(z < 2).all(axis=1)]
            sns.distplot(dataset[col], norm_hist = False, color = colours[n], kde=True, label=dataset_labels[n])
        plt.legend()
        plt.title(col)
        plt.show()

    for i, col in enumerate(variables):
        mean = [np.mean(d[col].values) for d in dfs]
        error = [np.std(d[col].values) for d in dfs]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(mean)), mean, yerr=error, align='center', alpha=0.5, 
                                    ecolor='black', capsize=25)
        ax.set_ylabel(col)
        ax.set_xticks(np.arange(len(mean)))
        ax.set_xticklabels(dataset_labels)
        ax.set_title('Range of Variable ' + col)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('../figs/' + col + '-bar.png')
        plt.show()

def autoencoder_plot(X, y, label, init = 0):
    dist = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        dist[i] = np.linalg.norm(x-y[i]) # euclidean distance

    if init:
        plt.figure(figsize=(10,7))
        plt.xlabel('Index')
        plt.ylabel('Score')
        #plt.xlim((30000,50000))
        plt.title("Outlier Score")

    plt.plot(dist, label = label)
