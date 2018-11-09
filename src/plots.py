import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load
import numpy as np

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
