'''
Loads data, trains som.py
'''
import time
import numpy as np
import pandas as pd
from somtf import SOM
from preprocessing import *
import sompy
import matplotlib.pyplot as plt
from score import health_score
from plots import *
from sklearn.preprocessing import MinMaxScaler

def main():

    sc = MinMaxScaler(feature_range = (0, 1))
    plot_all_dist()
    exit()
    df = load("../data/Furnace 1 2-7-17 0056 to 1300.txt", True, 
                    ['time', 'F1S F1SMIBV Overall', 'F1S F1SMOBV Overall', 'F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']) # inboard and outboard vibrations
    df, start_time = transform(df)
    df = sc.fit_transform(df)

    df2 = load("../data/Furnace 1 1-1-17 0056 to 0400.txt", True, 
                    ['time','F1S F1SMIBV Overall', 'F1S F1SMOBV Overall', 'F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']) # inboard and outboard vibrations
    #print(df_healthy[1:25])
    #print(pd.Timedelta(df_healthy.index[0] - df_healthy.index[1]).seconds / 3600.0)
    #print(list(df_healthy))
    df2, start_time = transform(df2)
    df2 = sc.transform(df2)

    #print("som for data starting at ", start_time)
    df_failure = load("../data/Furnace 1 1730 to 2300.txt", True, 
        ['time','F1S F1SMIBV Overall', 'F1S F1SMOBV Overall','F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']) # inboard and outboard vibrations
    #print(list(df_failure))
    df_failure, start_time = transform(df_failure)
    df_failure = sc.transform(df_failure)
    print("som for data starting at ", start_time)
    #som = SOM(5,5,10,0.5,0.5)
    #som.train(df)
    #print(som.map_input(df[0]))

    #for i in range(0,df_failure.shape[1]):
    #    dist(df_healthy, df_failure, i)

    mapsize = [9,7]
    som = sompy.SOMFactory.build(df, mapsize, mask=None, mapshape='planar', 
        lattice='rect', normalization='var', initialization='pca', 
        neighborhood='gaussian', training='batch', name='sompy') 
    # this will use the default parameters, but i can change the initialization and neighborhood methods
    som.train(n_job=1, verbose=None)  
    # verbose='debug' will print more, and verbose=None wont print anything
    sompy.mapview.View2DPacked(30, 30, 'test').show(som)
    #first you can do clustering. Currently only K-means on top of the trained som
    v = sompy.mapview.View2DPacked(10, 10, 'test',text_size=8)  
    cl = som.cluster(n_clusters=4)
    # print cl
    print(getattr(som, 'cluster_labels'))
    v.show(som, what='cluster')
    u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)

    #This is the Umat value
    UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)

    #Here you have Umatrix plus its render
    u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)

    from sompy.visualization.bmuhits import BmuHitsView
    #sm.codebook.lattice="rect"
    vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
    vhts.show(som, anotate=True, onlyzeros=False, labelsize=7, cmap="autumn", logaritmic=False)
    plt.show()
    print(som.codebook.matrix)

    plt.plot(health_score(som.codebook.matrix, df), label = 'healthy')
    plt.plot(health_score(som.codebook.matrix, df2), label = 'healthy2')
    plt.plot(health_score(som.codebook.matrix, df_failure), label = 'day of failure')
    plt.legend()
    plt.title('health score')
    plt.show()

if __name__== "__main__":
    main()