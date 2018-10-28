'''
Loads data, trains som.py
'''
import time
import numpy as np
import pandas as pd
from somtf import SOM
import preprocessing
import sompy
import matplotlib.pyplot as plt
from score import health_score

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

def main():
    df = load("../data/Furnace 1 1-16-17 0000 to 1100.txt", True, ['time','F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']) # inboard and outboard vibrations
    chunk_size = len(df.index)/1000     # number of chunks
    df = np.array_split(df, chunk_size)
    df, start_time = preprocessing.transform(df)
    print("som for data starting at ", start_time)
    #som = SOM(5,5,10,0.5,0.5)
    #som.train(df)
    #print(som.map_input(df[0]))

    mapsize = [10,10]
    som = sompy.SOMFactory.build(df, mapsize, mask=None, mapshape='planar', 
        lattice='rect', normalization='var', initialization='pca', 
        neighborhood='gaussian', training='batch', name='sompy') 
    # this will use the default parameters, but i can change the initialization and neighborhood methods
    som.train(n_job=1, verbose=None)  
    # verbose='debug' will print more, and verbose=None wont print anything
    sompy.mapview.View2DPacked(300, 300, 'test').show(som)
    #first you can do clustering. Currently only K-means on top of the trained som
    v = sompy.mapview.View2DPacked(2, 2, 'test',text_size=8)  
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

    print(health_score(som.codebook.matrix, df))

if __name__== "__main__":
    main()