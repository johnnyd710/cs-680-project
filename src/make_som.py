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
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def read_csv(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            record = line.rstrip().split(',')
            features = [float(n) for n in record]
            yield features

def get_dataset(f):
    generator = lambda: read_csv(f)
    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((n_features,), ()))

def main():
    # load training data
    train = np.loadtxt('../data/training-data-som.csv')
    #df_healthy = np.loadtxt("../data/testing-data-som.csv")
    #df_failure = np.loadtxt("../data/failure-data-som.csv")

    print("som for data starting at ", start_time)

    som = SOM(m = l,dim =  3, n_iterations=400)
    som.train(train)
    # Get output grid
    image_grid = som.get_centroids()
    print(image_grid)
    image_grid =np.zeros(20*30*3).reshape((20,30,3))
    plt.imshow(image_grid)
    # Map colours to their closest neurons
    mapped = som.map_vects(df)
    print(mapped)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()

    exit()

    
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