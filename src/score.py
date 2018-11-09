'''
given a model (.pkl) and a test observation, 
calculate the health score
'''
from sklearn.neighbors import NearestNeighbors


def health_score(neurons, test_points):
    '''
    given bmu's (or neurons), 
    calculate the health score,
    by using kNN clustering 
    and then the euclidean distance metric
    '''
    print("Classifying...")
    #print(neurons)
    clf = NearestNeighbors(n_neighbors=3)
    clf.fit(neurons)
    distances, indices = clf.kneighbors(test_points)
    return distances.min(1)