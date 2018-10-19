from som import SOM
from sklearn import datasets

iris = datasets.load_iris() 
X = iris.data
print(X[1:4])

som = SOM(x = 3, y = 3, input_dim=4, learning_rate=0.5, num_iter = 100, radius = 1.0)
som.train(X)
print(som._centroid_matrix)
print(som._weights_list)
print(som._locations)
print(som.map_input(X[1:4]))