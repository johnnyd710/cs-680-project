'''
base class model.py
for use with any asset
'''
import pandas as pd
import numpy as np

class model:
    def __init__(self, cols, flag):
        self.data = pd.DataFrame(columns = cols)
        self.labels = pd.Series([])
        self.flag = flag
        self.model = 0

    def plot(self, values, xaxis = 'xaxis', yaxis = 'yaxis', title = 'plot'):
        '''
        values is a dict of values to plot: {'Name1': [2.1, 2.3, ..., 5.7],
        'Name2': [1.2,1.7, ..., 4.5]}
        '''
        import matplotlib.pyplot as plt
        with plt.style.context('ggplot'):
            for key, d in values.items():
                plt.plot(d, label = key)
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            plt.title(title)
            plt.legend()
            plt.show()

    def write(self, objects, name):
        '''
        Writes data to csv file for later editing
        name should be a date or equivalent
        objects should be a list of things to write in one dateframe
        '''
        print('Writing...')
        import csv
        filename = '../Out/'+ name + '.csv'
        data = pd.DataFrame([])
        for obj in objects:
            data = pd.concat([data, obj], axis=1, sort=False)
        data.to_csv(filename, sep=',', index=False)

    def peek(self, n):
        '''
        Prints the top five rows of dataframe
        '''
        print(self.data.head(n))
        print(self.labels[0:n])

    def anomaly_detector(self):
        '''
        call this module before predictions to ensure incoming
        data is valid and normal
        '''
        pass
