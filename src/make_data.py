'''
py make_training_data.py <path to data> <name to call file> <autoencoder or som> <other parameters>
'''

import numpy as np 
from preprocessing import fast_load, shift_data, transform
import sys
from sklearn.preprocessing import MinMaxScaler
import csv

def main(datapath, name, model, parameter):   
    #sc = MinMaxScaler(feature_range = (0, 1))
    df = fast_load(datapath, ['F1S F1SFIBV Overall', 'F1S F1SFOBV Overall', 'F1S F1SMOBV Overall', 'F1S F1SMIBV Overall']) # = "../data/Furnace 1 2-7-17 0056 to 1300.txt" # 'F1S F1SMIBV Overall'
    if model == 'autoencoder':
        df = shift_data(df, parameter)
    elif model == 'som':
        df, st = transform(df, parameter)
    else:
        print("please input either autoencoder or som")
    #df = sc.fit_transform(df)

    print('saving to file...')

    myFile = open('../data/'+name+'-data-'+ model + '.csv', 'w', newline='')  
    with myFile:  
        writer = csv.writer(myFile)
        writer.writerows(df)

    return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))