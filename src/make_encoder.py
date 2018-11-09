'''
author: john dimatteo
created on: Nov 7th 2018
'''
import time
import numpy as np
from preprocessing import *
import matplotlib.pyplot as plt
from plots import *
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import csv
import tensorflow as tf

def build(X, encoding_dim, filename, epochs = 100):
        # this is the size of our encoded representations
    size_of_input = X.shape[1]  
    print('size of input layer', size_of_input)

    # this is our input placeholder
    input = Input(shape=(size_of_input,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(size_of_input, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input, outputs=decoded)  

    # this model maps an input to its encoded representation
    encoder = Model(inputs=input, outputs=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input)) 

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    my_training_batch_generator = get_dataset(filename)

    autoencoder.fit_generator(generator=my_training_batch_generator,
                                            epochs=epochs,
                                            verbose=1,
                                            use_multiprocessing=True,
                                            workers=16,
                                            max_queue_size=32,
                                            shuffle=True)

    return encoder, decoder

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

def main(path):
     
    print("autoencoder for data")

    encoder, decoder = build(df, 20, 10)

    # make healthy predictions
    encoded = encoder.predict(df)
    decoded = decoder.predict(encoded)

    # plot score
    autoencoder_plot(df, decoded, 'feb 7th', 1)

    df2 = load("../data/Furnace 1 1-1-17 0056 to 0400.txt", False, 
                    ['time','F1S F1SMIBV Overall']) #'F1S F1SMIBV Overall', 'F1S F1SMOBV Overall', 'F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']) # inboard and outboard vibrations
    df2 = shift_data(df2)
    df2 = sc.transform(df2)

    # make healthy predictions part 2
    encoded = encoder.predict(df2)
    decoded = decoder.predict(encoded)

    # plot score
    autoencoder_plot(df2, decoded, 'jan 1st')

    #print("som for data starting at ", start_time)
    df_failure = load("../data/Furnace 1 1730 to 2300.txt", False, 
        ['time','F1S F1SMIBV Overall']) #'F1S F1SMIBV Overall', 'F1S F1SMOBV Overall','F1S F1SFIBV Overall', 'F1S F1SFOBV Overall']) # inboard and outboard vibrations
    df_failure = shift_data(df_failure)
    df_failure = sc.transform(df_failure)

    # make healthy predictions
    encoded = encoder.predict(df_failure)
    decoded = decoder.predict(encoded)

    # plot score
    autoencoder_plot(df_failure, decoded, 'day of failure')
    
    plt.savefig("ae-outlier-training.png", format="png")
    plt.legend()
    plt.show()



if __name__== "__main__":
    main(sys.argv[1])