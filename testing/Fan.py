import pandas as pd
import numpy as np
import datetime
from model import model
import time

class Fan(model):

    def __init__(self, cols, test, id = 'NA'):
        super().__init__(cols, test)
        self.id = id
        self.cols = cols
        self.pieces_of_steel = []
        self.folds = pd.Series([])
        self.means = pd.DataFrame([])
        print("Fan Model Loaded, ID: %s" % id)

    def load(self,data_path):
        # load data
        print("Loading data...")
        start = time.time()
        if self.flag: 
            df = pd.read_csv(data_path, parse_dates = [0], date_parser = pd.to_datetime, 
            usecols = self.cols,
            index_col=0, skiprows=[0,2], sep='\t', nrows=50000)
            import matplotlib.pyplot as plt
            normalized_df=(df-df.mean())/df.std()
            normalized_df.plot()

            plt.show()

        else:
            df = pd.read_csv(data_path, parse_dates = [0], date_parser = pd.to_datetime, 
            usecols=self.cols,
            index_col=0, skiprows=[0,2], sep='\t')

        self.data = df

        print("Starting on %s" % df.index[0])
        end = time.time()
        print("Time elasped:", end - start)

    def vib_analysis(self, v):
        '''
        Returns the RMS, x1, x2 and x3 of the vibration signals
        for each piece of steel
        '''
        v = v - np.mean(v)
        V = v*np.kaiser(v.size, 10) # hanning
        # FFT algorithm
        Y = np.fft.fft(V)
        N = int((Y.size /2)+1) # array size

        T = 2 / 1000 # two milliseconds to seconds
        sampling_rate = (1.0/T) # inverse of the sampling rate

        # Nyquist Sampling Criteria
        x = np.linspace(0.0, sampling_rate/2, N)
        y = 2.0*np.abs(Y[:N])/N

        flag = True
        import matplotlib.pyplot as plt
        # Plotting the results
        if flag:
            plt.plot(x, y)  # 1 / N is a normalization factor
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Vibration (g)')
            plt.title('Frequency Domain (Healthy Machinery)')
            plt.show()


    def cluster_plot(self):
        '''
        Returns several 3D plots of the data with folds/false folds labelled
        Number of plots is number of features/3
        '''
        pass