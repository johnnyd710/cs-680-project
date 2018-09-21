import pandas as pd
import numpy as np
import datetime
from model import model

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
        if self.flag: 
            df = pd.read_csv(data_path, parse_dates = [0], date_parser = pd.to_datetime, 
            usecols = self.cols,
            index_col=0, skiprows=[0,2], sep='\t', nrows=10000)

        else:
            df = pd.read_csv(data_path, parse_dates = [0], date_parser = pd.to_datetime, 
            usecols=self.cols,
            index_col=0, skiprows=[0,2], sep='\t')


        print("Starting on %s" % df.index[0])

        import matplotlib.pyplot as plt
        normalized_df=(df-df.mean())/df.std()
        normalized_df.plot()

        plt.show()
        exit()

        # only focus on lead folds for now...
        #data_labels['Fold'] = np.where((data_labels['Fold Present Yes/No'] == 'Yes') & (data_labels["Fold Location   ID/OD"] == "ID"), 'Yes', data_labels["Fold Present Yes/No"])
        data_labels['Fold'] = np.where((data_labels['Fold Present Yes/No'] == 'Yes') & (data_labels["Fold Location   ID/OD"] == "OD"), 'tail', data_labels['Fold Present Yes/No'])
        data_labels['Fold'] = np.where((data_labels['Fold Present Yes/No'] != 'Yes') & (data_labels["Fold Present Yes/No"] != "No"), 'not labelled', data_labels['Fold'])

        data_labels = data_labels.set_index("GEN_EST")
        #labels = data_labels['Fold Present Yes/No']
        labels = data_labels['Fold']

        print("Number of folds in dataset: %d " % np.sum(np.where(labels == 'Yes',1,0)))
        print("Number of false folds in dataset: %d " % np.sum(np.where(labels == 'No',1,0)))

        self.labels = self.labels.append(labels)
        self.data = self.data.append(new_data)

    def steel_in_mandrel(self):
        '''
        Returns a list (pieces_of_steel) with each element being
        a dataframe representing data for a piece of steel (p)
        going in the mandrel
        Returns a list of failures corresponding with each piece of
        steel (0 is healthy, 1 is fold, 2 is false fold)
        '''
        data, flag, labels = self.data, self.flag, self.labels
        pieces_of_steel = []
        tmp_index = []
        tmp=0
        #head_tracker = data['C1 HEAD TRACKER'].values
        head_tracker = data['TDC_HeadTailTracker'].values        
        #data['C1 HEAD TRACKER'] = head_tracker/140
        #data[0:210000].plot()
        #plt.show()
        #del data['C1 HEAD TRACKER']
        del data['TDC_HeadTailTracker']
        index_start = 0
        index_end = 0
        tmp=0 # 0 for tenth, 1 for ninth
        while True:
            datum = np.argmax(head_tracker == 0)
            start = datum + np.argmax(head_tracker[datum:] > 105)
            end = start + 2000
            notgood = True
            if tmp:
                status = 'y'
                tmp = 0
            elif tmp == 0:
                status='next'
                tmp=1
            while notgood:
                keep=True
                #super().plot({'Head Tracker': head_tracker[start:end]}, 'Time' + data.index[index_end + start], 'Metres', 'Head Tracker')
            #    status = input("Good? (y/n/next)\n")
                if status == 'n':
                    change_start = int(input("Change start by how much? (+/-)"))
                    start = start + change_start
                    change_end = int(input("Change end by how much? (+/-)"))
                    end = end + change_end
                elif status == 'y':
                    #print('keep')
                    notgood = False
                elif status == 'next':
                    #print('skip')
                    keep = False
                    notgood = False
                else:
                    print("Invalid. Please enter y or n or next")
                print(datum,start,end)
            index_start = index_end + start
            index_end += start + (end - start)
            if keep:
                #print(data.index[index_start], data.index[index_end])
                p = data.iloc[index_start:index_end]
                pieces_of_steel.append(p)
                #super().plot({'Current': p['C1 Mandrel Current Feedback'].reset_index(drop=True)}, 'Time', 'Metres', 'Head Tracker')
            head_tracker = head_tracker[end:]
            if start == 0: break

        print("Number of pieces of steel: %d" % len(pieces_of_steel))

        self.pieces_of_steel = pieces_of_steel

    def make_labels(self):
        '''
        now I need to align the failures with the features
        keep a list of all times for each piece and look for a failure then assign to a 1 for entire piece
         labeling. Locate where pieces//failures happened
        '''
        from datetime import datetime
        folds = np.zeros(len(self.pieces_of_steel))
        for index, label in self.labels.items(): # iterate through the rows [GEN_EST = index, FOLD PRESENT]
            n=0
            fold_time =  index
            start = datetime.strptime(self.pieces_of_steel[0].index[0], '%d.%m.%Y %H:%M:%S.%f')
            finish = datetime.strptime(self.pieces_of_steel[len(self.pieces_of_steel)-1].index[len(self.pieces_of_steel[len(self.pieces_of_steel)-1].index)-1], '%d.%m.%Y %H:%M:%S.%f')
            if fold_time < start:
                continue
            elif fold_time > finish:
                continue
            for i, steel in enumerate(self.pieces_of_steel,0):
                if i==0:
                    continue

                last_piece_time_end = datetime.strptime(self.pieces_of_steel[i-1].index[len(steel.index)-1], '%d.%m.%Y %H:%M:%S.%f')
                piece_time_end = datetime.strptime(steel.index[len(steel.index)-1], '%d.%m.%Y %H:%M:%S.%f')
                #print("between", last_piece_time_end, piece_time_end)
                mask = (fold_time > last_piece_time_end) and (fold_time < piece_time_end)
                if mask and (label == 'Yes'):
                    folds[n] += 1
                    print("fold @",fold_time, "placed between: ", last_piece_time_end, piece_time_end)
                elif mask and (label == 'No'):
                    folds[n] += 2
                    print("false fold @",fold_time, "placed between: ", last_piece_time_end, piece_time_end)
                elif mask and (label == 'not labelled'):
                    folds[n] += 3
                    print("unlabeled fold @",fold_time, "placed between: ", last_piece_time_end, piece_time_end)
                n+=1
        self.folds = pd.Series(folds)

    def get_features_mean(self):
        '''
        Returns a dataframe (mean) whose rows contain the mean of
        each piece of steel (p) going through the Fan
        and the columns are all of the sensors
        '''
        data, flag = self.data, self.flag

        #pieces_of_steel, self.labels = Fan.steel_in_mandrel(self)

        labels = self.folds
        pieces_of_steel = self.pieces_of_steel

        timestamp = []

        mean = pd.DataFrame(index = range(0,len(pieces_of_steel)))
        mean_piece = np.zeros(len(pieces_of_steel))
        max = pd.DataFrame(index = range(0,len(pieces_of_steel)))
        max_piece = np.zeros(len(pieces_of_steel))
        s_dev = pd.DataFrame(index = range(0,len(pieces_of_steel)))
        s_dev_piece = np.zeros(len(pieces_of_steel))
        #time = pd.DataFrame(index = range(0,len(pieces_of_steel)))
        #time_piece = np.zeros(len(pieces_of_steel))
        vars = list(pieces_of_steel[0])
        interval = 2 # 2 milliseconds between datapoints
        flag=True

        for i in range(0, len(pieces_of_steel)):
            timestamp.append(pieces_of_steel[i].index[0])
            if flag and (labels[i]==1):
                super().plot({name:pieces_of_steel[i][name].reset_index(drop=True) for name in vars}, 'Starting at ' + timestamp[i], 'Unit', 'Fold')
            elif flag and (labels[i]==2): #flag
                super().plot({name:pieces_of_steel[i][name].reset_index(drop=True) for name in vars}, 'Starting at ' + timestamp[i], 'Unit', 'False Fold')
            elif flag and (labels[i]==3): #flag
                super().plot({name:pieces_of_steel[i][name].reset_index(drop=True) for name in vars}, 'Starting at ' + timestamp[i], 'Unit', 'Unlabeled Fold')
            elif flag: #flag
                super().plot({name:pieces_of_steel[i][name].reset_index(drop=True) for name in vars}, 'Starting at ' + timestamp[i], 'Unit', 'No Folds or False Folds')

        for var in vars:
            for i in range(0, len(pieces_of_steel)):
                mean_piece[i] = pieces_of_steel[i][var].mean()
                max_piece[i] = np.abs(pieces_of_steel[i][var]).max()
                s_dev_piece[i] = pieces_of_steel[i][var].std()
                #time_piece[i] = len(pieces_of_steel[i][var]) * interval / 1000 # milliseconds to seconds
            print("competed variable: ", var)
            #Fan.vib_analysis(self,pieces_of_steel[i][var].values)
            mean['Mean ' + var] = mean_piece
            max['Max ' + var] = max_piece
            s_dev['Variance ' + var] = s_dev_piece
            #time['Time ' + var] = time_piece
            timestamp = pd.DataFrame(timestamp)

        #self.data = pd.concat([mean,max,s_dev], axis=1, sort = False)[:-1]
        self.means = pd.concat([timestamp, mean,max,s_dev], axis=1)

    def clean_signal(s):
        s_new = []
        for i in range(0, len(s)-1):
            while s[i] == s[i+1]:
                i+=1
                continue
            print('keep')
            s_new.append(s[i])
        return s_new

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

        print(y[1])

        flag = True
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