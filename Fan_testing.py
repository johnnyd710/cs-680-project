'''
author: john dimatteo
created: Jun-15-18

example of usage: python3 PythonScripts/Fan_testing.py < flag >
-- flag: can be 1 or 0. 1 for debugging.
'''
variables = ['time','F1S F1SFIBV Overall', 'F1S F1SFOBV Overall', 'F1S F1SMIBV Overall', 'F1S F1SMOBV Overall',
     'F1S F1NFIBV Overall', 'F1S F1NFOBV Overall', 'F1S F1NMIBV Overall', 'F1S F1NMOBV Overall', 'F1S North Fan Impeller Side Bearing Temp',
      'F1S North Fan Motor Side Bearing Temp', 'F1S South Fan Motor Side Bearing Temp', 'F1S South Fan Impeller Side Bearing Temp']

data_path = '/home/johnny/Code/MachineLearning/Data/Fan/Furnance-1-1730-2300.txt'
# pd.concat(list_of_dataframes)
import Fan
import numpy as np
import pandas as pd

df = Fan.Fan(variables, 1, 'sept 21')
df.load(data_path)

