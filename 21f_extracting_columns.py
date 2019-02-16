'''from __future__ import absolute_import, division, print_function

import pandas as pd
import webbrowser
import os

import warnings
from tqdm import tqdm
from datetime import datetime
import json

warnings.filterwarnings("ignore")

import numpy as  np

import seaborn as sns
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorboard

from sklearn.preprocessing import MinMaxScaler
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from numpy import linspace
from sklearn import metrics

from tensorflow.contrib.learn import *

import keras
from keras import *
import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf

# from tensorflow import keras
# from tensorflow.keras import layers




print(tf.__version__)

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# print(tf.VERSION)


# col_names=['PHY Date Week Num','PHY to TH','Studio', 'Rating', 'Box Office' ]

'''

import pandas as pd


data_table = pd.read_csv("21f_data_phy_only.csv", low_memory=False)

print( data_table.columns)


data_table = data_table.drop(['Title'],axis=1 )

dataset = data_table.copy()






# transormation for Rating and Studio (now removed from dataset and muted)

dataset=pd.get_dummies(data=dataset, columns=['Studio'])


dataset=pd.get_dummies(data=dataset, columns=['Rating'])


dataset=pd.get_dummies(data=dataset, columns=['PHY Date Week Num'])



dataset.rename(index=str, columns={'Box Office': 'boxoffice',
                                   'Physical W1 Units': 'phy_w1_units',
                                   'Physical Date': 'phy_release_date'
                                   }, inplace=True)



dataset['phy_release_date']=pd.to_datetime(dataset['phy_release_date'])





#releaseDate = pd.to_datetime(dataset['phy_release_date'])
#dataset['phy_release_dayofweek']=releaseDate.dt.dayofweek
#dataset['phy_release_quarter']=releaseDate.dt.quarter









#dataset.info()

#print(dataset.describe)

#print(dataset.isna().sum())







# split train and test from dataset

train = dataset.sample(frac=0.8, random_state=200)
test = dataset.drop(train.index)

train.info()

train.describe()