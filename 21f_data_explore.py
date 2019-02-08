
from __future__ import absolute_import, division, print_function


import pandas as pd
import webbrowser
import os

import  numpy as  np


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from numpy import linspace
from sklearn import metrics

from tensorflow.contrib.learn import *


import keras
from  keras import *
import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

print(tf.__version__)



#print(tf.VERSION)



#col_names=['PHY Date Week Num','PHY to TH','Studio', 'Rating', 'Box Office' ]


data_table=pd.read_csv("21f_1.csv",  low_memory=False )

#print(data_table.head())

col_list=['PHY Date Week Num','PHY to TH','Studio', 'Rating', 'Box Office']

data_table=data_table.loc[:, col_list]


dataset=data_table.copy()

#print(dataset.tail())




dataset=pd.get_dummies(data=dataset, columns=['Studio'])


dataset=pd.get_dummies(data=dataset, columns=['Rating'])




print(dataset.tail())

# encoded = to_categorical(data)

print(dataset.isna().sum())



'''

features_df= pd.get_dummies (df,columns=['Studio'])
del features_df['Physical W1 Units']




X=features_df.as_matrix()
y= df['Physical W1 Units'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)




# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)


# Save the trained model to a file so we can use it in other programs
joblib.dump(model, '21f_trained_model_test_1.pkl')





html=data_table[0:100].to_html()

with open('data.html', 'w') as f:
    f.write(html)

full_filename=os.path.abspath('data.html')
webbrowser.open('file://{}'.format(full_filename))


'''
