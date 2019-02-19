from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

data_table=pd.read_csv("21f_2.csv",  low_memory=False )

#print(data_table.head())

col_list=['PHY Date Week Num',
          'PHY to TH', 'Box Office',
          #'Physical Date',
          'Physical W1 Units'
          ]

data_table=data_table.loc[:, col_list]


dataset=data_table.copy()

dataset.rename(index=str, columns={'Box Office':'boxoffice','Physical W1 Units':'phy_w1_units', 'Physical Date':'phy_release_date' }, inplace=True)





print(dataset.shape)




 
tf.keras.utils.normalize(
    dataset,
    axis=-1,
    order=1

)





#setting up split datasets for the models

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)

#train.info()

print('here is:', train.describe())

#test.info()

#test.describe()

#print(test.head())





# Pulling out columns for X  and Y in train set
X_training = train.drop(['phy_w1_units',
                         #'phy_release_date',
                         #'phy_release_dayofweek',
                         #'phy_release_quarter',
                         #'PHY Date Week Num',
                         'PHY to TH'
                         ], axis=1).values


#X_training = train.drop('phy_release_date', axis=1).values
Y_training = train[['phy_w1_units']].values







#verify succeful selection

print(X_training.shape)
print(Y_training.shape)


# set up a data scaler to a range of 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))


# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)





print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))






print (X_scaled_training.shape)


print(X_scaled_training.dim)
