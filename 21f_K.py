from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

data_table=pd.read_csv("21f_2.csv",  low_memory=False )

#print(data_table.head())

col_list=['PHY Date Week Num','PHY to TH', 'Box Office','Physical Date','Physical W1 Units']

data_table=data_table.loc[:, col_list]


dataset=data_table.copy()

dataset.rename(index=str, columns={'Box Office':'boxoffice','Physical W1 Units':'phy_w1_units', 'Physical Date':'phy_release_date' }, inplace=True)



print(dataset.shape)


#setting up split datasets for the models

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)

train.info()

train.describe()

test.info()

test.describe()




# Pulling out columns for X  and Y in train set
train_data = train.drop(['phy_w1_units','phy_release_date'], axis=1).values
#X_training = train.drop('phy_release_date', axis=1).values
train_labels = train[['phy_w1_units']].values


# Pulling out columns for X  and Y in test sett
test_data = test.drop(['phy_w1_units','phy_release_date'], axis=1).values
test_labels = test[['phy_w1_units']].values











baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(3,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()



baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=200,
                                      batch_size=10,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


plot_history([('baseline', baseline_history),
              #('smaller', smaller_history),
              #('bigger', bigger_history)
             ])