
from __future__ import absolute_import, division, print_function


import pandas as pd
import webbrowser
import os


import warnings
from tqdm import tqdm
from datetime import datetime
import json
warnings.filterwarnings("ignore")

import  numpy as  np

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
from  keras import *
import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

print(tf.__version__)

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#print(tf.VERSION)



#col_names=['PHY Date Week Num','PHY to TH','Studio', 'Rating', 'Box Office' ]


data_table=pd.read_csv("21f_2.csv",  low_memory=False )

#print(data_table.head())

col_list=['PHY Date Week Num','PHY to TH', 'Box Office','Physical Date','Physical W1 Units']

data_table=data_table.loc[:, col_list]


dataset=data_table.copy()

#print(dataset.tail())



#transormation for Rating and Studio (now removed from dataset and muted)

#dataset=pd.get_dummies(data=dataset, columns=['Studio'])


#dataset=pd.get_dummies(data=dataset, columns=['Rating'])


dataset.rename(index=str, columns={'Box Office':'boxoffice','Physical W1 Units':'phy_w1_units', 'Physical Date':'phy_release_date' }, inplace=True)


dataset.info()


print(dataset.describe)

dataset['phy_release_date']=pd.to_datetime(dataset['phy_release_date'])

# encoded = to_categorical(data)

print(dataset.isna().sum())

releaseDate = pd.to_datetime(dataset['phy_release_date'])
dataset['phy_release_dayofweek']=releaseDate.dt.dayofweek
dataset['phy_release_quarter']=releaseDate.dt.quarter

dataset.info()

print(dataset.columns)



dataset.phy_release_date.describe()


#sns.jointplot(x='boxoffice',  y= 'phy_w1_units',data=dataset, height=8, ratio=4, color='r' )
#plt.show()


#sns.jointplot(x='PHY Date Week Num',  y= 'phy_w1_units',data=dataset, height=9, ratio=4, color='r' )
#plt.show()


#sns.distplot(dataset.boxoffice)
#dataset.plot.hist()

# plt.figure(figsize=(20,12))


#sns.countplot(dataset['PHY Date Week Num'].sort_values())
#plt.title('movie release count by week',fontsize=20)
#loc, labels=plt.xticks()
#plt.xticks(fontsize=12,rotation=90)
#plt.show()



#sns.countplot(dataset['phy_release_quarter'].sort_values())
#plt.title('movie release count by quarter',fontsize=20)
#loc, labels=plt.xticks()
#plt.xticks(fontsize=12,rotation=90)
#plt.show()










'''

dataset['meanUnitsByWeek'] = dataset.groupby("PHY to TH")["phy_w1_units"].aggregate('mean')
dataset['meanUnitsByWeek'].plot(figsize=(15,10),color="g")

plt.xlabel("Release week")
plt.ylabel("Units Sold")
plt.title("Movie Mean Units Release by Week",fontsize=20)
plt.show()

'''


#split train and test from dataset

train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)

train.info()

train.describe()

test.info()

test.describe()



# Pulling out columns for X  and Y in train set
X_training = train.drop(['phy_w1_units','phy_release_date'], axis=1).values
#X_training = train.drop('phy_release_date', axis=1).values
Y_training = train[['phy_w1_units']].values


# Pulling out columns for X  and Y in test set
X_testing = test.drop(['phy_w1_units','phy_release_date'], axis=1).values
Y_testing = test[['phy_w1_units']].values





#verify succeful selection

print(X_training.shape)
print(Y_training.shape)

print(X_testing.shape)
print(Y_testing.shape)


# set up a data scaler to a range of 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))


# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)



# training and test data get scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)



print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))



# model parameters
learning_rate = 0.0001
training_epochs = 10000

#  inputs and outputs in neural network
number_of_inputs = 5
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 10
layer_2_nodes = 100
layer_3_nodes = 10


# Section One: Define the layers of the neural network itself


# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)



# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases





# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))



# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()



'''
# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # Print the current training status to the screen
        print("Training pass: {}".format(epoch))

    # Training is now complete!
    print("Training is complete!")
    
'''

saver = tf.train.Saver()

# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:


    #Restorre variables insteadd of initializing them
    #saver.restore(session, 'logs/trained_model.ckpt')

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.summary.FileWriter('./logs/training', session.graph)
    testing_writer = tf.summary.FileWriter('./logs/testing', session.graph)

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # Every 5 training steps, log our progress
        if epoch % 5 == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

            # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))

    # Training is now complete!

    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    # Now that the neural network is trained, let's use it to make predictions for our test data.
    # Pass in the X testing data and run the "prediciton" operation
    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})

    # Unscale the data back to it's original units
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled).astype(int)

    real_units = dataset['phy_w1_units'].values[10]
    predicted_units = Y_predicted[10][0]

    print("The actual units of Movie #1  were:  {}".format(real_units))
    print("The neural network predicted units: {}".format(predicted_units))

    save_path = saver.save(session, "logs/trained_model.ckpt")
    print("Model saved: {}".format(save_path))


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