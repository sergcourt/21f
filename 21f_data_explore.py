
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


data_table=pd.read_csv("21f_2.csv",  low_memory=False )

#print(data_table.head())

col_list=['PHY Date Week Num','PHY to TH','Studio', 'Rating', 'Box Office','Physical Date','Physical W1 Units']

data_table=data_table.loc[:, col_list]


dataset=data_table.copy()

#print(dataset.tail())




dataset=pd.get_dummies(data=dataset, columns=['Studio'])


dataset=pd.get_dummies(data=dataset, columns=['Rating'])


dataset.rename(index=str, columns={'Box Office':'boxoffice','Physical W1 Units':'phy_w1_units', 'Physical Date':'phy_release_date' }, inplace=True)


dataset.info()


print(dataset.describe)



# encoded = to_categorical(data)

print(dataset.isna().sum())

releaseDate = pd.to_datetime(dataset['phy_release_date'])
dataset['phy_release_dayofweek']=releaseDate.dt.dayofweek
dataset['phy_release_quarter']=releaseDate.dt.quarter

dataset.info()

print(dataset.columns)
#sns.jointplot(x='boxoffice',  y= 'phy_w1_units',data=dataset, height=11, ratio=4, color='r' )
#plt.show()



#sns.distplot(dataset.boxoffice)
#dataset.plot.hist()

# plt.figure(figsize=(20,12))

'''
sns.countplot(dataset['PHY Date Week Num'].sort_values())
plt.title('movie release count by week',fontsize=20)
loc, labels=plt.xticks()
plt.xticks(fontsize=12,rotation=90)
plt.show()



sns.countplot(dataset['phy_release_quarter'].sort_values())
plt.title('movie release count by quarter',fontsize=20)
loc, labels=plt.xticks()
plt.xticks(fontsize=12,rotation=90)
plt.show()


sns.countplot(dataset['phy_release_dayofweek'].sort_values())
plt.title('movie release count by dayofweek',fontsize=20)
loc, labels=plt.xticks()
plt.xticks(fontsize=12,rotation=90)
plt.show()

'''



dataset['meanUnitsByWeek'] = dataset.groupby("PHY to TH")["phy_w1_units"].aggregate('mean')
dataset['meanUnitsByWeek'].plot(figsize=(15,10),color="g")

plt.xticks(np.arange(0,60,1))
plt.xlabel("Release week")
plt.ylabel("Units Sold")
plt.title("Movie Mean Units Release by Week",fontsize=20)
plt.show()












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