import sklearn
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from sklearn.model_selection import train_test_split
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
display(df.head())

X=df.iloc[:,:df.shape[1]-1]
y=df['MEDV']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



# Kerasでネットワークを設計し、モデルを返す関数を定義
def create_model():
    model=Sequential()
    model.add(Dense(X_train.shape[1],input_dim=X_train.shape[1],kernel_initializer='normal',activation='relu'))
    model.add(Dense(16,activation='relu',kernel_initializer='normal'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model




model=KerasRegressor(build_fn=create_model,epochs=10,batch_size=5,verbose=0)
# BostonHousing データ
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(model.score(X_test,y_test))
print(y_test.values)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test.values,y_pred))

