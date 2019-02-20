#!/usr/bin/env python
# coding: utf-8

print("Importing modules...")
import os, subprocess
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys
sys.path.append('/project/hep/demers/mnp3/AI/dancing-with-robots/')
# set a seed to control all randomness
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(1)
seed(1)

print("Modules imported!")

# Load some data:
X = np.load('/project/hep/demers/mnp3/AI/dancing-with-robots/data/npy/mariel_knownbetter.npy')
n_joints, n_timeframes, n_dims = X.shape
labels = ['ARIEL.position', 'C7.position', 'CLAV.position', 'LANK.position', 'LBHD.position', 'LBSH.position', 'LBWT.position', 'LELB.position', 'LFHD.position', 'LFRM.position', 'LFSH.position', 'LFWT.position', 'LHEL.position', 'LIEL.position', 'LIHAND.position', 'LIWR.position', 'LKNE.position', 'LKNI.position', 'LMT1.position', 'LMT5.position', 'LOHAND.position', 'LOWR.position', 'LSHN.position', 'LTHI.position', 'LTOE.position', 'LUPA.position', 'LabelingHips.position', 'MBWT.position', 'MFWT.position', 'RANK.position', 'RBHD.position', 'RBSH.position', 'RBWT.position', 'RELB.position', 'RFHD.position', 'RFRM.position', 'RFSH.position', 'RFWT.position', 'RHEL.position', 'RIEL.position', 'RIHAND.position', 'RIWR.position', 'RKNE.position', 'RKNI.position', 'RMT1.position', 'RMT5.position', 'ROHAND.position', 'ROWR.position', 'RSHN.position', 'RTHI.position', 'RTOE.position', 'RUPA.position', 'STRN.position', 'SolvingHips.position', 'T10.position']
print("Input dataset shape (n_joints, n_timeframes, n_dimensions):", X.shape) # (number of joints) X (number of time frames) X (x,y,z dimensions)


from math import floor

# define functions to flatten and unflatten data

def flatten(df, run_tests=True):
  '''
  df is a numpy array with the following three axes:
    df.shape[0] = the index of a vertex
    df.shape[1] = the index of a time stamp
    df.shape[2] = the index of a dimension (x, y, z)
  
  So df[1][0][2] is the value for the 1st vertex (0-based) at time 0 in dimension 2 (z).
  
  To flatten this dataframe will mean to push the data into shape:
    flattened.shape[0] = time index
    flattened.shape[1] = [vertex_index*3] + dimension_vertex
    
  So flattened[1][3] will be the 3rd dimension of the 1st index (0-based) at time 1. 
  '''
  if run_tests:
    assert df.shape == X.shape and np.all(df == X)
  
  # reshape X such that flattened.shape = time, [x0, y0, z0, x1, y1, z1, ... xn-1, yn-1, zn-1]
  flattened = X.swapaxes(0, 1).reshape( (df.shape[1], df.shape[0] * df.shape[2]), order='C' )

  if run_tests: # switch to false to skip tests
    for idx, i in enumerate(df):
      for jdx, j in enumerate(df[idx]):
        for kdx, k in enumerate(df[idx][jdx]):
          assert flattened[jdx][ (idx*df.shape[2]) + kdx ] == df[idx][jdx][kdx]
          
  return flattened

def unflatten(df, run_tests=True, start_time_index=0):
  '''
  df is a numpy array with the following two axes:
    df.shape[0] = time index
    df.shape[1] = [vertex_index*3] + dimension_vertex
    
  To unflatten this dataframe will mean to push the data into shape:
    unflattened.shape[0] = the index of a vertex
    unflattened.shape[1] = the index of a time stamp
    unflattened.shape[2] = the index of a dimension (x, y, z)
    
  So df[2][4] == unflattened[1][2][0]
  '''
  if run_tests:
    assert (len(df.shape) == 2) and (df.shape[1] == X.shape[0] * X.shape[2])
  
  unflattened = np.zeros(( X.shape[0], df.shape[0], X.shape[2] ))

  for idx, i in enumerate(df):
    for jdx, j in enumerate(df[idx]):
      kdx = int(floor(jdx / 3))
      ldx = int(jdx % 3)
      unflattened[kdx][idx][ldx] = df[idx][jdx]

  if run_tests: # set to false to skip tests
    for idx, i in enumerate(unflattened):
      for jdx, j in enumerate(unflattened[idx]):
        for kdx, k in enumerate(unflattened[idx][jdx]):
          assert( unflattened[idx][jdx][kdx] == X[idx][int(start_time_index)+jdx][kdx] )

  return unflattened

flat = flatten(X)
unflat = unflatten(flat)

#  Reduce dimensionality

### Move each frame to (x,y)=(0,0), leaving the z dimension free
X[:,:,:2] -= X[:,:,:2].mean(axis=0, keepdims=True)

### Then "flatten" dimensions, i.e. instead of n timestamps x 55 joints x 3 dimensions, use n timestamps x 165 joints
# reshape such that flattened.shape = time, [x0, y0, z0, x1, y1, z1, ... xn-1, yn-1, zn-1]
flat = X.swapaxes(0, 1).reshape( (X.shape[1], X.shape[0] * X.shape[2]), order='C' )
column_names = [ 'joint'+str(i)+'_'+str(j) for i in range(int(flat.shape[1]/3)) for j in ['x','y','z']]
df = pd.DataFrame(flat, columns=column_names)
print('Size of the dataframe: {}'.format(df.shape))

### Use PCA
print("Starting PCA decomposition...")

from sklearn.decomposition import PCA
pca = PCA(n_components=2) # can either do this by num of desired components...
# pca = PCA(.95) # ...or by percentage variance you want explained 
pca_columns=[]

pca_result = pca.fit_transform(df.values)
for i in range(pca_result.shape[1]):
    df['pca_'+str(i)] = pca_result[:,i]
    pca_columns.append('pca_'+str(i))
    
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# Get the transformed dataset 
df[pca_columns]
pca_data = df[pca_columns]

# train_x has shape: n_samples, look_back, n_vertices*3
look_back = 50 # number of previous time slices to use to predict the time positions at time `i`
train_x = []
train_y = []

# each i is a time slice; these time slices start at idx `look_back` (so we can look back `look_back` slices)
for i in range(look_back, n_timeframes-1, 1):
    train_x.append( pca_data.loc[i-look_back:i-1].to_numpy() )
    train_y.append( pca_data.loc[i] )
    
train_x = np.array(train_x)
train_y = np.asarray(train_y)

print("Training input dataset shape:", train_x.shape)
print("Training output dataset shape:", train_y.shape)

# Build the Model

from utils.mdn import MDN
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Activation, CuDNNLSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras import backend as K
import keras, os

# config
cells = [32, 32, 32, 32] # number of cells in each lstm layer
output_dims = int(pca_data.shape[1]) # number of coordinate values to be predicted by each gaussian model
input_shape = (look_back, output_dims) # shape of each input feature
use_mdn = True # whether to use the MDN final layer or not
n_mixes = 2 # number of gaussian models to build if use_mdn == True

# optimizer params
lr = 0.00001 # the learning rate of the model
optimizer = Adam(lr=lr, clipvalue=0.5)

# use tensorflow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# determine the LSTM cells to use (hinges on whether GPU is available to keras)
gpus = K.tensorflow_backend._get_available_gpus()
LSTM_UNIT = CuDNNLSTM if len(gpus) > 0 else LSTM
print('GPUs found:', gpus)

# build the model
model = Sequential()
model.add(LSTM_UNIT(cells[0], return_sequences=True, input_shape=input_shape, ))
model.add(LSTM_UNIT(cells[1], return_sequences=True, ))
model.add(LSTM_UNIT(cells[2], ))
model.add(Dense(cells[3]), )

if use_mdn:
    mdn = MDN(output_dims, n_mixes)
    model.add(mdn)
    model.compile(loss=mdn.get_loss_func(), optimizer=optimizer, metrics=['accuracy'])
else:
    model.add(Dense(output_dims, activation='tanh'))
    model.compile(loss=mean_squared_error, optimizer=optimizer, metrics=['accuracy'])

model.summary()


# check untrained (baseline) accuracy
model.evaluate(train_x, train_y)


# Train the model

from keras.callbacks import TerminateOnNaN
from livelossplot import PlotLossesKeras
from datetime import datetime
import time, keras, os, json
  
class Logger(keras.callbacks.Callback):
  '''Save the model and its weights every `self.save_frequency` epochs'''
  def __init__(self):
    self.epoch = 0 # stores number of completed epochs
    self.save_frequency = 1 # configures how often we'll save the model and weights
    self.date = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
    if not os.path.exists('models'): os.makedirs('models')
    self.save_config()
    
  def save_config(self):
    with open('models/' + self.date + '-config.json', 'w') as out:
      json.dump({
        'look_back': look_back,
        'cells': cells,
        'use_mdn': use_mdn,
        'n_mixes': n_mixes,
        'lr': lr,
      }, out)
  
  def on_batch_end(self, batch, logs={}, shape=train_x.shape):
    if (batch+1 == shape[0]): # batch value is batch index, which is 0-based
      self.epoch += 1
      if (self.epoch > 0) and (self.epoch % self.save_frequency == 0):
        path = 'models/' + self.date + '-' + str(batch)
        model.save(path + '.model')
        model.save_weights(path + '.weights')

#K.set_value(optimizer.lr, 0.00001)
callbacks = [Logger(), TerminateOnNaN()]
history = model.fit(train_x, train_y, epochs=1, batch_size=1, shuffle=False, callbacks=callbacks)

from datetime import datetime
model_path = '/project/hep/demers/mnp3/AI/dancing-with-robots/models/'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

# Save trained model
model.save(model_path + '.model')
model.save_weights(model_path + '.weights')
