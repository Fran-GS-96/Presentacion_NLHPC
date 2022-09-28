#!/usr/bin/env python
# coding: utf-8

# # Paquetes

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg') 

from sklearn.model_selection import train_test_split

import tensorflow as tf


#Datos

path = os.getcwd()

df_ml_predictores = pd.read_csv('df_ml_predictores.csv', index_col = [0])
df_ml_predictores.set_index(pd.to_datetime(df_ml_predictores.index), inplace = True)

df_ml_target = pd.read_csv('df_ml_target.csv', index_col = [0])
df_ml_target.set_index(pd.to_datetime(df_ml_target.index), inplace = True)

df_2022_predictores = pd.read_csv('df_2022_predictores.csv', index_col = [0])
df_2022_predictores.set_index(pd.to_datetime(df_2022_predictores.index), inplace = True)

df_2022_target = pd.read_csv('df_2022_target.csv', index_col = [0])
df_2022_target.set_index(pd.to_datetime(df_2022_target.index), inplace = True)

df_ml_predictores.index

#Entrenamiento y Validación

X_train, X_test, y_train, y_test = train_test_split(df_ml_predictores, df_ml_target, test_size = 0.30, random_state = 42)


#Long Short Term Memory

#0.- Preprocesamiento LSTM

def rshp_features_lstm(features, n_steps):
  ini_batch = features.shape[0]
  n_features = features.shape[1]
  array_lstm = np.zeros((ini_batch, n_steps, n_features)) 
  for i in range(n_features):
    for j in range(n_steps):
      array_lstm[:,j,i] = np.roll(features.iloc[:,i],-j)

  array_lstm = np.delete(array_lstm, range(n_steps -1 ), axis = 0)
  return array_lstm

n_steps = 2

predictores_lstm = rshp_features_lstm(df_ml_predictores,n_steps)

target_lstm = df_ml_target.drop(df_ml_target.index[0:n_steps-1])

X_lstm_2022 = rshp_features_lstm(df_2022_predictores, n_steps)

X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(predictores_lstm, target_lstm.values, test_size = 0.30, random_state = 42)

#1.- Arquitectura

model_LSTM = tf.keras.models.Sequential()

model_LSTM.add(tf.keras.layers.LSTM(50,input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2]), activation = tf.keras.activations.relu))

model_LSTM.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.linear))

model_LSTM.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'])

model_LSTM.summary()

#2.- Entrenamiento

lstm_fit = model_LSTM.fit(X_train_lstm, Y_train_lstm, epochs = 250, batch_size = 100)

fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(lstm_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
plt.yscale('log')
#plt.ylim([10**-1,10**3])
#plt.show()

plt.savefig('MAPE_entrenamiento_lstm.png')

#3.- Evaluación

loss, mae, mse, mape = model_LSTM.evaluate(X_test_lstm,Y_test_lstm)

#4.- Predicción

pred_2022_LSTM = model_LSTM.predict(X_lstm_2022)

# # Comparación de modelos

font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 16,
        'alpha': 0.3
        }

fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))

plt.axhline(y = 0, color = 'k', linestyle = '-',linewidth = 1)
plt.axhline(y = 50, color = 'b', linestyle = '-', alpha = 0.3)
plt.axhline(y = 80, color = 'b', linestyle = '-', alpha = 0.3)
plt.axhline(y = 110, color = 'b', linestyle = '-', alpha = 0.3)
plt.axhline(y = 170, color = 'b', linestyle = '-', alpha = 0.3)

plt.plot(df_2022_target, '-k', label = r'Medición')
plt.plot(df_2022_target.index[(n_steps-1)::],pred_2022_LSTM, '-.r', label = r'LSTM')

plt.legend()

plt.text(df_2022_target.index[0], 50, r'Regular', font)
plt.text(df_2022_target.index[0], 80, r'Alerta', font)
plt.text(df_2022_target.index[0], 110, r'Pre-emergencia', font)
plt.text(df_2022_target.index[0], 170, r'Emergencia', font)

plt.title(r'Comparación con datos reales')
plt.xlabel(r'Fechas')
plt.ylabel(r'Concentración de PM2.5, $\mu g/m³$')

#plt.show()

plt.savefig('Comparacion_datos_lstm.png')

print('MAPE del set de entrenamiento: ', np.round(lstm_fit.history['mean_absolute_percentage_error'][-1],3))
print('MAPE del set de test: ', np.round(mape,3))