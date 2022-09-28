#!/usr/bin/env python
# coding: utf-8

# ## **Ejercicios propuestos:**
# 
# * **Arquitectura:**
#     1. Cambiar la cantidad de neuronas en cada capa
#     2. Cambiar la función de activación de cada capa (relu, softmax, linear, tanh)
#     3. Agregar otra capa de neuronas
# * **Entrenamiento:** 
#     1. Modificar la cantidad de epochs
#     2. Modificar el batch_size

# # Paquetes

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf


# # Datos

# In[2]:


path = os.getcwd()


# In[3]:


df_ml_predictores = pd.read_csv('df_ml_predictores.csv', index_col = [0])
df_ml_predictores.set_index(pd.to_datetime(df_ml_predictores.index), inplace = True)

df_ml_target = pd.read_csv('df_ml_target.csv', index_col = [0])
df_ml_target.set_index(pd.to_datetime(df_ml_target.index), inplace = True)

df_2022_predictores = pd.read_csv('df_2022_predictores.csv', index_col = [0])
df_2022_predictores.set_index(pd.to_datetime(df_2022_predictores.index), inplace = True)

df_2022_target = pd.read_csv('df_2022_target.csv', index_col = [0])
df_2022_target.set_index(pd.to_datetime(df_2022_target.index), inplace = True)


# In[4]:


df_ml_predictores.index


# ## Entrenamiento y Validación

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df_ml_predictores, df_ml_target, test_size = 0.30, random_state = 42)


# # Feed Forward Neural Network

# ## 1.- Arquitectura

# In[6]:


model_ffnn = tf.keras.models.Sequential()


model_ffnn.add(tf.keras.layers.Dense(50, activation = tf.keras.activations.relu))
model_ffnn.add(tf.keras.layers.Dense(120, activation = tf.keras.activations.relu))
model_ffnn.add(tf.keras.layers.Dense(50, activation = tf.keras.activations.relu))

model_ffnn.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.linear))

model_ffnn.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'])

input_shape = X_train.shape
model_ffnn.build(input_shape)

model_ffnn.summary()


# ## 2.- Entrenamiento

# In[7]:


fnnn_fit = model_ffnn.fit(X_train, y_train, epochs = 250, batch_size = 100,verbose = 0)


# In[8]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(fnnn_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
plt.yscale('log')
plt.ylim([10**1,10**3])
plt.show()

plt.savefig('MAPE_entrenamiento_ffnn.png')


# ## 3.- Evaluación

# In[9]:


loss, mae, mse, mape = model_ffnn.evaluate(X_test,y_test)


# ## 4.- Predicción

# In[10]:


pred_2022_FFNN = model_ffnn.predict(df_2022_predictores)


# # Comparación de modelos

# In[12]:


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
plt.plot(df_2022_target.index[1::],pred_2022_FFNN[0:-1], '-.r', label = r'FFNN')

plt.legend()

plt.text(df_2022_target.index[0], 50, r'Regular', font)
plt.text(df_2022_target.index[0], 80, r'Alerta', font)
plt.text(df_2022_target.index[0], 110, r'Pre-emergencia', font)
plt.text(df_2022_target.index[0], 170, r'Emergencia', font)

plt.title(r'Comparación con datos reales')
plt.xlabel(r'Fechas')
plt.ylabel(r'Concentración de PM2.5, $\mu g/m³$')

plt.show()

plt.savefig('Comparacion_datos_ffnn.png')

