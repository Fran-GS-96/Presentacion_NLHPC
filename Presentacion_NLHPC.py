#!/usr/bin/env python
# coding: utf-8

# # Paquetes

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

import tensorflow as tf


# # Visualizacion previa de datos

# In[2]:


path = os.getcwd()


# In[3]:


c_df = pd.read_csv(os.path.join(path, 'Datos', 'Data_Coyhaique.csv'), index_col = [0])
c_df = c_df.set_index(pd.to_datetime(c_df.index))


# In[4]:


c_df.isna().sum()/len(c_df)*100


# In[5]:


c_df = c_df.fillna(value = c_df.mean())


# In[6]:


c_df_mean = c_df.resample('D',kind = 'timestamp').mean()


# In[7]:


c_df_mean


# In[8]:


c_df_mean.plot(subplots = True, layout = (4,4), figsize = (7*(1+np.sqrt(5))/2,7), rot=45);


# # Separación de datos

# In[9]:


df_predictores = c_df_mean
df_predictores


# In[10]:


df_target = c_df_mean['PM25']
df_target = df_target.shift(-1)


# In[11]:


df_target.drop(df_target.index[-1], inplace = True)
df_predictores.drop(c_df_mean.index[-1], inplace = True)


# In[12]:


df_target


# In[13]:


df_ml_predictores = df_predictores.loc['2018':'2021']
df_ml_target = df_target.loc['2018':'2021']


# In[14]:


df_2022_predictores = df_predictores.loc['2022-01-01':'2022-08-30']
df_2022_target = df_target.loc['2022-01-01':'2022-08-30']


# ## Entrenamiento y Validación

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(df_ml_predictores, df_ml_target, test_size = 0.30, random_state = 42)


# # Feed Forward Neural Network

# ## 1.- Arquitectura

# In[16]:


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

# In[17]:


fnnn_fit = model_ffnn.fit(X_train, y_train, epochs = 250, batch_size = 100,verbose = 0)


# In[18]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(fnnn_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
plt.show()


# ## 3.- Evaluación

# In[19]:


loss, mae, mse, mape = model_ffnn.evaluate(X_test,y_test)


# ## 4.- Predicción

# In[20]:


pred_2022_FFNN = model_ffnn.predict(df_2022_predictores)


# # Long-Short Term Memory

# ## 0.- Preprocesamiento LSTM

# In[21]:


def rshp_features_lstm(features, n_steps):
  ini_batch = features.shape[0]
  n_features = features.shape[1]
  array_lstm = np.zeros((ini_batch, n_steps, n_features)) 
  for i in range(n_features):
    for j in range(n_steps):
      array_lstm[:,j,i] = np.roll(features.iloc[:,i],-j)

  array_lstm = np.delete(array_lstm, range(n_steps -1 ), axis = 0)
  return array_lstm


# In[22]:


#ini_batch = df_ml_predictores.shape[0]
#n_features = df_ml_predictores.shape[1]
n_steps = 5

new_features_n = rshp_features_lstm(df_ml_predictores,n_steps)

new_target_n = df_ml_target.drop(df_ml_target.index[0:n_steps-1])

X_lstm_2022 = rshp_features_lstm(df_2022_predictores, n_steps)


# In[23]:


X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(new_features_n, new_target_n.values, test_size = 0.30, random_state = 42)


# ## 1.- Arquitectura

# In[24]:


model_LSTM1 = tf.keras.models.Sequential()

model_LSTM1.add(tf.keras.layers.LSTM(50,input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2]), activation = tf.keras.activations.relu))


model_LSTM1.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.linear))

model_LSTM1.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'])

#input_shape = X_train_lstm.shape[1],X_train_lstm.shape[2])
#model_LSTM1.build((2, 9))

model_LSTM1.summary()


# ## 2.- Entrenamiento

# In[25]:


lstm_fit = model_LSTM1.fit(X_train_lstm, Y_train_lstm, epochs = 250, batch_size = 100,verbose = 0)


# In[26]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(lstm_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
plt.show()


# ## 3.- Evaluación

# In[27]:


loss, mae, mse, mape = model_LSTM1.evaluate(X_test_lstm,Y_test_lstm)


# ## 4.- Predicción

# In[28]:


pred_2022_LSTM = model_LSTM1.predict(X_lstm_2022)


# # Comparación de modelos

# In[29]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
df_2022_LR = reg.predict(df_2022_predictores) 


# In[30]:


# # df_2022_target, pred_2022_FFNN, pred_2022_LSTM
# font = {'family': 'serif',
#         'color':  'red',
#         'weight': 'normal',
#         'size': 16,
#         'alpha': 0.3
#         }

# fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))

# plt.axhline(y = 0, color = 'k', linestyle = '-',linewidth = 1)
# plt.axhline(y = 50, color = 'r', linestyle = '-', alpha = 0.3)
# plt.axhline(y = 80, color = 'r', linestyle = '-', alpha = 0.3)
# plt.axhline(y = 110, color = 'r', linestyle = '-', alpha = 0.3)
# plt.axhline(y = 170, color = 'r', linestyle = '-', alpha = 0.3)


# plt.plot(df_2022_target, '-k', label = r'Medición')
# plt.plot(df_2022_target.index[1::],pred_2022_FFNN[0:-1], '-c', label = r'FFNN')
# plt.plot(df_2022_target.index[4::],pred_2022_LSTM, '-.m', label = r'LSTM')
# plt.plot(df_2022_target.index[1::],df_2022_LR[0:-1], '--y', label = r'LR')

# plt.legend()

# plt.text(df_2022_target.index[0], 50, r'Regular', font)
# plt.text(df_2022_target.index[0], 80, r'Alerta', font)
# plt.text(df_2022_target.index[0], 110, r'Pre-emergencia', font)
# plt.text(df_2022_target.index[0], 170, r'Emergencia', font)

# #plt.xticks(range(len(df_2022_target.index)),df_2022_target.index)
# plt.title(r'Comparación de Predicciones')
# plt.xlabel(r'Fechas')
# plt.ylabel(r'Concentración de PM2.5, $\mu g/m³$')

# plt.show()


# In[46]:


# df_2022_target, pred_2022_FFNN, pred_2022_LSTM
font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 16,
        'alpha': 0.3
        }

fig, axs = plt.subplots(2,1,figsize = (7*(1+np.sqrt(5))/2,8), sharex = True, sharey = True)

axs[0].axhline(y = 0, color = 'k', linestyle = '-',linewidth = 1)
axs[0].axhline(y = 50, color = 'b', linestyle = '-', alpha = 0.3)
axs[0].axhline(y = 80, color = 'b', linestyle = '-', alpha = 0.3)
axs[0].axhline(y = 110, color = 'b', linestyle = '-', alpha = 0.3)
axs[0].axhline(y = 170, color = 'b', linestyle = '-', alpha = 0.3)

axs[0].plot(df_2022_target, '-k', label = r'Medición')
axs[0].plot(df_2022_target.index[1::],pred_2022_FFNN[0:-1], '-.g', label = r'FFNN')
#axs[0].plot(df_2022_target.index[4::],pred_2022_LSTM, '-.m', label = r'LSTM')
axs[0].plot(df_2022_target.index[1::],df_2022_LR[0:-1], '--y', label = r'LR')

axs[0].legend()

axs[0].text(df_2022_target.index[0], 50, r'Regular', font)
axs[0].text(df_2022_target.index[0], 80, r'Alerta', font)
axs[0].text(df_2022_target.index[0], 110, r'Pre-emergencia', font)
axs[0].text(df_2022_target.index[0], 170, r'Emergencia', font)

axs[1].axhline(y = 0, color = 'k', linestyle = '-',linewidth = 1)
axs[1].axhline(y = 50, color = 'b', linestyle = '-', alpha = 0.3)
axs[1].axhline(y = 80, color = 'b', linestyle = '-', alpha = 0.3)
axs[1].axhline(y = 110, color = 'b', linestyle = '-', alpha = 0.3)
axs[1].axhline(y = 170, color = 'b', linestyle = '-', alpha = 0.3)

axs[1].plot(df_2022_target, '-k', label = r'Medición')
#axs[1].plot(df_2022_target.index[1::],pred_2022_FFNN[0:-1], '-c', label = r'FFNN')
axs[1].plot(df_2022_target.index[4::],pred_2022_LSTM, '-.m', label = r'LSTM')
axs[1].plot(df_2022_target.index[1::],df_2022_LR[0:-1], '--y', label = r'LR')

axs[1].legend()

axs[1].text(df_2022_target.index[0], 50, r'Regular', font)
axs[1].text(df_2022_target.index[0], 80, r'Alerta', font)
axs[1].text(df_2022_target.index[0], 110, r'Pre-emergencia', font)
axs[1].text(df_2022_target.index[0], 170, r'Emergencia', font)

#plt.xticks(range(len(df_2022_target.index)),df_2022_target.index)
plt.suptitle(r'Comparación de Predicciones')
plt.xlabel(r'Fechas')
#plt.ylabel(r'Concentración de PM2.5, $\mu g/m³$')
fig.text(0.04, 0.5, r'Concentración de PM2.5, $\mu g/m³$', va='center', rotation='vertical')
plt.show()


# In[32]:


# AGREGAR CODIGO PARA CONTEO DE ERRORES


# In[33]:


eval_modelos = {'FFNN':np.zeros((1,5)),
                'LSTM':np.zeros((1,5)),
                'LR':np.zeros((1,5))}
for medicion, FFNN, LSTM, LR in zip(df_2022_target.values[4::], pred_2022_FFNN[3:-1], pred_2022_LSTM, df_2022_LR[0:-1]):
    # Bueno
    if medicion < 50 and not(FFNN < 50):
        eval_modelos['FFNN'][0][0] = eval_modelos['FFNN'][0][0] + 1
    if medicion < 50 and not(LSTM < 50):
        eval_modelos['LSTM'][0][0] = eval_modelos['LSTM'][0][0] + 1 
    if medicion < 50 and not(LR < 50):
        eval_modelos['LR'][0][0] = eval_modelos['LR'][0][0] + 1 
    # Regular
    if medicion > 50 and medicion < 80 and not(FFNN > 50 and FFNN < 80):
        eval_modelos['FFNN'][0][1] = eval_modelos['FFNN'][0][1] + 1
    if medicion > 50 and medicion < 80 and not(LSTM > 50 and LSTM < 80):
        eval_modelos['LSTM'][0][1] = eval_modelos['LSTM'][0][1] + 1
    if medicion > 50 and medicion < 80 and not(LR > 50 and LR < 80):
        eval_modelos['LR'][0][1] = eval_modelos['LR'][0][1] + 1
    # Alerta
    if medicion > 80 and medicion < 110 and not(FFNN > 80 and FFNN < 110):
        eval_modelos['FFNN'][0][2] = eval_modelos['FFNN'][0][2] + 1
    if medicion > 80 and medicion < 110 and not(LSTM > 80 and LSTM < 110):
        eval_modelos['LSTM'][0][2] = eval_modelos['LSTM'][0][2] + 1
    if medicion > 80 and medicion < 110 and not(LR > 80 and LR < 110):
        eval_modelos['LR'][0][2] = eval_modelos['LR'][0][2] + 1
    # Pre-emergencia
    if medicion > 110 and medicion < 170 and not(FFNN > 110 and FFNN < 170):
        eval_modelos['FFNN'][0][3] = eval_modelos['FFNN'][0][3] + 1
    if medicion > 110 and medicion < 170 and not(LSTM > 110 and LSTM < 170):
        eval_modelos['LSTM'][0][3] = eval_modelos['LSTM'][0][3] + 1
    if medicion > 110 and medicion < 170 and not(LR > 110 and LR < 170):
        eval_modelos['LR'][0][3] = eval_modelos['LR'][0][3] + 1
    # Emergencia
    if medicion > 170 and not(FFNN > 110):
        eval_modelos['FFNN'][0][4] = eval_modelos['FFNN'][0][4] + 1
    if medicion > 170 and not(LSTM > 170):
        eval_modelos['LSTM'][0][4] = eval_modelos['LSTM'][0][4] + 1  
    if medicion > 170 and not(LR > 170):
        eval_modelos['LR'][0][4] = eval_modelos['LR'][0][4] + 1  


# In[34]:


eval_modelos

