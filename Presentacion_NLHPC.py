#!/usr/bin/env python
# coding: utf-8

# # Paquetes

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.layers import LSTM, Dropout, Dense


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


met_col = ['Presion','Temperatura','HR','RapViento','DoY','DoW']


# In[11]:


for col in met_col:    
    df_predictores[col+'_forecast'] = df_predictores[col].shift(-1)


# In[12]:


df_target = c_df_mean['PM25']
df_target = df_target.shift(-1)


# In[13]:


df_target.drop(df_target.index[-1], inplace = True)
df_predictores.drop(c_df_mean.index[-1], inplace = True)


# In[14]:


df_target


# In[15]:


df_ml_predictores = df_predictores.loc['2018':'2021']
df_ml_target = df_target.loc['2018':'2021']

df_ml_predictores.to_csv('Ejercicios/df_ml_predictores.csv')
df_ml_target.to_csv('Ejercicios/df_ml_target.csv')


# In[16]:


df_2022_predictores = df_predictores.loc['2022-01-01':'2022-08-30']
df_2022_target = df_target.loc['2022-01-01':'2022-08-30']

df_2022_predictores.to_csv('Ejercicios/df_2022_predictores.csv')
df_2022_target.to_csv('Ejercicios/df_2022_target.csv')


# ## Entrenamiento y Validación

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(df_ml_predictores, df_ml_target, test_size = 0.30, random_state = 42)


# # Feed Forward Neural Network

# ## 1.- Arquitectura

# In[18]:


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

# In[19]:


fnnn_fit = model_ffnn.fit(X_train, y_train, epochs = 300, batch_size = 150)


# In[20]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(fnnn_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
#plt.yscale('log')
plt.ylim([0,200])
plt.show()


# ## 3.- Evaluación

# In[21]:


loss, mae, mse, mape = model_ffnn.evaluate(X_test,y_test)


# ## 4.- Predicción

# In[22]:


pred_2022_FFNN = model_ffnn.predict(df_2022_predictores)


# # Long-Short Term Memory

# ## 0.- Preprocesamiento LSTM

# <img src= "Datos/time_series.png">

# In[23]:


def reshape_for_lstm(data, timesteps, features):
    """
    Reshapes an array into the format required by an LSTM model.

    Args:
        data (numpy.ndarray): Input data array.
        timesteps (int): Number of time steps in the input sequence.
        features (int): Number of features at each time step.

    Returns:
        numpy.ndarray: Reshaped array.
    """
    # Calculate the number of samples
    num_samples = data.shape[0] - timesteps + 1

    # Create an empty array for reshaped data
    reshaped_data = np.zeros((num_samples, timesteps, features))

    # Reshape the data
    for i in range(num_samples):
        reshaped_data[i] = data[i:i + timesteps, :]

    return reshaped_data


# In[24]:


#def rshp_features_lstm(features, n_steps):
#    ini_batch = features.shape[0]
#    n_features = features.shape[1]
#    array_lstm = np.zeros((ini_batch, n_steps, n_features)) 
#    for i in range(n_features):
#        for j in range(n_steps):
#            array_lstm[:,j,i] = np.roll(features.iloc[:,i],-j)

#    array_lstm = np.delete(array_lstm, -(np.array(range(n_steps-1))+1), axis = 0)
#    return array_lstm


# In[25]:


n_steps = 4

predictores_lstm = reshape_for_lstm(df_ml_predictores.values,n_steps,df_ml_predictores.shape[1])

target_lstm = df_ml_target.drop(df_ml_target.index[0:n_steps-1])

X_lstm_2022 = reshape_for_lstm(df_2022_predictores.values, n_steps,df_2022_predictores.shape[1])


# In[26]:


X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(predictores_lstm, target_lstm.values, test_size = 0.30, random_state = 42)


# ## 1.- Arquitectura

# In[27]:


model_LSTM = tf.keras.models.Sequential()



model_LSTM.add(tf.keras.layers.LSTM(50,input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2]), activation = tf.keras.activations.relu))
#model_LSTM.add(Dropout(0.2))

model_LSTM.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.linear))

model_LSTM.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'])

model_LSTM.summary()


# ## 2.- Entrenamiento

# In[28]:


lstm_fit = model_LSTM.fit(X_train_lstm, Y_train_lstm, epochs = 300, batch_size = 150)


# In[29]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(lstm_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
#plt.yscale('log')
plt.ylim([0,200])
plt.show()


# In[30]:


#lstm_fit.history['mean_absolute_percentage_error'][-1]


# ## 3.- Evaluación

# In[31]:


loss, mae, mse, mape = model_LSTM.evaluate(X_test_lstm,Y_test_lstm)


# ## 4.- Predicción

# In[32]:


pred_2022_LSTM = model_LSTM.predict(X_lstm_2022)


# # Comparación de modelos

# In[33]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
df_2022_LR = reg.predict(df_2022_predictores) 


# In[34]:


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
axs[0].plot(df_2022_target.index[1::],pred_2022_FFNN[0:-1], '-.r', label = r'FFNN')
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
axs[1].plot(df_2022_target.index[(n_steps-1)::],pred_2022_LSTM, '-.m', label = r'LSTM')
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


# In[37]:


# AGREGAR CODIGO PARA CONTEO DE ERRORES


# In[38]:


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


# In[39]:


eval_modelos


# In[40]:


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
    if medicion > 50 and medicion < 80 and not(FFNN > 50 and FFNN < 80) and FFNN < 50:
        eval_modelos['FFNN'][0][1] = eval_modelos['FFNN'][0][1] + 1
    if medicion > 50 and medicion < 80 and not(LSTM > 50 and LSTM < 80) and LSTM < 50:
        eval_modelos['LSTM'][0][1] = eval_modelos['LSTM'][0][1] + 1
    if medicion > 50 and medicion < 80 and not(LR > 50 and LR < 80) and LR < 50:
        eval_modelos['LR'][0][1] = eval_modelos['LR'][0][1] + 1
    # Alerta
    if medicion > 80 and medicion < 110 and not(FFNN > 80 and FFNN < 110) and FFNN < 80:
        eval_modelos['FFNN'][0][2] = eval_modelos['FFNN'][0][2] + 1
    if medicion > 80 and medicion < 110 and not(LSTM > 80 and LSTM < 110) and LSTM < 80: 
        eval_modelos['LSTM'][0][2] = eval_modelos['LSTM'][0][2] + 1
    if medicion > 80 and medicion < 110 and not(LR > 80 and LR < 110) and LR < 80:
        eval_modelos['LR'][0][2] = eval_modelos['LR'][0][2] + 1
    # Pre-emergencia
    if medicion > 110 and medicion < 170 and not(FFNN > 110 and FFNN < 170) and FFNN < 110:
        eval_modelos['FFNN'][0][3] = eval_modelos['FFNN'][0][3] + 1
    if medicion > 110 and medicion < 170 and not(LSTM > 110 and LSTM < 170) and LSTM < 110:
        eval_modelos['LSTM'][0][3] = eval_modelos['LSTM'][0][3] + 1
    if medicion > 110 and medicion < 170 and not(LR > 110 and LR < 170) and LR < 110:
        eval_modelos['LR'][0][3] = eval_modelos['LR'][0][3] + 1
    # Emergencia
    if medicion > 170 and not(FFNN > 110):
        eval_modelos['FFNN'][0][4] = eval_modelos['FFNN'][0][4] + 1
    if medicion > 170 and not(LSTM > 170):
        eval_modelos['LSTM'][0][4] = eval_modelos['LSTM'][0][4] + 1  
    if medicion > 170 and not(LR > 170):
        eval_modelos['LR'][0][4] = eval_modelos['LR'][0][4] + 1  


# In[41]:


eval_modelos

