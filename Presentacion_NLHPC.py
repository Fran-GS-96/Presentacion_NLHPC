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

from keras.layers import LSTM, Dropout, Dense, Conv1D, Attention


# # Visualizacion previa de datos

# In[2]:


path = os.getcwd()

c_df = pd.read_csv(os.path.join(path, 'Datos', 'Data_Coyhaique.csv'), index_col = [0])
c_df = c_df.set_index(pd.to_datetime(c_df.index))

c_df.isna().sum()/len(c_df)*100


# In[3]:


c_df = c_df.fillna(value = c_df.mean())

c_df_mean = c_df.resample('D',kind = 'timestamp').mean()

c_df_mean.plot(subplots = True, layout = (4,4), figsize = (7*(1+np.sqrt(5))/2,7), rot=45);


# # Separación de datos

# In[4]:


df_predictores = c_df_mean
df_predictores


# In[5]:


met_col = ['Presion','Temperatura','HR','RapViento','DoY','DoW']

for col in met_col:
    if col == 'DoY' or col == 'DoW':
        df_predictores[col+'_forecast'] = df_predictores[col].shift(-1)
    else:
        df_predictores[col+'_forecast'] = df_predictores[col].shift(-1)*np.random.randint(90,110, df_predictores[col].shape)/100
    
df_target = c_df_mean['PM25']
df_target = df_target.shift(-1)

df_target.drop(df_target.index[-1], inplace = True)
df_predictores.drop(c_df_mean.index[-1], inplace = True)

df_ml_predictores = df_predictores.loc['2018':'2021']
df_ml_target = df_target.loc['2018':'2021']

df_ml_predictores.to_csv('Ejercicios/df_ml_predictores.csv')
df_ml_target.to_csv('Ejercicios/df_ml_target.csv')

df_2022_predictores = df_predictores.loc['2022-01-01':'2022-08-30']
df_2022_target = df_target.loc['2022-01-01':'2022-08-30']

df_2022_predictores.to_csv('Ejercicios/df_2022_predictores.csv')
df_2022_target.to_csv('Ejercicios/df_2022_target.csv')


# In[6]:


df_predictores


# ## Entrenamiento y Validación

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(df_ml_predictores, df_ml_target, test_size = 0.30, random_state = 42)


# # Feed Forward Neural Network

# ## 1.- Arquitectura

# In[8]:


model_ffnn = tf.keras.models.Sequential()
model_ffnn.add(Dropout(0.2))
model_ffnn.add(Dense(16, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(32, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(64, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(128, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(64, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(32, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(16, activation = tf.keras.activations.relu))
model_ffnn.add(Dense(8, activation = tf.keras.activations.relu))

## Última capa indica tarea de la red
model_ffnn.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.linear))

model_ffnn.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'])

input_shape = X_train.shape
model_ffnn.build(input_shape)

model_ffnn.summary()


# ## 2.- Entrenamiento

# In[9]:


fnnn_fit = model_ffnn.fit(X_train, y_train, epochs = 300, batch_size = 150)


# In[10]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
plt.plot(fnnn_fit.history['mean_absolute_percentage_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
#plt.yscale('log')
plt.ylim([0,200])
plt.show()


# ## 3.- Evaluación

# In[11]:


loss, mae, mse, mape = model_ffnn.evaluate(X_test,y_test)


# ## 4.- Predicción

# In[12]:


pred_2022_FFNN = model_ffnn.predict(df_2022_predictores)


# # Long-Short Term Memory

# ## 0.- Preprocesamiento LSTM

# <img src= "Datos/time_series.png">

# In[13]:


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


# In[14]:


n_steps = 3

predictores_lstm = reshape_for_lstm(df_ml_predictores.values,n_steps,df_ml_predictores.shape[1])

target_lstm = df_ml_target.drop(df_ml_target.index[0:n_steps-1])

X_lstm_2022 = reshape_for_lstm(df_2022_predictores.values, n_steps,df_2022_predictores.shape[1])


# In[15]:


X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(predictores_lstm, target_lstm.values, test_size = 0.30, random_state = 42)


# ## 1.- Arquitectura

# In[16]:


model_LSTM = tf.keras.models.Sequential()

#model_LSTM.add(LSTM(32,input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2]), activation = tf.keras.activations.relu))#,return_sequences=True))

model_LSTM.add(Conv1D(filters=128, kernel_size=3, 
                      activation= tf.keras.activations.relu, 
                      input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2])))

## Capa de LSTM
model_LSTM.add(LSTM(128, 
                    activation = tf.keras.activations.relu, 
                    return_sequences=True))
model_LSTM.add(LSTM(64, 
                    activation = tf.keras.activations.relu, 
                    return_sequences=True))
model_LSTM.add(LSTM(32, 
                    activation = tf.keras.activations.relu))

## Capa de neuronas clásicas
model_LSTM.add(Dense(32, 
                     activation = tf.keras.activations.linear))

## Última capa indica función de la red
model_LSTM.add(Dense(1, activation = tf.keras.activations.linear))

model_LSTM.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'])

model_LSTM.summary()


# ## 2.- Entrenamiento

# In[17]:


lstm_fit = model_LSTM.fit(X_train_lstm, Y_train_lstm, epochs = 300, batch_size = 150)


# In[18]:


fig = plt.figure(figsize = (7*(1+np.sqrt(5))/2,7))
#plt.plot(lstm_fit.history['mean_absolute_percentage_error'])
plt.plot(lstm_fit.history['mean_squared_error'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error, %')
plt.title('Evolución del MAPE en el set de entrenamiento')
#plt.yscale('log')
#plt.ylim([0,200])
plt.show()


# In[19]:


#lstm_fit.history['mean_absolute_percentage_error'][-1]


# ## 3.- Evaluación

# In[20]:


loss, mae, mse, mape = model_LSTM.evaluate(X_test_lstm,Y_test_lstm)


# ## 4.- Predicción

# In[21]:


pred_2022_LSTM = np.squeeze(model_LSTM.predict(X_lstm_2022))


# # Comparación de modelos

# In[22]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
df_2022_LR = reg.predict(df_2022_predictores) 


# In[23]:


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

