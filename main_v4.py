import __init__
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
import os
import sys
from data import load_data_with_sent, train_to_xy, val_to_xy
import os
import ast  # for safely evaluating string representations of lists
from utils import comp_metrics, get_callbacks
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten, Concatenate, MaxPooling1D
from tensorflow.keras.models import Model


version = 'v4'
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'input/stocks_step4.csv')
output_path = os.path.join(dir_path, f'output/{version}')
hpo_path = os.path.join(output_path, 'hpo4')
model_path = os.path.join(output_path, 'model')
log_path = os.path.join(output_path, 'logs')
os.makedirs(output_path, exist_ok=True)

# Daten laden
td, vd = load_data_with_sent(data_path, window_size=30)

# Trainings- und Validierungsdaten in X und y aufteilen
X_train, y_train = train_to_xy(td)
X_val, y_val = val_to_xy(vd)


# Funktion für das Modell mit Hyperparametern
def build_model(hp):
    # Input for LSTM (single feature)
    lstm_input = Input(shape=(X_train.shape[1], 1), name='lstm_input')
    
    # LSTM branch
    lstm_layer = LSTM(
        units=hp.Int('lstm1_units', min_value=64, max_value=128, step=16),
        return_sequences=True
    )(lstm_input)
    lstm_dropout = Dropout(
        0.2
    )(lstm_layer)
    
    lstm_layer2 = LSTM(
        units=hp.Int('lstm2_units', min_value=32, max_value=64, step=16),
        return_sequences=False
    )(lstm_dropout)
    lstm_dropout2 = Dropout(
        0.4
    )(lstm_layer2)
    
    # LSTM branch (replacing CNN)
    lstm_seq_input = Input(shape=(X_train.shape[1], 3), name='lstm_input2')

    # Number of LSTM layers as a hyperparameter
    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3, default=1)

    x = lstm_seq_input
    for i in range(num_lstm_layers):
        x = LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=128, step=32),
            return_sequences=(i != num_lstm_layers - 1),  # Only last LSTM should output 2D
            activation='tanh',
            name=f'lstm_seq_layer_{i}'
        )(x)

        x = Dropout(0.2)(x)
       
    lstm_seq_out = x 

    cnn_input = Input(shape=(X_train.shape[1], 768), name='cnn_input')
    
    # CNN branch (replacing CNN)
    num_cnn_layers = hp.Int('num_cnn_layers', min_value=1, max_value=3, default=1)

    x = cnn_input
    for i in range(num_cnn_layers):
        x = Conv1D(
            filters=hp.Int(f'cnn_filters_{i}', min_value=2, max_value=18, step=4),
            kernel_size=3,
            activation='elu',
            name=f'cnn_layer_{i}'
        )(x)
        
        x = Dropout(
                0.2
            )(x)

    cnn_flatten = Flatten()(x)

    # Concatenate LSTM (first branch) and LSTM sequence (second branch) outputs
    concatenated = Concatenate()([lstm_dropout2, lstm_seq_out, cnn_flatten])
    
    # Dense layers
    dense_layer1 = Dense(64, activation='relu')(concatenated)
    dense_layer2 = Dense(32, activation='relu')(dense_layer1)
    output_layer = Dense(1)(dense_layer2)
    
    # Create the model
    model = Model(inputs=[lstm_input, lstm_seq_input, cnn_input], outputs=output_layer)
    
    # Lernrate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, step=1e-5)),
                  loss='mean_squared_error')
    
    return model

# Hyperparameter-Tuning mit Keras Tuner
tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=10, factor=3, directory=hpo_path, project_name='stock_price')

X_train_lstm = X_train[:, :, 0:1]  # First feature for LSTM
X_train_lstm2 = X_train[:, :, 1:4]   # Features 1-3 for CNN
X_train_cnn = X_train[:, :, 4:]   # Features 1-3 for CNN

X_val_lstm = X_val[:, :, 0:1]  # First feature for LSTM
X_val_lstm2 = X_val[:, :, 1:4]   # Features 1-3 for CNN
X_val_cnn = X_val[:, :, 4:]   # Features 1-3 for CNN

# Modell suchen
tuner.search([X_train_lstm, X_train_lstm2, X_train_cnn], y_train, epochs=10, validation_data=([X_val_lstm, X_val_lstm2, X_val_cnn], y_val))

# Beste Hyperparameter ausgeben
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Beste Hyperparameter:", best_hyperparameters)

# Modell mit den besten Hyperparametern trainieren
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    [X_train_lstm, X_train_lstm2, X_train_cnn], 
    y_train,
    epochs=50, 
    batch_size=64, 
    validation_data=([X_val_lstm, X_val_lstm2, X_val_cnn], y_val), 
    callbacks=get_callbacks(model_path, log_path),
    verbose=1)