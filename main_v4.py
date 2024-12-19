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

# best loss: 0.00218
version = 'v4'
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'input/stocks_step4.csv')
output_path = os.path.join(dir_path, f'output/{version}')
hpo_path = os.path.join(output_path, 'hpo')
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
    
    # LSTM branch
    lstm_input = Input(shape=(X_train.shape[1], 1), name='lstm_input')
    lstm_layer = LSTM(
        units=hp.Int('lstm1_units', min_value=8, max_value=64, step=16),
        return_sequences=False
    )(lstm_input)
    lstm_dropout = Dropout(
        hp.Float(f'lstm1_dropout', min_value=0.1, max_value=0.3, step=0.1)
    )(lstm_layer)
    

    # CNN branch (replacing CNN)
    cnn_input = Input(shape=(10, 771), name='cnn_input')
    
    x = Conv1D(
            filters=hp.Int(f'cnn_filters_1', min_value=16, max_value=32, step=4),
            kernel_size=3,
            activation='elu',
            name=f'cnn_layer_1'
        )(cnn_input)
        
    x = Dropout(
                hp.Float(f'cnn_dropout_1',  min_value=0.1, max_value=0.3, step=0.1)
            )(x)

    x = Conv1D(
        filters=hp.Int(f'cnn_filters_2', min_value=2, max_value=14, step=4),
        kernel_size=3,
        activation='elu',
        name=f'cnn_layer_2'
    )(x)
    
    x = Dropout(
            hp.Float(f'cnn_dropout_2',  min_value=0.1, max_value=0.3, step=0.1)
        )(x)
    
    x = Conv1D(
        filters=hp.Int(f'cnn_filters_3', min_value=2, max_value=14, step=4),
        kernel_size=3,
        activation='elu',
        name=f'cnn_layer_3'
    )(x)
    
    x = Dropout(
            hp.Float(f'cnn_dropout_3',  min_value=0.1, max_value=0.3, step=0.1)
        )(x)

    cnn_flatten = Flatten()(x)

    # Concatenate LSTM (first branch) and CNN (second branch) outputs
    concatenated = Concatenate()([lstm_dropout, cnn_flatten])
    
    # Dense layers
    dense_layer1 = Dense(hp.Int(f'dense_layer1', min_value=32, max_value=64, step=16), activation='relu')(concatenated)
    output_layer = Dense(1)(dense_layer1)
    
    # Create the model
    model = Model(inputs=[lstm_input, cnn_input], outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, step=1e-5)),
                  loss='mean_squared_error')
    
    return model

# Hyperparameter-Tuning mit Keras Tuner
tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=10, factor=3, directory=hpo_path, project_name='stock_price')

X_train_lstm = X_train[:, :, 0:1]  # First feature for LSTM
X_train_cnn = X_train[:, :10, 1:]   # Features 1-3 for CNN

X_val_lstm = X_val[:, :, 0:1]  # First feature for LSTM
X_val_cnn = X_val[:, :10, 1:]   # Features 1-3 for CNN

# Modell suchen
tuner.search([X_train_lstm, X_train_cnn], y_train, epochs=10, validation_data=([X_val_lstm, X_val_cnn], y_val))

# Beste Hyperparameter ausgeben
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Beste Hyperparameter:", best_hyperparameters)

# Modell mit den besten Hyperparametern trainieren
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    [X_train_lstm, X_train_cnn], 
    y_train,
    epochs=50, 
    batch_size=64, 
    validation_data=([X_val_lstm, X_val_cnn], y_val), 
    callbacks=get_callbacks(model_path, log_path),
    verbose=1)