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
from data import load_data, train_to_xy, val_to_xy
import os
import ast  # for safely evaluating string representations of lists
from utils import comp_metrics, get_callbacks

# best loss: 0.00211
version = 'v3'
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'input/stocks_step4.csv')
output_path = os.path.join(dir_path, f'output/{version}')
hpo_path = os.path.join(output_path, 'hpo')
model_path = os.path.join(output_path, 'model')
log_path = os.path.join(output_path, 'logs')
os.makedirs(output_path, exist_ok=True)

# Daten laden
td, vd, _ = load_data(data_path, window_size=30)

# Trainings- und Validierungsdaten in X und y aufteilen
X_train, y_train = train_to_xy(td)
X_val, y_val = val_to_xy(vd)


# Funktion für das Modell mit Hyperparametern
def build_model(hp):
    model = Sequential()
    
    # Erste LSTM-Schicht
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32), 
                   return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.3, step=0.1)))
    
    # Zweite LSTM-Schicht
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), 
                   return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1)))
    
    # Dense Schicht
    model.add(Dense(hp.Int(f'dense_layer1', min_value=32, max_value=64, step=32), activation='relu'))
    model.add(Dense(1))
    
    # Lernrate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, step=1e-5)),
                  loss='mean_squared_error')
    
    return model

# Hyperparameter-Tuning mit Keras Tuner
tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=10, factor=3, directory=hpo_path, project_name='stock_price')

# Modell suchen
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Beste Hyperparameter ausgeben
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Beste Hyperparameter:", best_hyperparameters)

# Modell mit den besten Hyperparametern trainieren
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(X_val, y_val), 
    callbacks=get_callbacks(model_path, log_path),
    verbose=1)