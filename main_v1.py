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


sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Daten laden
df = pd.read_csv('input/stocks_step4.csv')
mask = df.stock == 'AMZN'
df = df[mask]

# Best val_loss So Far: 0.0026194197125732

# Konvertiere 'Date' in datetime-Format und sortiere nach Datum
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Nur das 'Close'-Feature für die Vorhersage verwenden
close_prices = df['close'].values.reshape(-1, 1)

# Daten normalisieren
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Trainings- und Testdatensätze erstellen
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60  # 60 Tage als Eingabe für die Vorhersage
X, y = create_sequences(scaled_data, sequence_length)

# Daten in Trainings- und Testsets aufteilen
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Daten für das LSTM-Modell anpassen
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Funktion für das Modell mit Hyperparametern
def build_model(hp):
    model = Sequential()
    
    # Erste LSTM-Schicht
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32), 
                   return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Zweite LSTM-Schicht
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), 
                   return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Dense Schicht
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Lernrate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, step=1e-5)),
                  loss='mean_squared_error')
    
    return model

# Hyperparameter-Tuning mit Keras Tuner
tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=10, factor=3, directory='hpt_v1', project_name='stock_price')

# Modell suchen
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Beste Hyperparameter ausgeben
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Beste Hyperparameter:", best_hyperparameters)

# Modell mit den besten Hyperparametern trainieren
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=0)

# Verlust über Epochen visualisieren
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Trainingsverlust')
plt.plot(history.history['val_loss'], label='Validierungsverlust')
plt.title('Verlust über die Epochen')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Vorhersagen auf Testdaten
y_pred = best_model.predict(X_test)

# Werte zurückskalieren
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inverse = scaler.inverse_transform(y_pred)

# Berechnung der Metriken
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
mse = mean_squared_error(y_test_inverse, y_pred_inverse)
accuracy = 1 - np.mean(np.abs((y_test_inverse - y_pred_inverse) / y_test_inverse))  # Accuracy-Rate

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Ergebnisse visualisieren
plt.figure(figsize=(10, 6))
plt.plot(df['date'].iloc[-len(y_test):], y_test_inverse, label='Echte Preise')
plt.plot(df['date'].iloc[-len(y_test):], y_pred_inverse, label='Vorhergesagte Preise', linestyle='dashed', color='red')
plt.title('Echte vs. Vorhergesagte Preise')
plt.xlabel('Datum')
plt.ylabel('Preis')
plt.legend()
plt.show()