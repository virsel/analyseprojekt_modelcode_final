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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten, Concatenate
from tensorflow.keras.models import Model
import ast

# best val loss:
# Best val_loss So Far: 0.002181291813030839
# Mean Absolute Error (MAE): 14.125694332398677
# Mean Squared Error (MSE): 341.7898679577467
# Accuracy: 97.59%

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Daten laden
df = pd.read_csv('input/stocks_step4.csv')
mask = df.stock == 'AMZN'
df = df[mask]

# Konvertiere 'Date' in datetime-Format und sortiere nach Datum
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# konvertiere tweet_embs zurück in Liste
df['tweet_embs'] = df['tweet_embs'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
# Nur das 'Close'-Feature für die Vorhersage verwenden
close_prices = df['close'].values.reshape(-1, 1)
features = df[['positive', 'negative', 'num_tweets']].values.reshape(-1, 3)
features_emb = np.stack(df['tweet_embs'].values, dtype=np.float32)

# Daten normalisieren
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
scaled_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

# Trainings- und Testdatensätze erstellen
def create_sequences(data, features, features_emb, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        x_close = data[i-sequence_length:i, 0]
        x_features = features[i-sequence_length:i, :]
        x_emb = features_emb[i-sequence_length:i, :]
        x = np.concatenate((x_close.reshape(-1,1), x_features), axis=1)
        x = np.concatenate((x, x_emb), axis=1)
        X.append(x)
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60  # 60 Tage als Eingabe für die Vorhersage
X, y = create_sequences(scaled_data, scaled_features, features_emb, sequence_length)

# Daten in Trainings- und Testsets aufteilen
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Funktion für das Modell mit Hyperparametern
def build_model(hp):
    # Input for LSTM (single feature)
    lstm_input = Input(shape=(X_train.shape[1], 1), name='lstm_input')
    
    # LSTM branch
    lstm_layer = LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        return_sequences=False
    )(lstm_input)
    lstm_dropout = Dropout(
        hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
    )(lstm_layer)
    
    # CNN branch
    cnn_input = Input(shape=(X_train.shape[1], 3), name='cnn_input')

    # Number of CNN layers as a hyperparameter
    num_cnn_layers = hp.Int('num_cnn_layers', min_value=1, max_value=3, default=1)

    x = cnn_input
    for i in range(num_cnn_layers):
        x = Conv1D(
            filters=hp.Int(f'cnn_filters_{i}', min_value=2, max_value=16, step=4),
            kernel_size=3,
            activation='relu',
            name=f'cnn_layer_{i}'
        )(x)

        # Optional: Add dropout or pooling between layers
        # x = MaxPooling1D(pool_size=2)(x)

    cnn_flatten = Flatten()(x)
    
    # Concatenate LSTM and CNN outputs
    concatenated = Concatenate()([lstm_dropout, cnn_flatten])
    
    # Dense layers
    dense_layer1 = Dense(24, activation='relu')(concatenated)
    output_layer = Dense(1)(dense_layer1)
    
    # Create the model
    model = Model(inputs=[lstm_input, cnn_input], outputs=output_layer)
    
    # Lernrate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, step=1e-5)),
                  loss='mean_squared_error')
    
    return model

# Hyperparameter-Tuning mit Keras Tuner
tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=10, factor=3, directory='hpt_v2', project_name='stock_price')

X_train_lstm = X_train[:, :, 0:1]  # First feature for LSTM
X_train_cnn = X_train[:, :, 1:4]   # Features 1-3 for CNN

X_test_lstm = X_test[:, :, 0:1]  # First feature for LSTM
X_test_cnn = X_test[:, :, 1:4]   # Features 1-3 for CNN

# Modell suchen
tuner.search([X_train_lstm, X_train_cnn], y_train, epochs=10, validation_data=([X_test_lstm, X_test_cnn], y_test))

# Beste Hyperparameter ausgeben
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Beste Hyperparameter:", best_hyperparameters)

# Modell mit den besten Hyperparametern trainieren
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit([X_train_lstm, X_train_cnn], y_train, epochs=50, batch_size=64, validation_data=([X_test_lstm, X_test_cnn], y_test), verbose=0)

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
y_pred = best_model.predict([X_test_lstm, X_test_cnn])

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