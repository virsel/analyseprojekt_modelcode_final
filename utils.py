import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import json

def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Trainingsverlust')
    plt.plot(history.history['val_loss'], label='Validierungsverlust')
    plt.title('Verlust über die Epochen')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def comp_metrics(scaler, y_val, y_pred, df):
    # Werte zurückskalieren
    y_test_inverse = scaler.inverse_transform(y_val.reshape(-1, 1))
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
    plt.plot(df['date'].iloc[-len(y_val):], y_test_inverse, label='Echte Preise')
    plt.plot(df['date'].iloc[-len(y_val):], y_pred_inverse, label='Vorhergesagte Preise', linestyle='dashed', color='red')
    plt.title('Echte vs. Vorhergesagte Preise')
    plt.xlabel('Datum')
    plt.ylabel('Preis')
    plt.legend()
    plt.show()
    
    
def get_callbacks(model_save_path, log_dir):
    # Create a ModelCheckpoint callback to save the best model during training
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path, 
        monitor='val_loss', 
        mode='min', 
        save_best_only=True, 
        save_weights_only=False,
        verbose=1
    )

    # Optional: Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
        # TensorBoard Callback
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log histogram of weights
        write_graph=True,  # Log model graph
        write_images=True,  # Log model weights as images
        update_freq='epoch'  # Log at every epoch
    )
    
    return [model_checkpoint, early_stopping, tensorboard_callback]



def describe_model_architecture(loaded_model):
    # Analyze the model layers
    layer_details = []
    for layer in loaded_model.layers:
        layer_info = {
            'name': layer.name,
            'type': type(layer).__name__,
        }
        
        # Try to extract specific layer details
        if hasattr(layer, 'units'):
            layer_info['units'] = layer.units
        
        if hasattr(layer, 'rate'):
            layer_info['dropout_rate'] = layer.rate
        
        layer_details.append(layer_info)
    
    return layer_details