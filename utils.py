import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from tensorflow.keras.callbacks import ReduceLROnPlateau



# Schriftgrößen festlegen
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

# Schriftgrößen für verschiedene Plotelemente setzen
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
    
def comp_metrics(scaler, y_val, y_pred, df):
    # Werte zurückskalieren
    y_test_inverse = scaler.inverse_transform(y_val.reshape(-1, 1))
    y_pred_inverse = scaler.inverse_transform(y_pred)

    # Berechnung der Metriken
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    mape = np.mean(np.abs((y_test_inverse - y_pred_inverse) / y_test_inverse)) * 100  # Accuracy-Rate

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Visualisierung vorbereiten
    dates = df['date'].iloc[-len(y_val):]  
    y_test_inverse = y_test_inverse.flatten()
    y_pred_inverse = y_pred_inverse.flatten()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_test_inverse, label='Echte Preise', marker='o')
    plt.plot(dates, y_pred_inverse, label='Vorhergesagte Preise', linestyle='dashed', color='red', marker='x')
    plt.fill_between(dates, y_test_inverse, y_pred_inverse, color='gray', alpha=0.3, label='Abweichung')

    # Anpassung der X-Achse (nur Monate als Ticks anzeigen)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Ticks jede Woche
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.xticks(rotation=45)  # Drehe die Achsenticks für bessere Lesbarkeit
    
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
        patience=20, 
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
    
    # Define the callback to reduce learning rate based on validation loss
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',           # Metric to monitor
        factor=0.7,               # Factor by which the learning rate will be reduced
        patience=7,               # Number of epochs with no improvement to wait
        min_lr=1e-5,              # Minimum learning rate
        verbose=1                 # Print updates when learning rate changes
    )
    
    return [model_checkpoint, early_stopping, tensorboard_callback, reduce_lr]



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

def get_loss_data(log_dir, tag='epoch_loss'):
    """
    Extrahiert Verlustdaten aus TensorBoard-Logs
    
    Parameter:
    log_dir (str): Pfad zum TensorBoard-Log-Verzeichnis
    tag (str): Name des Skalars für den Plot
    
    Rückgabe:
    pandas.DataFrame: DataFrame mit Epochen und Verlustwerten
    """
    event_files = tf.io.gfile.glob(f"{log_dir}/events.out.tfevents.*")
    
    if not event_files:
        raise ValueError(f"Keine Event-Dateien in {log_dir} gefunden")
    
    steps = []
    values = []
    
    for event_file in event_files:
        try:
            for e in summary_iterator(event_file):
                for v in e.summary.value:
                    if v.tag == tag:
                        tensor_bytes = v.tensor.tensor_content
                        if tensor_bytes:
                            value = struct.unpack('f', tensor_bytes)[0]
                            steps.append(e.step)
                            values.append(value)
        except Exception as e:
            print(f"Fehler beim Lesen von {event_file}: {e}")
            continue
    
    if not values:
        raise ValueError(f"Keine Werte für Tag '{tag}' gefunden")
        
    df = pd.DataFrame({'Step': steps, 'Loss': values})
    return df.sort_values('Step')


def plot_multiple_training_runs(log_dirs, labels, tag='epoch_loss'):
    """
    Erstellt einen Vergleichsplot mehrerer Trainingsläufe
    
    Parameter:
    log_dirs (list): Liste der Pfade zu den TensorBoard-Verzeichnissen
    labels (list): Bezeichnungen für die Trainingsläufe
    tag (str): Name des zu plottenden Skalars
    """



    
    plt.figure(figsize=(10, 5))
    
    colors = ['b', 'r']
    
    # Logarithmische Y-Achse
    plt.yscale('log')
    
    for log_dir, label, color in zip(log_dirs, labels, colors):
        df = get_loss_data(log_dir, tag)
        
        # Verlustkurve plotten
        plt.plot(df['Step'], df['Loss'], f'{color}-', linewidth=2, label=f'{label}')
        
        # Minimum markieren
        min_loss = df['Loss'].min()
        min_step = df.loc[df['Loss'].idxmin(), 'Step']
        plt.plot(min_step, min_loss, f'{color}o',
                label=f'Min: {min_loss:.4f}')
    
    plt.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust (log)')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()




