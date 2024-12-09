import numpy as np
import tensorflow as tf
import random
import os
import sys

# Set seeds for reproducibility
def set_seeds(seed=42):
    # Python's random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # OS environment
    os.environ['PYTHONHASHSEED'] = str(seed)

# Call this function before your data preparation and model training
set_seeds(42)  # You can choose any integer seed value

# Additional TensorFlow-specific configurations for deterministic behavior
tf.keras.utils.set_random_seed(42)

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'