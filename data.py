from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import ast  # for safely evaluating string representations of lists

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, 'input/stocks_step4.csv')


def load_data(path=data_path, window_size=30, only_goog=False):
    # Read data
    df = pd.read_csv(path)
    if only_goog:
        mask = df.stock == 'GOOG'
        df = df[mask]
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Group the dataframe by stock
    grouped = df.groupby('stock')

    train_data = []
    test_data = []
    val_data = []

    # Iterate through each stock group
    for stock, df_stock in grouped:
        # Reset the dataframe for this stock group
        df_stock = df_stock.reset_index(drop=True)
        df_stock.sort_values('date', ascending=True)
        
        # Normalize close prices
        train_val_size = int(len(df_stock) * 0.8)
        train_size = int(train_val_size * 0.85)
        val_size = train_val_size - train_size
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        df_stock.loc[:train_size - 1, 'close'] = scaler.fit_transform(df_stock.loc[:train_size - 1, 'close'].values.reshape(-1,1))
        df_stock.loc[train_size:, 'close'] = scaler.transform(df_stock.loc[train_size:, 'close'].values.reshape(-1,1))

        # Create training data for this stock
        td, vd, testd = create_data(df_stock, train_size, val_size, window_size)
        
        train_data.extend(td)
        
        val_data.append({'stock': stock, 'val_data': vd, 'scaler': scaler, 'df': df_stock})
        test_data.append({'stock': stock, 'test_data': testd, 'scaler': scaler, 'df': df_stock})
    
    return train_data, val_data, test_data 

# Transform to list of tuples
def create_data(df, train_size, val_size, window_size=30):
    df = df.sort_values('date', ascending=True)
    numeric_columns = ['close']
    
    data = []
    # Iterate through the dataframe to create training samples
    for i in range(0, len(df)- window_size -2, 1):  
        X_nums = df.iloc[i:i+window_size][numeric_columns].values.astype(np.float32)
    
        y_regr = df.loc[i+window_size, 'close'].astype(np.float32)
        
        data.append((X_nums, y_regr))

    # Daten in train, val, test aufteilen
    training_val_data = data[:train_size + val_size]
    train_data = training_val_data[:train_size]
    val_data = training_val_data[train_size:]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

# Trainings- und Testdatensätze erstellen
def train_to_xy(data):
    X = []
    y = []
    for i in range(len(data)):
        data2 = data[i]
        X.append(data2[0])
        y.append(data2[1])
    return np.array(X), np.array(y)

def val_to_xy(stock_data):
    X = []
    y = []
    for i in range(len(stock_data)):
        data = stock_data[i]
        for j in range(len(data['val_data'])):
            data2 = data['val_data'][j]
            X.append(data2[0])
            y.append(data2[1])
    return np.array(X), np.array(y)

def load_data_with_sent(path=data_path, window_size=30, only_goog=False):
    # Read data
    df = pd.read_csv(path)
    if only_goog:
        mask = df.stock == 'GOOG'
        df = df[mask]
    df['tweet_embs'] = df['tweet_embs'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Group the dataframe by stock
    grouped = df.groupby('stock')

    train_data = []
    val_data = []
    test_data = []

    # Iterate through each stock group
    for stock, df_stock in grouped:
        # Reset the dataframe for this stock group
        df_stock = df_stock.reset_index(drop=True)
        df_stock.sort_values('date', ascending=True)
        
        # Normalize close prices
        train_val_size = int(len(df_stock) * 0.8)
        train_size = int(train_val_size * 0.85)
        val_size = train_val_size - train_size
        
        # scale close prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_stock.loc[:train_size - 1, 'close'] = scaler.fit_transform(df_stock.loc[:train_size - 1, 'close'].values.reshape(-1,1))
        df_stock.loc[train_size:, 'close'] = scaler.transform(df_stock.loc[train_size:, 'close'].values.reshape(-1,1))
        
        # scale sentiment values
        scaler_sent = MinMaxScaler(feature_range=(0, 1))
        df_stock.loc[:train_size - 1, ['positive', 'negative', 'num_tweets']] = scaler_sent.fit_transform(df_stock.loc[:train_size - 1,['positive', 'negative', 'num_tweets']].values.reshape(-1,3))
        df_stock.loc[train_size:, ['positive', 'negative', 'num_tweets']] = scaler_sent.transform(df_stock.loc[train_size:,['positive', 'negative', 'num_tweets']].values.reshape(-1,3))

        # Create training data for this stock
        td, vd, testd = create_data_with_sent(df_stock, train_size, val_size, window_size)
        
        train_data.extend(td)
        
        val_data.append({'stock': stock, 'val_data': vd, 'scaler': scaler, 'df': df_stock})
        test_data.append({'stock': stock, 'test_data': testd, 'scaler': scaler, 'df': df_stock})
    
    return train_data, val_data, test_data 

# Transform to list of tuples
def create_data_with_sent(df, train_size, val_size, window_size=30):
    df = df.sort_values('date', ascending=True)
    sent_columns = ['positive', 'negative', 'num_tweets']
    numeric_columns = ['close']
    
    data = []
    
    # Iterate through the dataframe to create training samples
    for i in range(0, len(df)- window_size -2, 1):  # Ensure we have 30 days + 1 target day
        # Extract 30 days of numeric data
        X_nums = df.iloc[i:i+window_size][numeric_columns].values.astype(np.float32)
        X_sent = df.iloc[i:i+window_size][sent_columns].values.astype(np.float32)
        X_embs = np.stack(df.iloc[i:i+window_size]['tweet_embs'].values, dtype=np.float32)
        
        # add embs to X_sent
        X_sent = np.concatenate((X_sent, X_embs), axis=1)
        
        # Combine numeric data with news tokens
        X = np.concatenate((X_nums, X_sent), axis=1)

        y_regr = df.loc[i+window_size, 'close'].astype(np.float32)
        
        data.append((X, y_regr))
        # Calculate target (1 if price rises, 0 if not)
    
    # Daten in train, val, test aufteilen
    training_val_data = data[:train_size + val_size]
    train_data = training_val_data[:train_size]
    val_data = training_val_data[train_size:]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data


if __name__ == '__main__':
    load_data()