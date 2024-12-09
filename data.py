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
data_path = os.path.join(dir_path, '../input/stocks_step4.csv')


def load_data(batch_size=32):
    # Read data
    df = pd.read_csv(data_path)
    df['tweet_embs'] = df['tweet_embs'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'positive', 'negative', 'num_tweets']
    df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])
    
    # Group the dataframe by stock
    grouped = df.groupby('stock')

    train_data = []
    val_data = []
    test_data = []

    # Iterate through each stock group
    for stock, df_stock in grouped:
        # Reset the dataframe for this stock group
        df_stock = df_stock.reset_index(drop=True)

        # Normalize close prices
        

        # Create training data for this stock
        td, vd, testd = create_data(df_stock)
        train_data.extend(td)
        val_data.extend(vd)
        test_data.extend(testd)
    
    # Prepare the dataloaders
    train_dataset = StockDataset(train_data)
    val_dataset = StockDataset(val_data)
    test_dataset = StockDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader



# Transform to list of tuples
def create_data(df, window_size=30):
    df = df.sort_values('date', ascending=True)
    sent_columns = ['positive', 'negative', 'num_tweets']
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    training_data = []
    val_data = []
    test_data = []
    
    data = []
    
    # Iterate through the dataframe to create training samples
    for i in range(0, len(df)-32, 1):  # Ensure we have 30 days + 1 target day
        # Extract 30 days of numeric data
        X_nums = df.iloc[i:i+30][numeric_columns].values.astype(np.float32)
        X_sent = df.iloc[i:i+30][sent_columns].values.astype(np.float32)
        X_embs = np.stack(df.iloc[i:i+30]['tweet_embs'].values, dtype=np.float32)
        
        # add embs to X_sent
        X_sent = np.concatenate((X_sent, X_embs), axis=1)
        
        # Combine numeric data with news tokens
        X = (X_nums, X_sent)

        y_regr = df.loc[i+30, 'close'].astype(np.float32)
        
        data.append((X, y_regr))
        # Calculate target (1 if price rises, 0 if not)
    
    # np.random.shuffle(data)
    training_data = data[:-80]
    val_test_data = data[-80:]
    np.random.shuffle(training_data)
    np.random.shuffle(val_test_data)
    val_data = val_test_data[-80:-45]
    test_data = val_test_data[-45:]
    
    return training_data, val_data, test_data


if __name__ == '__main__':
    load_data()