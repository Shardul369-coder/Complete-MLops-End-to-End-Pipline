import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

#Ensure log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger('data_injection')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_handler = os.path.join(log_dir, 'data_injection.log')
file_handler = logging.FileHandler(log_file_handler)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data from {data_url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by separating features and target."""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
        df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
        logger.debug('Data preprocessed successfully.')
        return df
    except KeyError as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in preprocessing data: {e}")
        raise

def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame, data_path: str) -> None:
    """Save DataFrame to a CSV file."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'),index=False)
        logger.debug('Train and test data saved successfully to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error saving data: %s', e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/4f789fcf983829f18a0aa9032858aed0e0698bd4/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Data injection process failed: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
