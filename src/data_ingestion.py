import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Create 'logs' directory
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data Loaded from {data_url}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Faild to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) ->pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug('Data preprocessing completed!')
        return df
    except Exception as e:
        logger.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path:str) ->None:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug(f'Train and Test data saved to: {raw_data_path}')

    except Exception as e:
        logger.error(f'Unexpected error occured while saving the data: {e}')
        raise

def main():
    try:
        test_size =0.2
        url = "https://raw.githubusercontent.com/Nag28endra/datasets/refs/heads/main/spam.csv"
        df = load_data(url)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=0.2,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error(f'Failed to complete the data ingestion process: {e}')
        raise

if __name__=='__main__':
    main()