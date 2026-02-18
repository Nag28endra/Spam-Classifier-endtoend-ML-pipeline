import os
import pandas as pd
import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from :{file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'error file loading data from :{e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error while loading data from : {e}')
        raise
    except FileNotFoundError as e:
        logger.error(f'file not found : {e}')
        raise

def train_model(X_train, y_train, params:dict)->RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('The number of samples in X_train and y_train must be same.')
        
        logger.debug(f'Intializing RandomForest model with parameters: {params}')
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            random_state=params['random_state']
        )

        logger.debug(f'Model training started with {X_train.shape[0]} samples')
        clf.fit(X_train,y_train)
        logger.debug('Model training completed.')

        return clf
    except ValueError as e:
        logger.error(f' Value error during model training: {e}')
        raise
    except Exception as e:
        logger.error(f'Error during model training: {e}')
        raise

def save_model(file_path: str,model)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug(f'Model saved to : {file_path}')

    except FileNotFoundError as e:
        logger.error(f'File path not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error occured while saving the model : {e}')
        raise

def main():
    try:
        params = {'n_estimators':25,'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train,y_train,params)        

        model_save_path = 'models/model.pkl'
        save_model(model=clf,file_path=model_save_path)

    except Exception as e:
        logger.error(f'failed to complete the model building process: {e}')
        raise

if __name__=='__main__':
    main()
