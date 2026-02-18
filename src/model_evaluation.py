import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import logging

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug(f'Model loaded from :{file_path}')
        return model
    except FileNotFoundError as e:
        logger.erro(f'File not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected issue file loading the model: {e}')
        raise

def load_data(file_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from: {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the csv file: {e}')
        raise
    except Exception as e:
        logger.error(f'Error occurred while loading the data: {e}')
        raise

def evaluate_model(clf,X_test: np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall':recall,
            'auc':auc
        }

        logger.debug(f'Model evalution metrics calculated.')
        return metrics_dict
    except Exception as e:
        logger.error(f'Error during model evaluation: {e}')
        raise

def save_metrics(metrics:dict, file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug(f'Metrics saved to : {file_path}')
    
    except Exception as e:
        logger.error(f'Error occurred while saving the metrics: {e}')
        raise

def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(clf,X_test,y_test)

        save_metrics(metrics,'reports/metrics.json')

    except Exception as e:
        logger.error(f'Failed to complete model evaluation process: {e}')
        raise

if __name__=='__main__':
    main()
