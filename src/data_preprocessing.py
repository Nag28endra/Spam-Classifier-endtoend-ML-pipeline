import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Create 'logs' directory
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str):
    ps = PorterStemmer()
    text =text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text = [ps.stem(word) for word in text]

    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target')->pd.DataFrame:
    try:
        logger.debug('Started preprocessing for Dataframe')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target variable is encoded!')

        df = df.drop_duplicates(keep='first')
        logger.debug('duplicates are removed!')

        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('text column transformed!')
        return df
    except KeyError as e:
        logger.error(f'Column not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error occurred during text normalization: {e}')
        raise

def main(text_column='text',target_column = 'target'):
    try:
        train_data = pd.read_csv(r'./data/raw/train.csv')
        test_data = pd.read_csv(r'./data/raw/test.csv')
        logger.debug('Data loaded!')

        train_processed_data = preprocess_df(train_data,text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)

        data_path = os.path.join('./data','interium')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug(f'Processed data saved to :{data_path}')
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f'No data found: {e}')
        raise
    except Exception as e:
        logger.error(f'Failed to complete the data transformation process: {e}')
        raise

if __name__=='__main__':
    main()






