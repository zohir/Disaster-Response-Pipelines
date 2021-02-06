import sys

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import pickle


import re
from sqlalchemy import create_engine


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")



def load_data(database_filepath):
    '''
    Function to load the data from a database specified as parameter
    
    Args:
        database_filepath (str): file path and name of the database 
        
    Returns:
        - A dataframe of the messages
        - A dataframe of the categories for the messages
        - A list of categories names 
    '''

    df = pd.read_sql_table(db_table_name, 'sqlite:///'+database_filepath)
    X = df['text'] 
    categories_list = [c for c in df.columns if c not in ['genre', 'text']]
    categories_list.sort()
    y =  df[categories_list]
    return (X, y, categories_list)
    


def tokenize(text):
    '''
    Function to prepare a text message to machine learning process.
    It will be apply processes as word_tokenize,lemmatizer, strip
    
    Args:
        text (str): text to process 
        
    Returns:
        list of tokens
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model():
    '''
    Generate the pipeline 
    
    Args:
        
    Returns:
        the sklearn pipeline
    '''

    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=10000)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('multi_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Calculate a numeric score of the capacity of a model to predict categories of a message
    
    Args:
     - model (sklearn model/pipeline) : sklearn model to va
     - X_test (dataframe)             : test dataset to processs
     - Y_test (dataframe)             : expected results of the test dataset 
     - category_names (list(str))     : list of categories names 
        
    Returns:
        the score obtained by this module on this dataset. It is a float between 0 and 1. 
        0 as none prediction was correct. 1 as all predictions were correct. 
    '''

    Y_pred = model.predict(X_test)
    
    all_scores = []
    for i, col in enumerate(category_names):
        cls_rep = classification_report(Y_test[col], Y_pred[:,i])

        avg_scores = (cls_rep.split('\n')[-2]).split()[-4:]
        cur_scores = {'col':col}
        for i, f in enumerate(['precision', 'recall','f1-score','support']):
            cur_scores[f] = float(avg_scores[i]) 
        
        all_scores.append(cur_scores)

    all_scores_df = pd.DataFrame(all_scores).set_index('col')

    general_score = all_scores_df['f1-score'].mean()
    return general_score



    

def save_model(model, model_filepath):
    '''
    Calculate a numeric score of the capacity of a model to predict categories of a message
    
    Args:
     - model (sklearn model/pipeline) : sklearn model to save in a file
     - model_filepath (str)           : Path and name to save the model 
        
    Returns: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('     messages loaded {}'.format(X.shape[0]))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        general_score = evaluate_model(model, X_test, Y_test, category_names)
        print('           model score = ', general_score)


        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
