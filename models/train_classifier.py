# import libraries
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
from nltk import pos_tag, ne_chunk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

import re
import os

import pickle
import sqlite3


# load data from sql 
def load_data_from_database(database_name):
    '''
    Function that creates a pandas DataFrame from the SQL table

    INPUT:
        database_filepath (str)

    OUTPUT:
        pandas.DataFrame
    '''
    engine = create_engine('sqlite:///..' + database_name)
    return pd.read_sql('disaster_response',engine )       


# load data from sql 
def load_data(database_name):

    '''
    A function that reads the data from the dataframe

    INPUT:
    database_filepath (str)
       

    OUTPUT:
        pandas.DataFrame

    '''
    df = load_data_from_database(database_name)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre', 'source_language'], axis=1)

    category_names = list(df.columns[4:40])    
    return X, Y, category_names    


def tokenize(text):
    
    """Function that cleans and tokenizes input text 
        
    INPUT:
        text string
        
        
    OUTPUT: 
        clean tokens
    """
       
    # tokenize text
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    

    # define stop words
    stop_words = stopwords.words("english")
    
    words = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    clean_tokens = []
    
    
    # iterate through each token
    for tok in tokens:
       
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().replace(" ", "")
        clean_tokens.append(clean_tok)         
        
    return clean_tokens


def build_model():
      
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', TfidfVectorizer(tokenizer=tokenize))
            ])),
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
      
    parameters = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__max_depth': [10, 20, 30]
    }

    cv = GridSearchCV(pipeline, parameters, cv=5, verbose=2, n_jobs=-1)
    return pipeline  


def evaluate_model(model, X_test, y_test, category_names):

    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        
        print('Category {}: {} '.format(i+1, category_names[i]))
        print(classification_report(y_test.iloc[:, i], y_pred[:, i],zero_division=1))
        print('Accuracy {}\n\n'.format(accuracy_score(y_test.iloc[:, i], y_pred[:, i])))
        print('-'*60)    


def save_model(model, model_filepath):   
    
    '''
    Pickle the model for unpickling later

    INPUTS:
        model (pipeline.Pipeline): model to be saved
        model_pickle_filename (str): destination pickle filename
    '''
    pickle.dump(model, open(model_filepath, 'wb'))     


def main():
    if len(sys.argv) == 3:
        database_name, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_name))
        X, Y, category_names = load_data(database_name)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../dbase/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()