#import relevant libraries

import sys
import os
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle
from sqlalchemy import create_engine

def load_data(database_filepath):
    '''
    function to load the database
    INPUT- database_filepath from etl pipeline
    OUTPUT- store features in x and targets in y
    '''
    table_name = 'messages'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df.drop(["message","id","genre","original"], axis=1)
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    '''
    Tokenizing the text
    INPUT- input the text 
    OUTPUT- cleaned tokens and store it in a list
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
    Function to build a model, create pipeline and gridsearchcv
    INPUT- No arguments
    OUTPUT- model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
       ])

    parameters = {
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 4]
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters)    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluating the model
    INPUT- model, X_test, Y_test, category_names
    OUTPUT- Evaluation score
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))

def save_model(model, model_filepath):
    '''
    Exporting the file to picle 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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