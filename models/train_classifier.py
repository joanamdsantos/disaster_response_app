import re
import sys
import nltk
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

import nltk
nltk.download('omw-1.4')
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from a SQLite database into a pandas DataFrame and return the features and target variables.

    Parameters:
        database_filepath (str): The path to the SQLite database file.

    Returns:
        X (ndarray): The features extracted from the DataFrame.
        y (ndarray): The target variables extracted from the DataFrame.

    Raises:
        ValueError: If the database filepath is None, if there is an error loading data from the database,
                    if the DataFrame is None, or if there is an error extracting data from the DataFrame.
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Remove rows with None values in the message column
    df = df.dropna(subset=['message'])
    
    X = df['message'].values
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(message):
        """
        Tokenize a given message by removing URLs, tokenizing the message into individual words,
        lemmatizing each word, and converting all words to lowercase.

        Parameters:
            message (str): The message to be tokenized.

        Returns:
            list: A list of clean tokens representing the tokenized message. If the input message is None or not a string,
                  an empty list is returned.
        """
                
        if message is None or not isinstance(message, str):
            return []  # Return an empty list for None or non-string inputs

        detected_urls = re.findall(url_regex, message)
        for url in detected_urls:
            message = message.replace(url, "urlplaceholder")

        tokens = word_tokenize(message)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens if clean_tokens else []  # Return an empty list if clean_tokens is None or []
    
def build_model():
    """
    Builds a machine learning model using a pipeline of CountVectorizer, TfidfTransformer, and MultiOutputClassifier.
    
    The model is optimized using GridSearchCV with a range of hyperparameters.
    
    Returns:
        A GridSearchCV object containing the optimized model.
    """
    # Hyperparameter tuning for XGBoost sources:
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        # https://medium.com/@rithpansanga/optimizing-xgboost-a-guide-to-hyperparameter-tuning-77b6e48e289d
        # https://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboost/notebook
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(random_state=42)))
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [100, 200, 500],
        'clf__estimator__learning_rate': [0.01,0.05,0.1],
        'clf__estimator__gamma': [0, 0.5, 1],
        #'clf__estimator__reg_alpha': [0, 0.5, 1],
        #'clf__estimator__reg_lambda': [0.5, 1, 5],
        #'clf__estimator__base_score': [0.2, 0.5, 1],
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, 
                      cv=3, 
                      #n_jobs=2, 
                      scoring='f1_weighted', 
                      verbose=2, 
                      error_score='raise'
                      )
    print("GridSearchCV object created successfully")
    
    #model = cv.best_estimator_(**cv.best_params_)
        
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of a trained model on the test dataset.

    Args:
        model (object): The trained machine learning model.
        X_test (array-like): The test input features.
        Y_test (array-like): The test target values.
        category_names (list): The names of the categories.

    Returns:
        dict: A dictionary containing the accuracy scores for each category.
            The keys are the category names and the values are the corresponding accuracy scores.
    """
    
    y_pred = model.predict(X_test)
    
    df_scores = {}
    for i in range(len(Y_test.columns)):
        category = Y_test.columns[i]
        print(f"Category: {category}")
        report = classification_report(Y_test.iloc[:, i], y_pred[:, i])
        accuracy = accuracy_score(Y_test.iloc[:, i], y_pred[:, i])
        print(report)
        print(f"Accuracy: {accuracy}")
        #df_scores[category] = {
            #'report': report,
            #'accuracy': accuracy
        #}
    
    return df_scores

def save_model(model, model_filepath):
    """
    Save the best model found by GridSearchCV to a pickle file.

    Args:
        model (object): The best model found by GridSearchCV.
        model_filepath (str): The filepath where the pickle file will be saved.

    Returns:
        None
    """
    # model = cv.estimator(**cv.best_params)
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        # Remove None values from X
        X = np.array([x if x is not None else '' for x in X])
        
        print(f"Shape of X: {X.shape}")
        print(f"Shape of Y: {Y.shape}")
        print(f"Sample of X: {X[:5]}")
        print(f"Number of None values in X: {sum(x is None for x in X)}")
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print(f"Type of model: {type(model)}")
        
        if model is None:
            print("Error: build_model() returned None")
            return
        
        print('Training model...')
        try:
            model.fit(X_train, Y_train)
        except Exception as e:
            print(f"An error occurred during model fitting: {str(e)}")
            print(f"Sample of X_train: {X_train[:5]}")
            print(f"Number of None values in X_train: {sum(x is None for x in X_train)}")
            raise
        
        print('Evaluating model...')
        scores = evaluate_model(model, X_test, Y_test, category_names)
        for category, score in scores.items():
            print(f"{category}: {score}")

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