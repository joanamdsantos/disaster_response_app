import re
import sys
import nltk
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

import nltk
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


def tokenize(text):
    """
    Tokenizes a given text by removing URLs and lemmatizing the words.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of cleaned and lemmatized tokens.

    """
    def tokenize(message):
        if text is None or not isinstance(text, str):
            return []  # Return an empty list for None or non-string inputs

        # Normalize case and remove punctuation
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        
        # Replace URLs
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "urlplaceholder")
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok.strip()]

        return clean_tokens or []
    
def build_model():
    """
    Builds a machine learning model using a pipeline of CountVectorizer, TfidfTransformer, and MultiOutputClassifier.
    
    The model is optimized using GridSearchCV with a range of hyperparameters.
    
    Returns:
        A GridSearchCV object containing the optimized model.
    """
    
    def custom_analyzer(doc):
        return tokenize(doc)

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer=custom_analyzer)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__C': [1, 10, 100],
        'clf__estimator__penalty': ['l2'],        
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, 
                      #cv=3, n_jobs=-1, 
                      #verbose=2, 
                      error_score='raise'
                      )

    print("GridSearchCV object created successfully")
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
        accuracy = (y_pred[:,i] == Y_test.iloc[:,i]).mean()
        df_scores[Y_test.columns[i]] = accuracy
    
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