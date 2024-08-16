import json
import os
import plotly
import pandas as pd
import joblib
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from xgboost.compat import XGBoostLabelEncoder

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

#database_path = os.path.join(os.path.dirname(__file__), '../data/DisasterResponse.db')
#database_path = '/app/data/DisasterResponse.db'
#engine = create_engine(f'sqlite:///{database_path}')

df = pd.read_sql_table('DisasterResponse', engine)

# load model
#model = joblib.load("/app/models/classifier.pk")

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "tokenize":
            return tokenize
        if module == "xgboost.compat" and name == "XGBoostLabelEncoder":
            return XGBoostLabelEncoder
        return super().find_class(module, name)

def custom_load(file_path):
    with open(file_path, 'rb') as f:
        return CustomUnpickler(f).load()

# Custom load method
model = custom_load("../models/classifier.pk")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data for category distribution
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='green')
                )
            ],
            'layout': {
                'title': {
                    'text': 'Distribution of Message Genres',
                    'font': {'size': 24, 'color': 'black', 'family': 'Arial, sans-serif', 'weight': 'bold'}
                },
                'yaxis': {
                    'title': "Count",
                        'font': {'size': 16, 'color': 'black', 'family': 'Arial, sans-serif', 'weight': 'bold'}
                },
                'xaxis': {
                    'title': "Genre",
                        'font': {'size': 16, 'color': 'black', 'family': 'Arial, sans-serif', 'weight': 'bold'}
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': {
                    'text': 'Distribution of Message Categories',
                    'font': {'size': 24, 'color': 'black', 'family': 'Arial, sans-serif', 'weight': 'bold'},
                    },
                'yaxis': {
                    'title': "Count",
                        'font': {'size': 16, 'color': 'black', 'family': 'Arial, sans-serif', 'weight': 'bold'}
                },
                'xaxis': {
                    'title': {
                        'text': "Category",
                        'font': {'size': 16, 'color': 'black', 'family': 'Arial, sans-serif', 'weight': 'bold'}
                    },
                    'tickangle': 90
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, 
                           genre_names=genre_names, genre_counts=genre_counts,
                           category_names=category_names, category_counts=category_counts)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    main()