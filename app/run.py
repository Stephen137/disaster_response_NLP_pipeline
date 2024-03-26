import json
import plotly
import plotly.graph_objs
import pandas as pd
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    
    """Function that cleans and tokenizes input text 
        
    Args:
        text string
        
        
    Returns: 
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


# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('disaster_response',engine )  
#print(df.columns)

# load model
model = joblib.load("../models/random_forest.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    # Below is an example - modify to extract data for your own visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    language_counts = df.groupby('source_language').count()['message'].sort_values(ascending=False)
    #print(language_counts)
    language_names = list(language_counts.index)
    
    sum_of_columns_sorted = df.iloc[:, 4:40].sum().sort_values(ascending=False)
       
              
    
    # create visuals
    # Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=language_names,
                    y=language_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message source language',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Source Language"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }

            }
        },
        {
            'data': [
                Bar(
                    x=df.columns[4:40],
                    y=sum_of_columns_sorted
                )
            ],

            'layout': {
                'title': 'Distribution of categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }

            }
        },
                       
               
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:40], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()