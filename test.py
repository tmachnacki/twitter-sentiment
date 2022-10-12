"""
EECS 486 - Group Project
test.py

This file will test the machine learning models.
"""

import pandas as pd
import numpy as np

import joblib

from sklearn.metrics import *

from pysentimiento import create_analyzer


def read_data():
    """Returns a pandas dataframe of the Twitter Airline Sentiment dataset."""
    # set filename of training data
    fname = "data/Tweets.csv"

    # return pandas datafram of data in file
    return pd.read_csv(fname, encoding="ISO-8859-1")


def process_dataframe(df, n):
    """Processes a pandas dataframe by dropping unneccessary columns and changing sentiment values to 0, 2, 4."""
    # only store text and sentiment data
    df.drop(df.columns.difference(['text','airline_sentiment']), axis=1, inplace=True)

    # change sentiment values to numerical
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 4)
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 2)

    return df.sample(frac=1)[:n]


def bert_library_model(df):
    """Get the predicitons from the sentiment analysis library."""
    # declare analyzer
    analyzer = create_analyzer(task="sentiment", lang="en")

    # function to get prediction label and convert to numerical value
    def get_pred(x):
        x = analyzer.predict(x)
        pred = x.output
        
        if pred == "POS":
            return 4
        elif pred == "NEG":
            return 0
        else:
            return 2

    # get predictions
    pred = df['text'].apply(lambda x: get_pred(x))

    return np.array(pred)


def main():
    # read in data
    df = read_data()

    # process data
    df = process_dataframe(df, 2000)

    # get true labels
    y_true = np.array(df['airline_sentiment'])

    # get predictions for the bert library model
    y_pred = bert_library_model(df)

    # display performance metrics for the bert library model
    print("Performance Report for Bert Library Model:")
    print(classification_report(y_true, y_pred))

    # change data to binary classification for naive-bayes
    df = df[df['airline_sentiment'] != 2]
    y_true = np.array(df['airline_sentiment'])

    # open naive bayes model for baseline
    with open('model/naive_bayes.pkl', 'rb') as f:
        naive_bayes = joblib.load(f)

    # get predictions for the naive bayes model
    X = np.array(df['text'])
    y_pred = naive_bayes.predict(X)

    # display performance metrics for the bert library model
    print("Performance Report for Naive-Bayes Model:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()