"""
EECS 486 
helper.py

This file will contain helper functions needed for multiple files.

pysentimiento reference:
{perez2021pysentimiento,
      title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks},
      author={Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
      year={2021},
      eprint={2106.09462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

from pysentimiento import create_analyzer

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


def generate_predictions(tweet_text):
    """Generates label predictions as list of values (0 = Negative, 2 = Neutral, 4 = Positive) given a list of tweets."""
    # apply prediction frunction 
    return [get_pred(x) for x in tweet_text]
    