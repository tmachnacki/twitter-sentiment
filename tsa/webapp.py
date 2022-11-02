"""
EECS 486 - Final Project
webapp.py

This file will run the web application for the UI.
"""

import matplotlib
matplotlib.use('Agg') # used to resolve threading issue with matplotlib GUIs in web apps

import flask
import tweepy
import json
import datetime
import helper
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import pathlib


# import Twitter API keys
api_keys_file = open("twitter_api_keys.json", "r")
api_keys = json.load(api_keys_file)["eecs486-project"]

client = tweepy.Client(bearer_token=api_keys["bearer_token"], wait_on_rate_limit=True)
app = flask.Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    context = {
        "min_date": datetime.date.today() - datetime.timedelta(days=5),
        "max_date": datetime.date.today(),
        "error": ""
    }
    
    if flask.request.method == "POST":
        query = flask.request.form["query"]

        if not query:
            context["error"] = "Error: query must be non-empty"
            return flask.render_template("homepage.html", **context)

        start = datetime.datetime.fromisoformat(flask.request.form["start"])
        end = datetime.datetime.fromisoformat(flask.request.form["end"])
        tweets_per_day = int(flask.request.form["tweets_per_day"])
        
        if end < start:
            context["error"] = "Error: End Date must be after Start Date"
            return flask.render_template("homepage.html", **context)

        search(query, start, end, tweets_per_day)
        return flask.redirect(flask.url_for("results", query=query, start=start, end=end, tweets_per_day=tweets_per_day))
    
    return flask.render_template("homepage.html", **context)

@app.route("/results/<query>/<start>/<end>/<tweets_per_day>/", methods=["GET"])
def results(query, start, end, tweets_per_day):
    context = {
        "query": query,
        "start": start,
        "end": end,
        "tweets_per_day": tweets_per_day,
    }
    return flask.render_template("results.html", **context)
    
@app.route("/about/", methods=["GET", "POST"])
def about():
    return flask.render_template("about.html")


@app.route("/usage/", methods=["GET"])
def usage():
    return flask.render_template("usage.html")


def search(query, start, end, tweets_per_day):
    tweet_text = []
    tweet_counts = []

    num_days = (end - start).days
    date_list = [start + datetime.timedelta(days=x) for x in range(num_days + 1)]

    # for each date in range, get tweets and tweet count for query
    for date in date_list:
        day_end = date.replace(hour=23, minute=59, second=59)

        if date.date() == datetime.date.today():
            minus_15 = datetime.datetime.now() - datetime.timedelta(seconds=15)
            day_end = date.replace(hour=minus_15.hour, minute=minus_15.minute, second=minus_15.second)

        tweets = client.search_recent_tweets(query=query,
                                            max_results=tweets_per_day,
                                            tweet_fields="text",
                                            start_time=date,
                                            end_time=day_end)

        count = client.get_recent_tweets_count  (query=query,
                                                granularity="day",
                                                start_time=date,
                                                end_time=day_end)

        tweet_counts.append(count.data[0]["tweet_count"])
        tweet_text.append(i.text for i in tweets[0])

    tweet_sentiment = [helper.generate_predictions(i) for i in tweet_text]
    avg_sentiment = [sum(i) / len(i) for i in tweet_sentiment]

    if not os.path.exists("static/images"):
        os.mkdir("static/images")
    
    # create plots
    x = np.array(date_list)
    y = np.array(tweet_counts)
    plt.plot(x, y, marker="o")
    plt.xticks(x)
    #plt.title("Total Tweet Count vs Date for \"{}\"".format(query))
    plt.xlabel("Date")
    plt.ylabel("Total Tweet Count")
    for i, j in zip(x, y): plt.text(i, j, str(j))
    plt.savefig("static/images/tweet_counts.jpg")
    plt.clf()

    y = np.array(avg_sentiment)
    #plt.title("Average Sentiment vs Date for \"{}\"".format(query))
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment (0 = Neg, 4 = Pos)")
    for i, j in zip(x, y): plt.text(i, j, str(round(j, 2)))
    plt.plot(x, y, marker="o")
    plt.xticks(x)
    plt.ylim(0, 4)
    plt.savefig("static/images/avg_sentiment.jpg")
    plt.clf()

    sentiment_counter = Counter()
    for i in tweet_sentiment:
        sentiment_counter.update(i)

    x = np.array(["Negative", "Neutral", "Positive"])
    y = np.array([sentiment_counter[0], sentiment_counter[2], sentiment_counter[4]])
    #plt.title("Count of Analyzed Tweets vs Sentiment for \"{}\"".format(query))
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    for i, j in zip(x, y): plt.text(i, j, str(j))
    plt.bar(x, y)
    plt.savefig("static/images/sentiment_counts.jpg")
    plt.clf()
    
@app.route('/images/<file>', methods=["GET"])
def return_img(file):
    """Functionaly for static image files."""
    
    # resolve path at /TSA/static/imgaes/<filename>
    file_path = pathlib.Path(__file__).resolve().parent/'static'/'images'/file
    return flask.send_file(file_path, attachment_filename=file)

    
@app.route('/documents/<file>', methods=["GET"])
def return_doc(file):
    """Functionaly for static document files."""
    # resolve path at /TSA/static/documents/<filename>
    file_path = pathlib.Path(__file__).resolve().parent/'static'/'documents'/file
    return flask.send_file(file_path, attachment_filename=file)
