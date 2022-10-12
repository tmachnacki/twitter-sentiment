"""
EECS 486 - Group Project
train.py

This file will train machine learning models.
"""


import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import transformers as ppb
from transformers import AutoTokenizer

import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm

import joblib

import warnings
warnings.filterwarnings('ignore')



def read_train_data():
    """Returns a pandas dataframe of the Sentiment140 dataset."""
    # set filename of training data
    fname = "data/training.1600000.processed.noemoticon.csv"

    # set column names
    col_names = ['target','ids','date','flag','user','text']

    # return pandas datafram of data in file
    return pd.read_csv(fname, encoding="ISO-8859-1", names=col_names)


def process_dataframe(df, class_size):
    """Processes a pandas dataframe by dropping unneccessary columns and changing sentiment values to 0, 1."""
    # only store text and sentiment data
    df.drop(['ids','date','flag','user'], axis=1, inplace=True)

    # change target values to 0 = negative sentiment and 1 = positive sentiment
    df['target'] = df['target'].replace(4, 1)

    # add processed text column
    df['processed'] = df['text']

    # get a sample of n from the dataframe
    positiveDF = df[df["target"] == 1].copy()
    negativeDF = df[df["target"] == 0].copy()

    # remake original df with equal classes
    df = (pd.concat([positiveDF[:class_size], negativeDF[:class_size]]).reset_index(drop=True).copy())

    return df


def tokenize_text(df):
    """Tokenizes tweets using the NLTK tokenizer."""
    # initialize an NLTK tweet tokenizer, which will have special behavior for #, @, etc.
    tkn = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)

    # apply tokenizer to dataframe
    df['processed'] = df['processed'].apply(lambda x : tkn.tokenize(x))

    return df


def remove_stopwords(df):
    """Removes stopwords and puntuation from a tokenized tweet."""
    # get stopwords from NLTK and puntuation from string.punctuation
    # nltk.download('stopwords')
    stops = stopwords.words("english")
    punctuation = string.punctuation

    # define helper function to remove stopwords and punctuation
    def remove_stopwords_helper(tokens):
        return [w for w in tokens if (w not in stops and w not in punctuation)]

    # remove stopwords and punctuation
    df['processed'] = df['processed'].apply(lambda x : remove_stopwords_helper(x))

    return df

def stem_text(df):
    """Applies porter stemmer to the tokens."""
    ps = PorterStemmer()
    df['processed'] = df['processed'].apply(lambda x : [ps.stem(w) for w in x])
    return df


def rejoin_text(df):
    """Rejoins the processed text for the vectorizer."""
    # rejoins tokens into single string
    df['text'] = df['processed'].apply(lambda x : ' '.join(w for w in x))
    return df


def preprocess_tweets(df):
    # tokenize
    df = tokenize_text(df)

    # remove stopwords and punctuation
    df = remove_stopwords(df)

    # stem
    df = stem_text(df)

    # rejoin text
    df = rejoin_text(df)

    return df


def generate_features(df, method):
    """Processes the text in tweets."""
    # declare tokenizer
    tkn = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)

    # choose method to generate feature matrix
    if method == "tf":
        # declare count vectorizer
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=tkn.tokenize)
    elif method == "tf.idf":
        # declare tf idf vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=tkn.tokenize)
    elif method == "bert":
        # return features from distilbert
        return generate_features_distilBert(df)
    else:
        return None

    # build matrix
    matrix = vectorizer.fit_transform(df['text'].values.astype('U'))

    return matrix


def generate_features_distilBert(df):
    # declare model class, tokenizer class, pretrained weights
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # declare tokenizer and model
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # get tokens dataframe
    tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # get the max length for padding value
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    # pad all lists of tokens to same size
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # get masked locations
    attention_mask = np.where(padded != 0, 1, 0)

    # get inputs and mask
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    # get output
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # get the feature matrix
    X = last_hidden_states[0][:,0,:].numpy()

    return X


def generate_model(classifier):
    """Generates a classifier."""
    if classifier == "NB":
        model = ComplementNB()
    elif classifier == "SVM":
        model = LinearSVC()
    elif classifier == "Logistic":
        model = LogisticRegression()
    else:
        return None

    return model


def generate_plot(acc, n):
    """Generates a plot for the test accuracies."""
    X = ['Naive-Bayes','Linear SVM','Logistic']
    Y_tf = acc[0]
    Y_tfidf = acc[1]
    Y_bert = acc[2]
    
    X_axis = np.arange(len(X))
    
    l1 = plt.bar(X_axis - 0.25, Y_tf, 0.23, label = 'TF')
    l2 = plt.bar(X_axis, Y_tfidf, 0.23, label = 'TF.IDF')
    l3 = plt.bar(X_axis + 0.25, Y_bert, 0.23, label = 'BERT')

    plt.xticks(X_axis, X)
    plt.ylim(.3, .9)
    plt.bar_label(l1, labels=[round(x, 2) for x in Y_tf])
    plt.bar_label(l2, labels=[round(x, 2) for x in Y_tfidf])
    plt.bar_label(l3, labels=[round(x, 2) for x in Y_bert])

    plt.xlabel("Classification Method")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title(f"Cross-Validation Accuracy for Various Models, n = {n}")
    plt.legend(loc='upper left')
    plt.savefig("Cross-Validation_Error.png")
    plt.close()


def bert(df):
    """Trains bert on the training dataframe."""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, df):

            self.labels = [x for x in df['target']]
            self.texts = [tokenizer(text, padding='max_length', max_length = 512, 
                                    truncation=True, return_tensors="pt") for text in df['text']]

        def classes(self):
            return self.labels

        def __len__(self):
            return len(self.labels)

        def get_batch_labels(self, idx):
            return np.array(self.labels[idx])

        def get_batch_texts(self, idx):
            return self.texts[idx]

        def __getitem__(self, idx):

            batch_texts = self.get_batch_texts(idx)
            batch_y = self.get_batch_labels(idx)

            return batch_texts, batch_y

    class BertClassifier(nn.Module):
        def __init__(self, dropout=0.5):
            super(BertClassifier, self).__init__()

            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(768, 2)
            self.relu = nn.ReLU()

        def forward(self, input_id, mask):

            _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
            dropout_output = self.dropout(pooled_output)
            linear_output = self.linear(dropout_output)
            final_layer = self.relu(linear_output)

            return final_layer

    def train(model, train_data, val_data, learning_rate, epochs):
        train, val = Dataset(train_data), Dataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr= learning_rate)

        if use_cuda:
                model = model.cuda()
                criterion = criterion.cuda()

        for epoch_num in range(epochs):
                total_acc_train = 0
                total_loss_train = 0

                for train_input, train_label in tqdm(train_dataloader):
                    train_label = train_label.to(device)
                    mask = train_input['attention_mask'].to(device)
                    input_id = train_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)
                    
                    batch_loss = criterion(output, train_label)
                    total_loss_train += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc

                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                total_acc_val = 0
                total_loss_val = 0

                with torch.no_grad():
                    for val_input, val_label in val_dataloader:
                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = model(input_id, mask)

                        batch_loss = criterion(output, val_label)
                        total_loss_val += batch_loss.item()
                        
                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
                
                print(
                    f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                    | Val Loss: {total_loss_val / len(val_data): .3f} \
                    | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6

    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.7*len(df)), int(.9*len(df))])
              
    train(model, df_train, df_val, LR, EPOCHS)

    return model


def main():
    # test various models using different processing methods and classifiers

    # set class size, n
    class_size = 2000
    n = class_size * 2

    # read and process data (no preprocessing as bert models need raw text)
    df = read_train_data()
    df = process_dataframe(df, class_size)

    # declare methods/classifiers
    methods = ["tf", "tf.idf", "bert"]
    classifiers = ["NB", "SVM", "Logistic"]

    # init test accuracy
    cross_val_acc = [[0 for i in range(3)] for j in range(3)]

    # get y-labels
    y = np.array(df['target'])

    # loop through combos of methods and classifiers
    for i, method in enumerate(methods):
        # generate features using the method
        X = generate_features(df, method)

        for j, classifier in enumerate(classifiers):
            if method == "bert" and classifier == "NB": 
                continue

            # get model using classification method
            model = generate_model(classifier)
            
            # get cross validation scores
            scores = cross_val_score(model, X, y, cv=5)

            # average cross validation scores
            cross_val_acc[i][j] = round(np.mean(scores), 4)
    
    # generate plot for cross validation accuracy
    generate_plot(cross_val_acc, n)


    # generate naive-bayes model on larger amount of training data for domain-specific testing

    # read and process data
    df = read_train_data()
    df = process_dataframe(df, 200000)

    # do preprocessing steps (remove stopwords, stem, etc.)
    df = preprocess_tweets(df)

    # change data to 0 - negative, 4 - positive to match testing domain
    df['target'] = df['target'].replace(1, 4)

    # get X, y
    X = np.array(df['text'])
    y = np.array(df['target'])

    # make classifier pipeline and fit model
    model = make_pipeline(CountVectorizer(), ComplementNB())
    model.fit(X, y)

    # save model to file
    with open('model/naive_bayes.pkl', 'wb') as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    main()