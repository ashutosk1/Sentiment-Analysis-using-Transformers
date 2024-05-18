import string
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import constants
from classifier import TextClassifier

# Import Utilities
import os
from pathlib import Path
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_data(filepath, num_examples):
    
    """Loads, preprocesses, and returns the sentiment data from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing sentiment data.
    Returns:
        tuple: A tuple containing two lists:
            - preprocessed_corpus (list): List of preprocessed text messages.
            - sentiment_labels (list): List of sentiment labels (0 for negative, 1 for positive).
    """
    
    # Read Data from the CSV file
    data = pd.read_csv(filepath, header=None, sep = ',', encoding='latin-1', engine='python')[:num_examples]
    logging.info("Data loaded successfully from path: %s. Found %d rows.", filepath, len(data))

    # Setting Data Column Names as the loading was done with Header as None.
    data.columns=["sentiment", "id", "timestamp", "flag", "username", "message"]

    # Mapping the labels of Sentiments in a binary form with '0' for negative and '1' for positive.
    sentiment_label_conversion = {0:0, 4:1}
    data["sentiment"]= data["sentiment"].map(sentiment_label_conversion)
    print("Printing from the Traditional preprocessor")
    print(data["message"].head())
    # Extracting the Corpus and Sentiment Labels
    corpus= list(data.message)
    sentiment= list(data.sentiment)

    # Placeholder for improved corpus after preprocessing the Text of the Messages
    corupus_processed = []

    for i, text in tqdm(enumerate(corpus)):
        # Pre-process the text 
        preprocessed_text = preprocess_text(text)

        # if ((i+1) % 10000==0): print(preprocessed_text)
        corupus_processed.append(preprocessed_text)

    logging.info("Data Preprocessed successfully for %d rows and %d labels.", len(corupus_processed), len(sentiment))
    # Length of Sequences
    max_seq_length=int(np.max([len(corupus_processed[i]) for i in range(len(corupus_processed))]))
    avg_seq_length=int(np.average([len(corupus_processed[i]) for i in range(len(corupus_processed))]))
    dev_seq_length= int(np.std([len(corupus_processed[i]) for i in range(len(corupus_processed))]))
    logging.info("Maximum Length of Sequence: %d\n Average length of Sequence: %d\n Std Deviation: %d", max_seq_length, avg_seq_length,dev_seq_length)
    
    return corupus_processed, sentiment


def preprocess_text(text):
    """
    Preprocesses a single text message for sentiment analysis.

    Args:
        text (str): The text message to be preprocessed.

    Returns:
        str: The preprocessed text message.
    """

    words = text.split(" ")

    messages=""
    text_classifer= TextClassifier()
    
    for word in words:
        # Checking for English or Non-English Characters
        if word.strip():
            # check for internet words
            if word.lower() in constants.LIST_OF_INTERNET_WORDS:
                messages += constants.LIST_OF_INTERNET_WORDS[word.lower()]

            # if the word is not in stopwords' set
            if word.lower() not in (set(STOPWORDS) - constants.LIST_OF_SENTIMENT_RELEVANT_STOPWORDS) and len(words)>1:

                word = lemmatizer.lemmatize(word)

                if text_classifer.is_emoji_not_punctuation(word):
                    word=constants.LIST_OF_EMOJIS[word]

                if text_classifer.is_url(word.lower()):
                    word="url"

                if text_classifer.is_username(word.lower()):
                    word="user"

                messages+=(text_classifer.filter_punctuation(word.lower()).strip()+ " ")

    return messages.strip()


def load_or_preprocess_corpus(corpus_dir, preprocess_data, filepath, num_examples):
    """
    Loads the preprocessed corpus and labels from pickle files, or preprocesses the data if not found.

    Args:
        corpus_dir (Path): Path to the directory containing corpus pickle files.
        preprocess_data (function): Function that takes a filepath and returns preprocessed corpus and labels.
        filepath (str): Path to the CSV file containing sentiment data.
        num_examples (int) : Number of rows of the csv file to be extracted for reading and preprocessing.

    Returns:
        tuple: A tuple containing two lists:
            - preprocessed_corpus (list): List of preprocessed text messages.
            - sentiment_labels (list): List of sentiment labels.
    """

    # Create the corpus_dir if it doesn't exist
    os.makedirs(corpus_dir, exist_ok=True)

    corpus_path = Path(corpus_dir) / "tweets.pickle"
    labels_path = Path(corpus_dir) / "labels.pickle"

    # Check if pickle files exist
    if corpus_path.exists() and labels_path.exists():
        print("Loading preprocessed corpus and labels from pickle files...")
        with corpus_path.open("rb") as f:
            preprocessed_corpus = pickle.load(f)
        with labels_path.open("rb") as f:
            sentiment_labels = pickle.load(f)
        print("Loaded corpus and labels successfully!")
    else:
        print("Pickle files not found. Preprocessing data...")
        preprocessed_corpus, sentiment_labels = preprocess_data(filepath, num_examples)  
        with corpus_path.open("wb") as f:
            pickle.dump(preprocessed_corpus, f)
        with labels_path.open("wb") as f:
            pickle.dump(sentiment_labels, f)
        print("Preprocessed data saved to pickle files.")

    return preprocessed_corpus, sentiment_labels















