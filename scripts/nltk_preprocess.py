import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import pickle
from sklearn.utils import shuffle

import os
from pathlib import Path


    

def preProcess(filepath, num_examples, shuffle_flag=True):
    """
    Preprocesses a CSV sentiment dataset file.

    Args:
        filepath (str): Path to the CSV file containing sentiment data.
        num_examples (int): Number of examples to load from the dataset.
        shuffle_flag (bool, optional): Flag indicating whether to shuffle the data. Defaults to True.

    Returns:
        list, list: A list of preprocessed messages and a list of corresponding sentiment labels.
    """
    # Load the CSV dataset to read the content of the data file
    data = pd.read_csv(filepath, header=None, sep = ',', encoding='latin-1', engine='python')
    
    # Shuffle the dataset to include both kinds of labels.
    if shuffle_flag:
        print("Data is shuffled!")
        data = shuffle(data)

    # Extract the rows upto `num_examples`    
    data = data[:num_examples]
    logging.info("Data loaded successfully from path: %s. Found %d rows.", filepath, len(data))

    # Setting Data Column Names as the loading was done with Header as None.
    data.columns=["sentiment", "id", "timestamp", "flag", "username", "message"]
    data = data [["sentiment", "message"]]

    # Mapping the labels of Sentiments in a binary form with '0' for negative and '1' for positive.
    sentiment_label_conversion = {0:0, 4:1}
    data["sentiment"]= data["sentiment"].map(sentiment_label_conversion)

    

    # Tweet Cleaning:
    """
    Steps:
    1. ** Lowercase conversion. **
    2. ** Remove username of any kind from the tweets. **
    3. ** Remove hashtags of any kind from the tweets. **
    4. ** Remove URLs that start with www, http:// or https://. **
    5. ** Remove all the punctuations and replace it with None. **
    6. ** Use the lemmantizer method to have a common verb form for all the non-stopwords. **
    7. ** Convert the message columns and sentiment labels in the form of lists. **

    """

    def remove_username(text):
        pattern = r'@\w+'
        return re.sub(pattern, "", text)
    
    def remove_hashtags(text):
        pattern = r'#\w+'
        return re.sub(pattern, "", text)
    
    def remove_url(text):
        pattern = pattern = r'http\S+|www\.\S+'
        return re.sub(pattern, "", text)

    def get_filtered_tokens(text):
        lemmatizer = WordNetLemmatizer()
        stop_words = stop_words = set(stopwords.words('english')) -{"no"}
        text = text.lower()
        text = text.translate(str.maketrans({key: None for key in string.punctuation}))  
        tokens = word_tokenize(text)       
        filtered_tokens = [token for token in tokens if token not in stop_words]  
        filtered_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        return filtered_tokens

    # preprocess
    data["message"] = data["message"].str.lower()
    data["message"] = data["message"].apply(lambda text: remove_username(text))
    data["message"] = data["message"].apply(lambda text: remove_hashtags(text))
    data["message"] = data["message"].apply(lambda text: remove_url(text))
    data["message"] = data["message"].apply(lambda text: get_filtered_tokens(text))
    return data["message"].tolist(), data["sentiment"].tolist()

    
def load_or_preprocess_corpus(filepath, corpus_dir, num_examples):
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
        preprocessed_corpus, sentiment_labels = preProcess(filepath, num_examples, shuffle_flag = True)  
        with corpus_path.open("wb") as f:
            pickle.dump(preprocessed_corpus, f)
        with labels_path.open("wb") as f:
            pickle.dump(sentiment_labels, f)
        print("Preprocessed data saved to pickle files.")

    return preprocessed_corpus, sentiment_labels 
        