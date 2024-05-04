import string
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import constants
from classifier import TextClassifier

# Import Utilities
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from tqdm import tqdm



def preprocess_data():
    """Loads, preprocesses, and returns the sentiment data.

    Returns:
      tuple: A tuple containing the preprocessed corpus and sentiment labels.
     """
    
    # Read Data from the CSV file
    data = pd.read_csv(constants.DATA_PATH, header=None, sep = ',', encoding='latin-1', engine='python')
    logging.info("Data loaded successfully from path: %s. Found %d rows.", constants.DATA_PATH, len(data))

    # Setting Data Column Names as the loading was done with Header as None.
    data.columns=["sentiment", "id", "timestamp", "flag", "username", "message"]


    # Mapping the labels of Sentiments in a binary form with '0' for negative and '1' for positive.
    sentiment_label_conversion = {0:0, 4:1}
    data["sentiment"]= data["sentiment"].map(sentiment_label_conversion)

    # Extracting the Corpus and Sentiment Labels
    corpus= list(data.message)
    sentiment= list(data.sentiment)

    # Placeholder for improved corpus after preprocessing the Text of the Messages
    corpus_preprocess = []

    for i, text in tqdm(enumerate(corpus)):
        # Pre-process the text 
        preprocessed_text = preprocess_text(text)

        # if ((i+1) % 10000==0): print(preprocessed_text)
        corpus_preprocess.append(preprocessed_text)

    logging.info("Data Preprocessed successfully for %d rows and %d labels.", len(corpus_preprocess), len(sentiment))
    # Length of Sequences
    max_seq_length=int(np.max([len(corpus_preprocess[i]) for i in range(len(corpus_preprocess))]))
    avg_seq_length=int(np.average([len(corpus_preprocess[i]) for i in range(len(corpus_preprocess))]))
    dev_seq_length= int(np.std([len(corpus_preprocess[i]) for i in range(len(corpus_preprocess))]))
    logging.info("Maximum Length of Sequence: %d\n Average length of Sequence: %d\n Std Deviation: %d", max_seq_length, avg_seq_length,dev_seq_length)
    
    return corpus_preprocess, sentiment


def preprocess_text(text):
    """  Preprocesses a single text message using the Methods of TextClassifier Class.
    Returns: 
        str: The processed text
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

                if text_classifer.is_emoji_not_punctuation(word):
                    word=constants.LIST_OF_EMOJIS[word]

                if text_classifer.is_url(word.lower()):
                    word="url"

                if text_classifer.is_username(word.lower()):
                    word="usermention"

                messages+=(text_classifer.filter_punctuation(word.lower()).strip()+ " ")

    return messages.strip()
















