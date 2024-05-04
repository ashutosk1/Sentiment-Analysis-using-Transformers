import constants 
from model import encoder, split_train_test_data
from classifier import TextClassifier
from preprocess import preprocess_data

import pickle

if __name__ == "__main__":
    # Data clean-up and pre-process
    #corpus_preprocessed, sentiment = preprocess_data()

    # with open("corpus.pickle", "wb") as f:
    #     pickle.dump(corpus_preprocessed, f)
    #     print("Corpus saved as pickle file")
    
    # with open("label.pickle", "wb") as f:
    #     pickle.dump(sentiment, f)
    #     print("Label saved as pickle file")

    try:
        with open("corpus.pickle", "rb") as f:
            corpus_preprocessed=pickle.load(f)
            print("Loaded corpus successfully")

        with open("label.pickle", "rb") as f:
            sentiment= pickle.load(f)
            print("Loaded labels successfully")
    except:
        print("Some Error encountered.")

    X_train, X_test, y_train, y_test = split_train_test_data(corpus_preprocessed[:1000], sentiment[:1000])
    encodings, _= encoder(X_train, 'distilbert-base-uncased')
 
    