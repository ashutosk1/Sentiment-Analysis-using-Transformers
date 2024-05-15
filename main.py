import constants 
from model import BERTForClassification, load_pretrained_model, load_pretrained_tokenizer, train
from classifier import TextClassifier
from preprocess import preprocess_data

from pathlib import Path

import pickle

if __name__ == "__main__":
    
    try:
        with open("corpus.pickle", "rb") as f:
            corpus=pickle.load(f)
            print("Loaded corpus successfully")

        with open("label.pickle", "rb") as f:
            labels= pickle.load(f)
            print("Loaded labels successfully")
    except:
        print("Some Error encountered.")

    # Classify and Train
    classifier = BERTForClassification()
    history = train(classifier, corpus[:constants.NUM_EXAMPLES], labels[:constants.NUM_EXAMPLES])

    

    
    