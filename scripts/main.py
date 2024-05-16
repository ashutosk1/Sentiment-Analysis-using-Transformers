import constants 
from model import BERTForClassification
from preprocess import preprocess_data, load_or_preprocess_corpus

from pathlib import Path
import pickle

if __name__ == "__main__":
    

    # Access params from the constants.py file
    filepath        = constants.DATA_PATH
    corpus_dir      = constants.CORPUS_DIR
    num_examples    = constants.NUM_EXAMPLES
    params          = constants.PARAMS

    """Check for the preprocessed corpus of tweets and their correponding labels. If the corpus is already
    saved in the pickle format, access it, otherwise run the preprocessing for the cleaning of tweets and
    extraction of labels.
    """
    preprocessed_corpus, sentiment_labels = load_or_preprocess_corpus(corpus_dir, preprocess_data, filepath, num_examples)
    
    """ Instantiate the BERT-based classifier object from the `BERTForClassification` class. Use hyperparams 
    as input parameters involved in building, compiling and training the classifier.
    """
    classifier = BERTForClassification(params)
    history = classifier.train(preprocessed_corpus, sentiment_labels)
    classifier.save_model_weights()

    
    

    
    