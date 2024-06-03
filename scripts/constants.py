# PATHS
DATA_PATH = "/content/drive/MyDrive/Sentiment140/training.1600000.processed.noemoticon.csv"
CORPUS_DIR = "/content/drive/MyDrive/Sentiment140/CORPUS_DATA_DIR_LSTM"


NUM_EXAMPLES = 100000
# MODEL PARAMS
PARAMS = { 
    "TEST_SIZE"       : 0.20,
    "BATCH_SIZE"      : 16,
    "SEQ_LENGTH"      : 100,
    "EPOCHS"          : 2,
    "LEARNING_RATE"   : 1e-5,
    "NUM_CLASSES"     : 2,
    "MODEL_NAME"      : "bert-base-uncased",
    "MODEL_DIR"       : "../MODEL_DIR"
}


#LIST OF EMOJIS WITH THEIR RESPECTIVE MEANINGS
LIST_OF_EMOJIS = {
            ':)': 'smile', 
            ':-)': 'smile', 
            ';d': 'wink', 
            ':-E': 'vampire', 
            ':(': 'sad',
            ':-(': 'sad', 
            ':-<': 'sad', 
            ':P': 'raspberry', 
            ':O': 'surprised',
            ':-@': 'shocked', 
            ':@': 'shocked',
            ':-$': 'confused', 
            ':\\': 'annoyed',
            ':#': 'mute', 
            ':X': 'mute', 
            ':^)': 'smile', 
            ':-&': 'confused', 
            '$_$': 'greedy',
            '@@': 'eyeroll', 
            ':-!': 'confused', 
            ':-D': 'smile', 
            ':-0': 'yell', 
            'O.o': 'confused',
            '<(-_-)>': 'robot', 
            'd[-_-]b': 'dj', 
            ":'-)": 'sadsmile', 
            ';)': 'wink',
            ';-)': 'wink', 
            'O:-)': 'angel',
            'O*-)': 'angel',
            '(:-D': 'gossip', 
            '=^.^=': 'cat'
                 }


# LIST OF FREQUENTLY USED INTERNET WORDS AND THEIR RESPECTIVE MEANINGS
LIST_OF_INTERNET_WORDS = {
                    "afaik": "as far as i know",
                    "afair": "as far as i remember",
                    "afk":  "away from keyboard",
                    "b/c": "because",
                    "fyi": "for your information",
                    "idk":  "i don't know",
                    "imo": "in my opinion",
                    "lol": "laughing out loud",
                    "omg": "oh my god",
                    "pov": "point of view",
                    "tyt": "take your time",
                    "ttyl": "talk to you later",
                    "wfm": "works for me",
                    "ymmd": "you made my day"
                     }


# EXCLUDED SENTIMENT RELEVANT STOPWORDS 
LIST_OF_SENTIMENT_RELEVANT_STOPWORDS = {
                                "no", 
                                "not", 
                                "nor"
                                }


# REPLACED LETTERS WITH THEIR RESPECTIVE MEANINGS
LIST_OF_REPLACED_LETTERS = {
                        "%": "percent",
                        "+": "plus",
                        "$": "dollar",
                    }