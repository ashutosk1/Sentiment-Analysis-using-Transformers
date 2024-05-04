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


# PATHS
DATA_PATH = "/home/ashutosk/DL_SENTIMENT_ANALYSIS/SENTIMENT140_DATA_DIR/training.1600000.processed.noemoticon.csv"
MODEL_PATH = "/home/ashutosk/DL_SENTIMENT_ANALYSIS/MODEL_DIR/"


# MODEL PARAMS
BATCH_SIZE = 16
SEQ_LENGTH = 10
MODEL_NAME = 'bert-base-uncased'
TEST_SIZE = 0.10
