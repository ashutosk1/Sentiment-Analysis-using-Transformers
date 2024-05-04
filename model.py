import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow.keras.optimizers import Adam, SGD


import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import tqdm

import constants

def split_train_test_data(corpus, labels):
    """ Splits the Training and Test Data in the proportion of the 'test_size'
        Returns:
            X_train, y_train, X_test, y_test
    """
    X_train, X_test, y_train, y_test= train_test_split(corpus, labels, test_size = constants.TEST_SIZE, random_state = 42)
    y_train = np.array(y_train, dtype="float32").reshape(-1,1)
    y_test = np.array(y_test, dtype="float32").reshape(-1,1)
    logging.info("Size of Training Data: %d, Size of Test Data: %d", len(X_train), len(X_test))
    
    return X_train, y_train, X_test, y_test


# def encoder(data, tokenizer, max_length):
#     """
#     """
#     input_ids=[]
#     attention_masks=[]

#     for i in range(len(data)):
#         encoded = tokenizer.encode_plus(
#                                         data[i], 
#                                         truncation=True,
#                                         max_length=constants.SEQ_LENGTH, 
#                                         padding="max_length",
#                                         add_special_tokens=True
#                                         )
#         input_ids.append(encoded["input_ids"])
#         attention_masks.append(encoded["attention_mask"])
    
#     return np.array(input_ids), np.array(attention_masks)
    


def encoder(data, model_name):
    """
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, force_download=True)
    input_ids=[]
    attention_masks=[]

    for i in (range(len(data))):
        encoded= tokenizer(data, 
                            truncation=True,                
                            max_length=constants.SEQ_LENGTH, 
                            padding="max_length",
                            add_special_tokens=True
                            )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

        if ((i+1)%100==0): print(f"PROGRESS: {i+1}")

    return np.array(input_ids), np.array(attention_masks)




def build_model(pre_trained_model, input_length):
    """
    """
    input_ids= tf.keras.Input(shape=(input_length,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(input_length,), dtype='int32')

    x_layer = pre_trained_model([input_ids, attention_masks])[-1]

    if len(x_layer.get_shape())> 2: x_layer = tf.squeeze(x_layer[:, -1:, :], axis=1)

    x_layer = tf.keras.layers.Dense(32, activation='relu')(x_layer)
    x_layer = tf.keras.layers.Dropout(0.2)(x_layer)
    x_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x_layer)

    model = tf.keras.models.Model(inputs= ([input_ids, attention_masks]) ,outputs = x_layer)
    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
    return model

