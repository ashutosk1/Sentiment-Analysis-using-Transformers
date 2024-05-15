import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow.keras.optimizers import Adam, SGD


from transformers import TFAutoModel, AutoTokenizer


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import tqdm

import constants



class BERTForClassification(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.bert = TFAutoModel.from_pretrained(constants.MODEL_NAME)
        self.num_classes = constants.NUM_CLASSES
        self.seq_length = constants.SEQ_LENGTH

        # Define the classification head
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(8, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')


    def call(self, inputs):

        x = self.bert(inputs)[1]  # Extract the output from the second layer (sequence embedding)

        # Classification head for multi-class classification
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        outputs = self.output_layer(x)
        return outputs


def tokenize(tokenizer, batch_data, seq_length):
  encoded_text = tokenizer(batch_data, padding=True, max_length=seq_length, truncation=True, return_tensors='tf')
  return [
          encoded_text['input_ids'],
          encoded_text['attention_mask'],
          encoded_text['token_type_ids']
          ]


def one_hot_label(label):
    return tf.one_hot(label, depth=2)

def load_pretrained_model():
    return TFAutoModel.from_pretrained(constants.MODEL_NAME)

def load_pretrained_tokenizer():
    return AutoTokenizer.from_pretrained(constants.MODEL_NAME)


def train(model, tweets, labels):
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=constants.TEST_SIZE, random_state=42, shuffle=True)

    tokenizer = load_pretrained_tokenizer()
    X_train_encoded = tokenize(tokenizer, X_train, constants.SEQ_LENGTH)
    X_test_encoded = tokenize(tokenizer, X_test, constants.SEQ_LENGTH)

    y_train_hot = one_hot_label(y_train)
    y_test_hot = one_hot_label(y_test)

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(x=X_train_encoded, y=y_train_hot,
                      validation_data=(X_test_encoded, y_test_hot),
                      epochs=constants.EPOCHS,
                      callbacks=[early_stopping],
                      batch_size=constants.BATCH_SIZE)
    return history