import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow.keras.optimizers import Adam, SGD
from transformers import TFAutoModel, AutoTokenizer


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import datetime
import os
from pathlib import Path


class BERTForClassification(tf.keras.Model):

    def __init__(self, params: dict):
        """
        Initializes the BERTForClassification model with hyperparameters.

        Args:
            params (dict): Dictionary containing hyperparameters for the model.
        """

        super().__init__()

        self.params = params

        # Access pre-trained model and tokenizer 
        self.bert = TFAutoModel.from_pretrained(params["MODEL_NAME"])
        self.tokenizer = AutoTokenizer.from_pretrained(params["MODEL_NAME"])


        # Define the classification head
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(8, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(self.params["NUM_CLASSES"], activation='softmax')


    def call(self, inputs):

        """
        Forward pass of the model.
        Args:
            inputs (dict or list): Input dictionary for the model - "input_ids" and "attention_masks" as keys
        Returns:
            tf.Tensor: Model output logits
        """

        x = self.bert(inputs)[1]  # Extract the output from the second layer (sequence embedding)
        
        # Classification head for multi-class classification
        x = self.dense1(x)
        #x = self.dropout1(x)
        x = self.dense2(x)
        #x = self.dropout2(x)
        x = self.dense3(x)
        #x = self.dropout3(x)
        outputs = self.output_layer(x)
        return outputs


    def tokenize(self, batch_data):
        """
        Tokenizes a batch of text data using the pre-trained tokenizer.

        Args:
            batch_data (list): List of text strings to be tokenized.

        Returns:
            list: List containing tokenized input ids and attention masks.
        """
        encoded_text = self.tokenizer(   
                                batch_data, 
                                padding=True, 
                                max_length=self.params["SEQ_LENGTH"], 
                                truncation=True, 
                                return_tensors='tf'
                            )
        return [
                encoded_text['input_ids'],
                encoded_text['attention_mask']
                ]


    def train(self, tweets, labels):
        """
        Trains the model on the provided data.

        Args:
            tweets (list): List of text strings for training and validation.
            labels (list): List of labels corresponding to the tweets for training and validation.

        Returns:
            keras.callbacks.History: Training history object.

        Steps:
        ** Train/Test Split of the tweets and labels.**
        ** Encoding the tweets using pre-trained BERT-based tokenizer and accessing a list of tensors 
            as input_ids and attention_masks. Similarly, One-hot encoding for the labels with depth as
            the number of classes (i.e., 2 in this case - Positive and Negative but can be extended 
            to include the Neutral sentiment as the third class).**
        ** Summarize the model **
        ** Compile the model with appropriate loss_fn, optimizer and metrics**
        ** Fit, train and return the History object. **
        """
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
                                                    tweets, 
                                                    labels, 
                                                    test_size=self.params["TEST_SIZE"],
                                                    random_state=42, 
                                                    shuffle=True
                                                )

        X_train_encoded = self.tokenize(X_train)
        X_test_encoded = self.tokenize(X_test)


        print(X_train_encoded)
        

        y_train_hot = tf.one_hot(y_train, depth=self.params["NUM_CLASSES"])
        y_test_hot = tf.one_hot(y_test, depth=self.params["NUM_CLASSES"])


        # Compile the Model
        self.compile(
                    loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy']
                )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)

        history = self.fit(
                        x=X_train_encoded, 
                        y=y_train_hot,
                        validation_data=(X_test_encoded, y_test_hot),
                        epochs=self.params["EPOCHS"],
                        batch_size=self.params["BATCH_SIZE"],
                        callbacks=[early_stopping]
                    )
        return history


    def save_model_weights(self):
        """Saves a TensorFlow model weight with unique identifier based on timestamp and params for future 
        inference.
        """
        # Create unique path based on timestamp and hyperparameters
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # hyperparam_string = "_".join([f"{k}-{v}" for k, v in self.params.items()])
        unique_name = f"{now}.weights.h5"

        # Create pathlib object for the save path
        os.makedirs(self.params["MODEL_DIR"], exist_ok=True)
        save_path = Path(self.params["MODEL_DIR"]) / unique_name

        self.save_weights(save_path)
        print(f"Model saved to: {save_path}")
        return save_path






