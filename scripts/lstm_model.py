# lstm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import numpy as np
def lstm_encode(tweets, labels):
    """
    Builds an LSTM model for text classification.
    """   
    # build layers
    vocab_size =100000
    max_len= 40
    embedding_dim = 300
    lstm_units = 128

    
    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(tweets)
        
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, maxlen=max_len).tolist()

    # Verify
    reconstructed_texts = []
    for sequence in padded_sequences:
        reconstructed_texts.append(" ".join([tokenizer.index_word.get(token, '') for token in sequence]))
    
    for original, reconstructed in zip(tweets[:10], reconstructed_texts[:10]):
        print(f"Original: {original}")
        print(f"Reconstructed: {reconstructed}")
   
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    
    return padded_sequences, labels

    
def get_lstm_model():
    vocab_size =100000
    max_len= 40
    embedding_dim = 300
    lstm_units = 128

    model =  Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train(tweets, labels):

    encoded_tweets, labels = lstm_encode(tweets, labels)

    encoded_tweets = np.array(encoded_tweets)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
                                                    encoded_tweets, 
                                                    labels, 
                                                    test_size=0.2,
                                                    random_state=42, 
                                                    shuffle=True
                                                )
    
    model = get_lstm_model()


    for i in range(10):
        print(encoded_tweets[i])
        print(labels[i])
        print(50*"/")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    callbacks =[early_stopping, reduce_lr]
    validation_data = (X_test, y_test)
    
    history = model.fit(X_train, y_train, epochs = 1, batch_size = 16)
    val_preds = model(X_test).numpy().flatten()
    val_preds = (val_preds >= 0.5).astype(int)
    acc = accuracy_score(labels, val_preds)
    (p, r, f, _) = precision_recall_fscore_support(y_pred=val_preds, y_true=labels, average='macro')
    print(f"Acc: {acc:.3f}, Precision: {p:.3f}, Recall: {r:.3f}, F-score: {f:.3f}\n")