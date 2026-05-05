import os
import pandas as pd
import numpy as np

# CRITICAL: Force all heavy HuggingFace/Keras downloads to the D drive
# This guarantees nothing touches the C drive space limits.
os.environ['HF_HOME'] = r'd:\MAJOR\ml_cache\huggingface'
os.environ['KERAS_HOME'] = r'd:\MAJOR\ml_cache\keras'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class SentimentClassifier:
    """
    A robust BiLSTM + CNN + Attention model for restaurant review sentiment.
    Product-ready, scalable architecture.
    """
    def __init__(self, max_words=10000, max_len=150):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None

    def build_model(self):
        """Builds the hybrid BiLSTM-CNN-Attention architecture."""
        input_layer = Input(shape=(self.max_len,))
        
        # 1. Word Embeddings (Learning contextual representation)
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(input_layer)
        
        # 2. Extracting Local Features (Phrases like "not good", "very fast")
        cnn_layer = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedding_layer)
        
        # 3. Extracting Global Context (Long term dependencies in the review)
        bilstm_layer = Bidirectional(LSTM(64, return_sequences=True))(cnn_layer)
        
        # 4. Attention Mechanism (Focusing on the most important words)
        # Self-attention requires Key, Value, Query. We use the BiLSTM output for all three here.
        attention_layer = Attention()([bilstm_layer, bilstm_layer])
        
        # 5. Pooling and Fully Connected Layers
        pooling_layer = GlobalMaxPooling1D()(attention_layer)
        dense_1 = Dense(64, activation='relu')(pooling_layer)
        dropout = Dropout(0.5)(dense_1)
        
        # Output Layer: 3 Classes (Negative, Neutral, Positive)
        output_layer = Dense(3, activation='softmax')(dropout)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])
        print(self.model.summary())
        return self.model

    def prepare_labels(self, ratings):
        """Converts 1-5 star ratings to Negative (0), Neutral (1), Positive (2)."""
        def map_rating_to_sentiment(rating):
            if rating <= 2: return 0  # Negative
            elif rating == 3: return 1  # Neutral
            else: return 2  # Positive
            
        mapped = [map_rating_to_sentiment(float(r)) for r in ratings]
        return self.label_encoder.fit_transform(mapped)

    def train(self, texts, ratings, epochs=5, batch_size=32):
        """
        End-to-end robust training function including data splitting.
        """
        print(f"Tokenizing {len(texts)} texts...")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        print("Preparing target labels...")
        y = self.prepare_labels(ratings)
        
        # Split: 70% Train, 15% Validation, 15% Test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42) # 0.1764 of 85% is ~15%
        
        if self.model is None:
            self.build_model()
            
        print("\n--- Starting Model Training ---")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n--- Evaluating on Unseen Test Dataset ---")
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return history

    def save_model(self, model_dir):
        """Saves the Keras model and the tokenizer."""
        os.makedirs(model_dir, exist_ok=True)
        # We split the save so we don't need TF just to load tokenizer strings later
        self.model.save(os.path.join(model_dir, "sentiment_bilstm.h5"))
        
        with open(os.path.join(model_dir, "sentiment_tokenizer.pkl"), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(os.path.join(model_dir, "sentiment_label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        print(f"Model and Tokenizer successfully saved to {model_dir}")

if __name__ == "__main__":
    # Load the processed data from Phase 1
    data_path = r"d:\MAJOR\data\processed\cleaned_reviews_full.csv"
    print(f"Loading data securely from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop rows where review text or rating is missing
    df = df.dropna(subset=['cleaned_review', 'rating'])
    
    # CRITICAL FIX: Convert rating to numeric, coercing text strings like "Like" into NaN
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    texts = df['cleaned_review'].astype(str).tolist()
    ratings = df['rating'].tolist()
    
    # Initialize and Build
    classifier = SentimentClassifier(max_words=10000, max_len=150)
    classifier.build_model()
    
    # Execute Full Training (70/15/15 split handles inside)
    # Using 3 epochs for now as this is a robust starting point for BiLSTMs
    classifier.train(texts, ratings, epochs=3, batch_size=32)
    
    classifier.save_model(r"d:\MAJOR\models")
