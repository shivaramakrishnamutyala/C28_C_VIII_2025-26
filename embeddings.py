import pandas as pd
import pickle
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel

# CRITICAL: Force all heavy HuggingFace/Keras downloads to the D drive
os.environ['HF_HOME'] = r'd:\MAJOR\ml_cache\huggingface'

class EmbeddingGenerator:
    """
    Handles generating numerical representations of text.
    For Phase 1, we start with TF-IDF which is great for statistical baseline
    sentiment analysis and aspect term extraction.
    Note: DistilBERT embeddings will be implemented alongside the Deep Learning models.
    """
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)  # Capture single words and pairs (e.g., "not good")
        )
        
    def fit_transform(self, texts):
        """Fits the vectorizer on the texts and transforms them to a matrix."""
        print(f"Fitting TF-IDF Vectorizer on {len(texts)} documents...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
        return tfidf_matrix
        
    def transform(self, texts):
        """Transforms new texts using a pre-fitted vectorizer."""
        return self.vectorizer.transform(texts)
        
    def save_vectorizer(self, filepath):
        """Saves the fitted vectorizer to disk."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {filepath}")
        
    def load_vectorizer(self, filepath):
        """Loads a fitted vectorizer from disk."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from {filepath}")

    def generate_transformer_embeddings(self, texts, batch_size=32):
        """
        Generates deep contextual embeddings using DistilBERT (Phase 1 requirement).
        Returns a numpy array of shape (len(texts), 768).
        """
        print("Loading DistilBERT model for transformer embeddings...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Use GPU if available (though unlikely here, standard PyTorch check)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        all_embeddings = []
        print(f"Generating DistilBERT embeddings for {len(texts)} texts...")
        
        # Process in batches to avoid RAM exhaustion
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and pad to max length expected by DistilBERT
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=150)
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # We take the embedding of the [CLS] token (the first token) as the sentence representation
            # Shape of last_hidden_state: (batch_size, sequence_length, hidden_size)
            cls_embeddings = outputs.last_hidden_state[:, 0, :] 
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
            if i > 0 and i % 500 == 0:
                print(f"Processed {i}/{len(texts)} texts for Transformer embeddings...")

        # Flatten the list of batches into a single numpy matrix
        import numpy as np
        final_embeddings = np.vstack(all_embeddings)
        print(f"DistilBERT Embedding Matrix Shape: {final_embeddings.shape}")
        return final_embeddings

if __name__ == "__main__":
    # Quick Test
    sample_cleaned_texts = [
        "ambience good food quite good saturday lunch",
        "not recommend place chicken terrible",
        "waiter soumen da really courteous helpful"
    ]
    
    generator = EmbeddingGenerator(max_features=100)
    matrix = generator.fit_transform(sample_cleaned_texts)
    print("Vocabulary sample:", list(generator.vectorizer.vocabulary_.keys())[:5])
