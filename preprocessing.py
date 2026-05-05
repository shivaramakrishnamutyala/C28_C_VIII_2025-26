import pandas as pd
import re
from sentence_transformers import SentenceTransformer

class TextPreprocessor:
    """
    V2 High-Intelligence Preprocessor.
    Drops destructive lemmatization/stopwords. 
    Preserves true semantic context, slang, and punctuation for Transformer models.
    """
    def __init__(self, model_name=None):
        # Load the SOTA sentence transformer for dense vector generation
        self.encoder = None
        if model_name:
            try:
                self.encoder = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer '{model_name}'. Exception: {e}")

    def clean_text(self, text):
        """
        Lightweight Semantic cleaning. Only strips formatting noise.
        """
        if pd.isna(text):
            return ""
            
        text = str(text)
        
        # 1. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 2. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 3. Strip excessive whitespace but KEEP punctuation for context
        text = " ".join(text.split())
        
        return text

    def generate_embeddings(self, text_list):
        """
        Converts a list of strings into high-dimensional semantic vectors.
        """
        if not self.encoder:
            raise ValueError("SentenceTransformer encoder is not loaded.")
        
        print(f"Generating embeddings for {len(text_list)} items...")
        embeddings = self.encoder.encode(text_list, show_progress_bar=True)
        return embeddings

    def process_dataframe(self, df, text_column='review_text', output_column='cleaned_review', embed_column='embedding'):
        """
        Cleans the text and attaches the dense vector representation.
        """
        print(f"Deep Preprocessing {len(df)} rows from column '{text_column}'...")
        
        # 1. Clean Text (No data destruction)
        df[output_column] = df[text_column].apply(self.clean_text)
        
        # 2. Generate Vector Embeddings
        if self.encoder:
            df[embed_column] = list(self.generate_embeddings(df[output_column].tolist()))
            
        print("Semantic Preprocessing & Vectorization complete!")
        return df

if __name__ == "__main__":
    # Test the new pipeline
    sample_texts = [
        "The ambience was good, food was quite good . had Saturday lunch.",
        "I DO NOT recommend this place! The chicken was terrible.",
        "Waiter Soumen Das was really courteous and helpful.",
        "The penne pasta wasn't fully cooked, but the music was amazing :) 10/10!"
    ]
    
    preprocessor = TextPreprocessor()
    df = pd.DataFrame({'review_text': sample_texts})
    
    processed_df = preprocessor.process_dataframe(df)
    
    print("\nTesting V2 Preprocessor Output:\n" + "-"*30)
    for i, row in processed_df.iterrows():
        print(f"Original : {row['review_text']}")
        print(f"Cleaned  : {row['cleaned_review']}")
        print(f"Embedding: Shape {row['embedding'].shape}\n")
