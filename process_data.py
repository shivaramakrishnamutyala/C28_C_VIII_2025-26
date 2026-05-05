import pandas as pd
import os
from preprocessing import TextPreprocessor
from embeddings import EmbeddingGenerator
from aspect_extractor import AspectExtractor
from dish_ner import DynamicDishExtractor
from fake_review_detector import FakeReviewDetector

def main():
    data_path = r"d:\MAJOR\data\updated_restaurant.csv"
    output_dir = r"d:\MAJOR\data\processed"
    models_dir = r"d:\MAJOR\models"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading Dataset...")
    df = pd.read_csv(data_path)
    
    # Process the entire dataset
    sample_df = df.copy()
    
    print("\n--- PHASE 1: Text Preprocessing ---")
    preprocessor = TextPreprocessor()
    
    # The clean_text is applied to the 'review_text' column
    # If some reviews are literally missing (NaN), pandas handles that cleanly.
    processed_df = preprocessor.process_dataframe(sample_df, text_column='review_text', output_column='cleaned_review')
    
    print("\nSample of Cleaned Text:")
    print(processed_df[['review_text', 'cleaned_review']].head(2))
    
    print("\n--- PHASE 2: Aspect-Based Sentiment Analysis (ABSA) ---")
    aspect_model = AspectExtractor()
    processed_df = aspect_model.process_dataframe(processed_df, 'review_text')
    
    print("\nSample of Extracted Aspects:")
    print(processed_df[['Aspect_Taste', 'Aspect_Service', 'Aspect_Ambience']].head(2))
    
    print("\n--- PHASE 3: Dynamic Dish Extraction (NER) ---")
    dish_extractor = DynamicDishExtractor()
    processed_df = dish_extractor.process_dataframe(processed_df, text_column='review_text')
    
    print("\nSample of Extracted Dishes:")
    print(processed_df[['review_text', 'Mentioned_Dishes']].head(2))
    
    print("\n--- PHASE 4: Fake Review Detection (XGBoost) ---")
    fake_detector = FakeReviewDetector()
    processed_df = fake_detector.train_model(processed_df)
    
    print("\nSample of Scored Fake Probabilities:")
    print(processed_df[['reviewer_metadata', 'fake_probability_score']].head(2))
    
    print("\n--- PHASE 1: TF-IDF Embeddings ---")
    embedder = EmbeddingGenerator(max_features=5000)
    
    # We fit the vectorizer ONLY on the corpus we have so far
    # The output is a matrix where rows=reviews, columns=TF-IDF word scores
    tfidf_matrix = embedder.fit_transform(processed_df['cleaned_review'].fillna(""))
    
    print("\n--- PHASE 1: Transformer Embeddings (DistilBERT) ---")
    # Generates deep contextual meaning vectors (Phase 1 abstract requirement)
    # CRITICAL: Batch size reduced to 8 to prevent Windows RAM (Out of Memory) silent crashes
    transformer_matrix = embedder.generate_transformer_embeddings(
        processed_df['cleaned_review'].fillna("").tolist(), 
        batch_size=8
    )
    
    # Save the deep embeddings to disk for the Sentiment models
    import numpy as np
    transformer_path = os.path.join(output_dir, "distilbert_embeddings_full.npy")
    np.save(transformer_path, transformer_matrix)
    print(f"DistilBERT embeddings saved to: {transformer_path}")
    
    # Save the processed DataFrame to disk so Phase 2/3 can just read it directly
    processed_data_path = os.path.join(output_dir, "cleaned_reviews_full.csv")
    processed_df.to_csv(processed_data_path, index=False)
    print(f"\nSaved cleaned CSV to: {processed_data_path}")
    
    # Save the trained vectorizer model so we can use it on new reviews later
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    embedder.save_vectorizer(vectorizer_path)
    
    print("\nPhase 1 Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
