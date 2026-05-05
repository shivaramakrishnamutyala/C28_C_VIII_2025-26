import pandas as pd
import sqlite3
import os
import time
from tqdm import tqdm
import json
import torch
import warnings
from src.preprocessing import TextPreprocessor
from src.dish_ner import DynamicDishExtractor
from src.aspect_extractor import AspectSentimentExtractor

# Disable gradients for inference to save massive memory
torch.set_grad_enabled(False)

warnings.filterwarnings('ignore')

def build_intelligence_database(input_csv, output_db):
    print("="*60)
    print("🚀 SurveilEat V2 High-Intelligence Pipeline Initializing...")
    print("="*60)
    
    # 1. Initialize Transformers
    print("\n[1/4] Booting Deep Learning Models (Sentence-Transformers, GLiNER, DeBERTa-v3)...")
    preprocessor = TextPreprocessor()
    dish_extractor = DynamicDishExtractor()
    sentiment_extractor = AspectSentimentExtractor()
    
    # 2. Load Dataset
    print(f"\n[2/4] Loading Dataset: {input_csv}")
    try:
        # Remove testing constraints to process the full 10,000 dataset
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error loading {input_csv}: {e}")
        return
        
    total_reviews = len(df)
    print(f"Total Reviews Found: {total_reviews}")
    
    # 3. Setup SQLite Database
    print(f"\n[3/4] Architecting Relational SQLite Database: {output_db}")
    os.makedirs(os.path.dirname(output_db), exist_ok=True)
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews_intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_name TEXT,
            raw_review TEXT,
            rating TEXT,
            extracted_dishes TEXT,
            sentiment_taste TEXT,
            sentiment_service TEXT,
            sentiment_ambience TEXT,
            sentiment_price TEXT,
            sentiment_quantity TEXT
        )
    ''')
    
    # Check what is already processed to support resuming safely
    cursor.execute('SELECT COUNT(*) FROM reviews_intelligence')
    processed_count = cursor.fetchone()[0]
    
    if processed_count >= total_reviews:
        print("\n✅ Dataset already fully processed!")
        conn.close()
        return
        
    if processed_count > 0:
        print(f"Resuming securely from review index {processed_count}...")
        df = df.iloc[processed_count:]
        
    BATCH_SIZE = 500
    if len(df) > BATCH_SIZE:
        print(f"\n[Batch Mode] Truncating processing to {BATCH_SIZE} reviews for this session.")
        df = df.iloc[:BATCH_SIZE]
        
    # 4. Process Data
    print("\n[4/4] Activating Intelligence Engines: Phase 1 -> Phase 2 -> Phase 3")
    
    start_time = time.time()
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
        restaurant = row.get('restaurant_name', 'Unknown')
        raw_text = row.get('review_text', '')
        rating = str(row.get('rating', ''))
        
        # Skip empty reviews
        if pd.isna(raw_text) or not isinstance(raw_text, str) or raw_text.strip() == "":
            continue
            
        try:
            # Phase 1: Pure NLP Context
            cleaned_text = preprocessor.clean_text(raw_text)
            
            # Phase 2: Dish NER
            dishes = dish_extractor.extract_dishes(cleaned_text)
            dishes_json = json.dumps(list(dishes))
            
            # Phase 3: Aspect Sentiments
            sentiments = sentiment_extractor.extract_aspect_sentiments(cleaned_text)
            
            # Database Injection
            cursor.execute('''
                INSERT INTO reviews_intelligence 
                (restaurant_name, raw_review, rating, extracted_dishes, 
                 sentiment_taste, sentiment_service, sentiment_ambience, sentiment_price, sentiment_quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                restaurant, raw_text, rating, dishes_json,
                sentiments.get('Taste', {}).get('sentiment', 'Not Mentioned'),
                sentiments.get('Service', {}).get('sentiment', 'Not Mentioned'),
                sentiments.get('Ambience', {}).get('sentiment', 'Not Mentioned'),
                sentiments.get('Price', {}).get('sentiment', 'Not Mentioned'),
                sentiments.get('Quantity', {}).get('sentiment', 'Not Mentioned')
            ))
            
            # Commit every 10 rows to safely save progress in case of interruption
            if index % 10 == 0:
                conn.commit()
                
        except Exception as e:
            print(f"\nError processing row {index}: {e}")
            
    # Final Commit
    conn.commit()
    conn.close()
    
    end_time = time.time()
    print(f"\n🎉 Intelligence Processing Complete!")
    print(f"Total Time: {round((end_time - start_time) / 60, 2)} minutes.")
    print(f"Data safely stored in {output_db}")

if __name__ == "__main__":
    INPUT_FILE = r'd:\MAJOR\data\updated_restaurant.csv'
    OUTPUT_DB = r'd:\MAJOR\data\surveileat_intelligence.db'
    build_intelligence_database(INPUT_FILE, OUTPUT_DB)
