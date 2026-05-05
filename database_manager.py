import sqlite3
import pandas as pd
import json
from datetime import datetime

class DatabaseManager:
    """
    Phase 5 API Bridge.
    Connects the Streamlit UI safely to the `surveileat_intelligence.db`.
    Abstracts SQL queries into clean Python DataFrames for real-time visualization.
    """
    def __init__(self, db_path=r'data\surveileat_intelligence.db'):
        self.db_path = db_path

    def get_connection(self):
        """Returns a fresh SQLite connection."""
        return sqlite3.connect(self.db_path)

    def fetch_all_restaurants(self):
        """Returns a list of unique restaurants currently in the intelligence database."""
        try:
            conn = self.get_connection()
            query = "SELECT DISTINCT restaurant_name FROM reviews_intelligence ORDER BY restaurant_name ASC"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df['restaurant_name'].tolist()
        except:
            return []

    def fetch_aspect_metrics(self, restaurant_name=None):
        """
        Calculates the exact Positive/Negative/Neutral split across the 5 Aspects
        (Taste, Service, Ambience, Quantity, Price) for the given restaurant.
        """
        conn = self.get_connection()
        query = "SELECT sentiment_taste, sentiment_service, sentiment_ambience, sentiment_price, sentiment_quantity FROM reviews_intelligence WHERE rating != 'Live'"
        
        if restaurant_name and restaurant_name != 'All Restaurants':
            query += f" AND restaurant_name = '{restaurant_name}'"
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        metrics = {}
        for col in df.columns:
            # Drop 'Not Mentioned' mathematically before calculating ratios
            valid_sentiments = df[df[col] != 'Not Mentioned'][col]
            if len(valid_sentiments) > 0:
                counts = valid_sentiments.value_counts(normalize=True) * 100
                metrics[col] = counts.to_dict()
            else:
                metrics[col] = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
                
        return metrics

    def fetch_top_dishes(self, restaurant_name=None, limit=10):
        """
        Parses the JSON arrays of extracted dishes and aggregates their frequencies.
        """
        conn = self.get_connection()
        query = "SELECT extracted_dishes FROM reviews_intelligence WHERE rating != 'Live'"
        
        if restaurant_name and restaurant_name != 'All Restaurants':
            query += f" AND restaurant_name = '{restaurant_name}'"
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        dish_counts = {}
        for json_str in df['extracted_dishes']:
            if pd.isna(json_str): continue
            try:
                dishes = json.loads(json_str)
                for dish in dishes:
                    if dish not in dish_counts:
                        dish_counts[dish] = 0
                    dish_counts[dish] += 1
            except:
                pass
                
        # Sort and limit dynamically
        sorted_dishes = sorted(dish_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        return pd.DataFrame(sorted_dishes, columns=["Dish", "Mentions"])

    def fetch_diverse_reviews(self, restaurant_name=None):
        """
        Samples reviews to provide a diverse view:
        If all restaurants: Takes top 6 restaurants. Then for each rating from 5 down to 1,
        picks 1 random review from each of the 6 restaurants (Round-robin pattern).
        If specific restaurant: Picks 1 random review for each rating from 5 to 1.
        """
        conn = self.get_connection()
        
        if restaurant_name and restaurant_name != 'All Restaurants':
            top_restaurants = [restaurant_name]
        else:
            top_restaurants_query = '''
                SELECT restaurant_name 
                FROM reviews_intelligence 
                WHERE rating IN ('1','2','3','4','5') 
                GROUP BY restaurant_name 
                ORDER BY COUNT(*) DESC 
                LIMIT 6
            '''
            top_restaurants_df = pd.read_sql_query(top_restaurants_query, conn)
            top_restaurants = top_restaurants_df['restaurant_name'].tolist()

        diverse_reviews = []
        for rating in ['5', '4', '3', '2', '1']:
            for rest in top_restaurants:
                query = '''
                    SELECT restaurant_name, rating, raw_review 
                    FROM reviews_intelligence 
                    WHERE restaurant_name = ? AND rating = ?
                    ORDER BY RANDOM() LIMIT 1
                '''
                df = pd.read_sql_query(query, conn, params=(rest, rating))
                if not df.empty:
                    diverse_reviews.append(df.iloc[0].to_dict())
                    
        conn.close()
        
        if diverse_reviews:
            return pd.DataFrame(diverse_reviews)
        else:
            return pd.DataFrame()

    def insert_live_review(self, restaurant_name, raw_review, dishes, sentiments, is_fake, **kwargs):
        """
        Dynamically injects a freshly analyzed Live Review from the Streamlit UI directly
        into the intelligence SQLite vault, complete with a timestamp.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Ensure the table schema supports the new columns dynamically
        for col_def in ["created_at TEXT", "is_fake TEXT", "fraud_signals TEXT", "overall_rating REAL"]:
            try:
                cursor.execute(f"ALTER TABLE reviews_intelligence ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass # Column already exists

            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rating = "Live" # Default flag for dynamically analyzed reviews
        dishes_json = json.dumps(list(dishes))
        
        cursor.execute('''
            INSERT INTO reviews_intelligence 
            (restaurant_name, raw_review, rating, extracted_dishes, 
             sentiment_taste, sentiment_service, sentiment_ambience, sentiment_price, sentiment_quantity,
             is_fake, created_at, fraud_signals, overall_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            restaurant_name, raw_review, rating, dishes_json,
            sentiments.get('Taste', {}).get('sentiment', 'Not Mentioned') if isinstance(sentiments.get('Taste'), dict) else sentiments.get('Taste', 'Not Mentioned'),
            sentiments.get('Service', {}).get('sentiment', 'Not Mentioned') if isinstance(sentiments.get('Service'), dict) else sentiments.get('Service', 'Not Mentioned'),
            sentiments.get('Ambience', {}).get('sentiment', 'Not Mentioned') if isinstance(sentiments.get('Ambience'), dict) else sentiments.get('Ambience', 'Not Mentioned'),
            sentiments.get('Price', {}).get('sentiment', 'Not Mentioned') if isinstance(sentiments.get('Price'), dict) else sentiments.get('Price', 'Not Mentioned'),
            sentiments.get('Quantity', {}).get('sentiment', 'Not Mentioned') if isinstance(sentiments.get('Quantity'), dict) else sentiments.get('Quantity', 'Not Mentioned'),
            str(is_fake), timestamp, json.dumps(kwargs.get('fraud_signals', {})), kwargs.get('overall_rating')
        ))
        
        conn.commit()
        conn.close()
        
    def fetch_recent_live_reviews(self, limit=50):
        """
        Retrieves the history of dynamically inserted Live Reviews.
        """
        conn = self.get_connection()
        # Look for the "Live" rating flag we just added
        query = '''
            SELECT restaurant_name, created_at, raw_review, extracted_dishes, 
            sentiment_taste as Taste, sentiment_service as Service, sentiment_ambience as Ambience, 
            sentiment_quantity as Quantity, sentiment_price as Price, is_fake, fraud_signals,
            overall_rating as "Overall Score"
            FROM reviews_intelligence 
            WHERE rating = 'Live' 
            ORDER BY id DESC LIMIT ?
        '''
        try:
            df = pd.read_sql_query(query, conn, params=(limit,))
        except sqlite3.OperationalError:
            # Table structure hasn't been updated with live reviews yet
            return pd.DataFrame()
            
        conn.close()
        return df

if __name__ == "__main__":
    db = DatabaseManager()
    print("Testing DB Connection...")
    print("Restaurants Found:", db.fetch_all_restaurants())
