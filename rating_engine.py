import numpy as np
import pandas as pd
import json

DEFAULT_ASPECT_WEIGHTS = {
    'taste': 0.40,
    'service': 0.25,
    'ambience': 0.15,
    'price': 0.10,
    'quantity': 0.10
}

def compute_aspect_score(sentiment, confidence):
    """
    Computes the weighted contribution of a single aspect.
    """
    return sentiment * confidence

def compute_authenticity_weight(fake_probability):
    """
    Calculates the AuthenticityWeight = 1 - FakeProbability.
    """
    return 1.0 - fake_probability

def compute_time_decay(review_age_days, decay_lambda=0.05):
    """
    Exponential decay based on review age.
    """
    return np.exp(-decay_lambda * review_age_days)

def compute_review_score(review_row, weights=DEFAULT_ASPECT_WEIGHTS):
    """
    Computes a dynamically normalized AspectScore.
    Aspects with confidence > 0 are considered 'mentioned'.
    """
    score = 0.0
    active_weight_sum = 0.0
    
    for aspect, weight in weights.items():
        sent_col = f"{aspect}_sentiment"
        conf_col = f"{aspect}_confidence"
        sentiment = review_row.get(sent_col, 0.0)
        confidence = review_row.get(conf_col, 0.0)
        
        if confidence > 0.0:
            active_weight_sum += weight
            score += compute_aspect_score(sentiment, confidence) * weight
            
    if active_weight_sum > 0.0:
        return score / active_weight_sum
    return 0.0

def get_stars_from_rating(rating):
    """
    Returns an HTML representation of the stars filled to the exact percentage.
    """
    rating_clamped = max(0.0, min(5.0, rating))
    percent = (rating_clamped / 5.0) * 100
    
    return f'''
    <span style="display: inline-block; position: relative; color: #333; font-size: inherit;">
        ★★★★★
        <span style="color: #FFD700; position: absolute; top: 0; left: 0; overflow: hidden; width: {percent}%; white-space: nowrap;">
            ★★★★★
        </span>
    </span>
    '''.strip()

def compute_restaurant_rating(reviews_df, weights=DEFAULT_ASPECT_WEIGHTS):
    """
    Computes mathematical restaurant rating aggregating multiple factors.
    Returns JSON.
    """
    if reviews_df is None or reviews_df.empty:
        return json.dumps({
            "overall_rating_stars": "N/A",
            "overall_rating_numeric": 0.0,
            "rating_confidence": 0.0,
            "review_count": 0,
            "aspect_summary": {}
        }, indent=4)
        
    num_reviews = len(reviews_df)
    
    # 1. Compute Authentic Weight
    reviews_df['authenticity_weight'] = compute_authenticity_weight(reviews_df['fake_probability'].astype(float))
    
    # 2. Compute Review (Aspect) Score
    reviews_df['aspect_score'] = reviews_df.apply(lambda row: compute_review_score(row, weights), axis=1)
    
    # 3. Adjusted Review Score (Aspects penalized by Fake Probability)
    reviews_df['adjusted_review_score'] = reviews_df['aspect_score'] * reviews_df['authenticity_weight']
    
    # 4. Compute Time Decay Factor
    reviews_df['time_weight'] = compute_time_decay(reviews_df['review_age_days'].astype(float))
    
    # 5. Final Review Score (Time penalized)
    reviews_df['final_review_score'] = reviews_df['adjusted_review_score'] * reviews_df['time_weight']
    
    # + NEW: Calculate dynamically composite denominator for a true Weighted Average
    reviews_df['voting_power'] = reviews_df['authenticity_weight'] * reviews_df['time_weight']
    total_voting_power = reviews_df['voting_power'].sum()
    
    # 6. Restaurant Rating Aggregation (True Weighted Average)
    if total_voting_power > 0:
        restaurant_score = reviews_df['final_review_score'].sum() / total_voting_power
    else:
        restaurant_score = 0.0  # Failsafe baseline
    
    # Normalize overall score to 0-5
    normalized_overall_rating = ((restaurant_score + 1) / 2.0) * 5.0
    normalized_overall_rating = max(0.0, min(5.0, normalized_overall_rating))
    
    # 7. Rating Confidence
    avg_fake_ratio = reviews_df['fake_probability'].astype(float).mean()
    
    conf_cols = [f"{aspect}_confidence" for aspect in weights.keys() if f"{aspect}_confidence" in reviews_df.columns]
    if conf_cols:
        avg_sentiment_confidence = reviews_df[conf_cols].astype(float).mean().mean()
    else:
        avg_sentiment_confidence = 0.0
        
    rating_confidence = min(
        1.0, 
        np.log10(num_reviews + 1) * avg_sentiment_confidence * (1.0 - avg_fake_ratio)
    )
    
    # 8. Aspect Summaries
    aspect_summaries = {}
    for aspect in weights.keys():
        sent_col = f"{aspect}_sentiment"
        conf_col = f"{aspect}_confidence"
        
        if sent_col in reviews_df.columns and conf_col in reviews_df.columns:
            active_mask = reviews_df[conf_col].astype(float) > 0.0
            active_df = reviews_df[active_mask]
            
            if not active_df.empty:
                # Mathematical transition to active weighted averages
                raw_contributions = compute_aspect_score(active_df[sent_col].astype(float), active_df[conf_col].astype(float))
                weighted_contributions = raw_contributions * active_df['voting_power']
                active_voting_power = active_df['voting_power'].sum()
                
                if active_voting_power > 0:
                    aspect_avg = weighted_contributions.sum() / active_voting_power
                else:
                    aspect_avg = 0.0
            else:
                aspect_avg = 0.0
                
            normalized_aspect_rating = ((aspect_avg + 1) / 2.0) * 5.0
            normalized_aspect_rating = max(0.0, min(5.0, normalized_aspect_rating))
            aspect_summaries[aspect] = float(round(normalized_aspect_rating, 2))
        else:
            aspect_summaries[aspect] = 2.5
            
    result = {
        "overall_rating_stars": get_stars_from_rating(normalized_overall_rating),
        "overall_rating_numeric": float(round(normalized_overall_rating, 2)),
        "rating_confidence": float(round(rating_confidence, 2)),
        "review_count": int(num_reviews),
        "aspect_summary": aspect_summaries
    }
    
    return json.dumps(result, indent=4)
