import os
# Global Timeout Injector for Huge AI Weights Over Slow Connections
os.environ['HF_HUB_REQUEST_TIMEOUT'] = '300'
os.environ['REQUESTS_TIMEOUT'] = '300'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
os.environ['HF_HOME'] = os.path.abspath('ml_cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('ml_cache/huggingface')

import streamlit as st
import pandas as pd
import plotly.express as px
from src.database_manager import DatabaseManager
from src.preprocessing import TextPreprocessor
from src.dish_ner import DynamicDishExtractor
from src.aspect_extractor import AspectSentimentExtractor
from src.fake_review_detector import StylometricAuthenticityDetector
from src.rating_engine import compute_restaurant_rating

st.set_page_config(page_title="SurveilEat Intelligence", layout="wide", page_icon="🍽️")

# Cache the Intelligence Engine initialization to prevent reloading weights
@st.cache_resource
def load_intelligence_engines():
    return {
        "preprocessor": TextPreprocessor(),
        "dish_extractor": DynamicDishExtractor(),
        "sentiment_extractor": AspectSentimentExtractor(),
        "fake_detector": StylometricAuthenticityDetector()
    }

# Connect to the SQLite Database
db = DatabaseManager()
restaurants = db.fetch_all_restaurants()

st.title("🍽️ SurveilEat: AI-Driven Restaurant Intelligence")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Global Analytics Dashboard", "🤖 Live AI Review Analysis"])

with tab1:
    # Selected restaurant hardcoded to "All Restaurants" to keep the logic intact without the sidebar filter
    selected_restaurant = "All Restaurants"
    
    # --- RATING ENGINE INTEGRATION ---
    st.markdown("### 🌟 Overall Platform Rating")
    import sqlite3, json
    conn = sqlite3.connect(r'data\surveileat_intelligence.db')
    df = pd.read_sql_query("SELECT * FROM reviews_intelligence WHERE rating != 'Live'", conn)
    conn.close()
    
    if not df.empty:
        smap = {'Positive': 1.0, 'Neutral': 0.0, 'Negative': -1.0, 'Not Mentioned': 0.0}
        for a in ['Taste', 'Service', 'Ambience', 'Price', 'Quantity']:
            al = a.lower()
            if f'sentiment_{al}' in df.columns:
                df[f'{al}_sentiment'] = df[f'sentiment_{al}'].map(smap).fillna(0.0)
            else:
                df[f'{al}_sentiment'] = 0.0
            df[f'{al}_confidence'] = 0.6  # default DB trust

        df['fake_probability'] = df['is_fake'].apply(lambda x: 0.95 if str(x).lower() == 'true' else 0.05)
        
        # Robust historic date merging to prevent mathematical NaN crashes
        if 'created_at' in df.columns and 'timestamp' in df.columns:
            df['created_at'] = df['created_at'].fillna(df['timestamp'])
        elif 'timestamp' in df.columns:
            df['created_at'] = df['timestamp']
        elif 'created_at' not in df.columns:
            df['created_at'] = pd.Timestamp.now()
            
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['created_at'] = df['created_at'].fillna(pd.Timestamp.now())
        
        df['review_age_days'] = (pd.Timestamp.now() - df['created_at']).dt.days.clip(lower=0).fillna(0)
        
    rating_json = compute_restaurant_rating(df)
    rating_data = json.loads(rating_json)
    
    if rating_data and rating_data.get('review_count', 0) > 0:
        score = rating_data['overall_rating_numeric']
        conf = rating_data['rating_confidence']
        reviews = rating_data['review_count']
        stars_visual = rating_data['overall_rating_stars']
        
        st.markdown(f"<div style='text-align: center; font-size: 64px;'>{stars_visual}</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{score} / 5.0 (Based on {reviews} reviews)</h3>", unsafe_allow_html=True)
        
        # DISPLAY ASPECT SUMMARIES HERE
        aspects = rating_data.get('aspect_summary', {})
        if aspects:
            st.markdown("<h3 style='text-align: center; margin-top:20px;'>✨ Aspect Level Ratings</h3>", unsafe_allow_html=True)
            cols = st.columns(len(aspects))
            for i, (asp, asp_score) in enumerate(aspects.items()):
                with cols[i]:
                    try:
                        score_val = float(asp_score)
                    except:
                        score_val = 0.0
                        
                    st.markdown(f"""
                    <div style='text-align:center;'>
                        <h4 style='margin-bottom: 10px; font-family: sans-serif; letter-spacing: 1px;'>{asp.capitalize()}</h4>
                        <h3 style='margin-top: 0px; color: #4A90E2; font-weight: bold;'>{asp_score} <span style='font-size: 14px; color: #888;'>/ 5.0</span></h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
        st.markdown("---")
    else:
        st.info("Not enough data to generate an Overall Platform Rating.")
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Sentiment Ratios: {selected_restaurant}")
        aspect_metrics = db.fetch_aspect_metrics(selected_restaurant)
        
        if aspect_metrics:
            # Reformat for Plotly Bar Chart
            aspect_data = []
            for aspect, scores in aspect_metrics.items():
                aspect_name = aspect.replace("sentiment_", "").capitalize()
                aspect_data.append({"Aspect": aspect_name, "Sentiment": "Positive", "Percentage": scores.get('Positive', 0)})
                aspect_data.append({"Aspect": aspect_name, "Sentiment": "Neutral", "Percentage": scores.get('Neutral', 0)})
                aspect_data.append({"Aspect": aspect_name, "Sentiment": "Negative", "Percentage": scores.get('Negative', 0)})
            
            df_aspects = pd.DataFrame(aspect_data)
            fig = px.bar(df_aspects, x="Aspect", y="Percentage", color="Sentiment", 
                         title="Multi-Task ABSA Polarity", barmode='group',
                         color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"})
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No sentiment data available yet. Please run `pipeline.py`.")
            
    with col2:
        st.subheader("High-Frequency Extracted Dishes")
        top_dishes = db.fetch_top_dishes(selected_restaurant)
        if not top_dishes.empty:
            st.dataframe(top_dishes, hide_index=True, width='stretch')
        else:
            st.info("No dish data available.")
            
    st.subheader("Raw Knowledge Stream")
    recent_reviews = db.fetch_diverse_reviews(selected_restaurant)
    if not recent_reviews.empty:
        st.dataframe(recent_reviews, hide_index=True, width='stretch')

with tab2:
    st.header("Upload Live Review for Analysis")
    st.markdown("Paste a raw, unstructured customer review below to test the High-Intelligence AI pipeline in real-time. It will be permanently added to the Intelligence Database.")
    
    live_review = st.text_area("Customer Review", height=150, placeholder="e.g. The biryani was cold but the waiters were nice...")
    
    if st.button("Activate Phase 1-4 Intelligence", type="primary"):
        if live_review.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Initializing Deep Learning Transformers..."):
                engines = load_intelligence_engines()
                
            with st.spinner("Executing V3 Mathematical Intelligence Pipeline..."):
                cleaned = engines['preprocessor'].clean_text(live_review)
                dishes = engines['dish_extractor'].extract_dishes(cleaned)
                sentiments = engines['sentiment_extractor'].extract_aspect_sentiments(cleaned)
                auth_result = engines['fake_detector'].analyze_authenticity(cleaned, rating="Live", aspect_sentiments=sentiments)
                
                # --- FAKE REVIEW ISOLATION LOGIC ---
                if auth_result.get('classification') == 'FAKE_REVIEW':
                    dishes = []  # Omit extracted dishes directly to secure DB integrity
                    for aspect in sentiments:
                        sentiments[aspect] = {'sentiment': 'Not Mentioned', 'confidence': 0.0}
                        
            # Prepare DataFrame for new rating engine Single Review Mode
            import json
            smap = {'Positive': 1.0, 'Neutral': 0.0, 'Negative': -1.0, 'Not Mentioned': 0.0}
            row_data = {'review_age_days': 0, 'fake_probability': auth_result.get('fraud_score', 0.0)}
            for aspect_title, asp_dict in sentiments.items():
                al = aspect_title.lower()
                row_data[f"{al}_sentiment"] = smap.get(asp_dict['sentiment'], 0.0)
                row_data[f"{al}_confidence"] = asp_dict['confidence']
                
            review_df = pd.DataFrame([row_data])
            rating_json = compute_restaurant_rating(review_df)
            rating_data = json.loads(rating_json)
            review_stars_visual = rating_data['overall_rating_stars']
            overall_numeric = rating_data['overall_rating_numeric']
            
            st.markdown(f"<h3 style='text-align: center;'>Live Review Score: {overall_numeric}/5.0</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 48px; margin-top: -10px; margin-bottom: 20px;'>{review_stars_visual}</div>", unsafe_allow_html=True)
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            
            # Streaming outputs
            with col_a:
                st.subheader("1. Fake Review Auth (5-Signal)")
                
                is_fake = auth_result.get('classification') == 'FAKE_REVIEW'
                f_score = auth_result.get('fraud_score', 0.0)
                if is_fake:
                    st.error(f"🚨 **{auth_result.get('classification')}** (FraudScore: {f_score})")
                else:
                    st.success(f"✅ **{auth_result.get('classification')}** (FraudScore: {f_score})")
                    
                with st.expander("View 5-Signal Matrix Diagnostics"):
                    st.json(auth_result.get('signals', {}))
                    
                st.subheader("2. Complex Dish Extraction (GLiNER)")
                if auth_result.get('classification') == 'FAKE_REVIEW':
                    st.info("🚫 Dish extraction skipped (Fake Review Detected).")
                elif dishes:
                    tags = " ".join([f"<span style='background-color:#4A90E2; padding:6px 14px; border-radius:20px; color:white; margin-right:8px; font-weight:bold; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);'>🍽️ {d.title()}</span>" for d in dishes])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.info("No explicit dishes detected.")
                    
            with col_b:
                st.subheader("3. Word-Level Sentiments (spaCy)")
                if sentiments:
                    st.markdown("<div style='background-color:rgba(128,128,128,0.1); padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
                    for aspect, data in sentiments.items():
                        sentiment = data['sentiment']
                        conf = data['confidence']
                        
                        if sentiment == 'Positive':
                            st.markdown(f"<h4 style='margin:10px 0;'>{aspect}: <span style='color:#2ca02c; font-weight:900;'>🟢 {sentiment}</span> <span style='font-size:12px; color:grey;'>(Conf: {conf})</span></h4>", unsafe_allow_html=True)
                        elif sentiment == 'Negative':
                            st.markdown(f"<h4 style='margin:10px 0;'>{aspect}: <span style='color:#d62728; font-weight:900;'>🔴 {sentiment}</span> <span style='font-size:12px; color:grey;'>(Conf: {conf})</span></h4>", unsafe_allow_html=True)
                        elif sentiment == 'Neutral':
                            st.markdown(f"<h4 style='margin:10px 0;'>{aspect}: <span style='color:#A0A0A0; font-weight:900;'>⚪ {sentiment}</span> <span style='font-size:12px; color:grey;'>(Conf: {conf})</span></h4>", unsafe_allow_html=True)
                        else:
                            if auth_result.get('classification') == 'FAKE_REVIEW':
                                st.markdown(f"<h6 style='margin:10px 0; color:#d62728;'>{aspect}: 🚫 Skipped (Fake)</h6>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h6 style='margin:10px 0; color:grey;'>{aspect}: ➖ Not Mentioned</h6>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No sentiments extracted.")
            
            st.success("✅ Real-Time Intelligence Extraction Complete.")
            
            # Formally commit the Live Analysis to the SQLite Database
            with st.spinner("Injecting mathematical intelligence into Live Database..."):
                 db.insert_live_review(
                     restaurant_name="General",
                     raw_review=live_review,
                     dishes=dishes,
                     sentiments=sentiments,
                     is_fake=auth_result.get('classification') == 'FAKE_REVIEW',
                     fraud_signals=auth_result.get('signals', {}),
                     overall_rating=overall_numeric
                 )
                 
    st.markdown("---")
    st.subheader("Live Analytics Dashboard")
    
    live_history_df = db.fetch_recent_live_reviews()
    if not live_history_df.empty:
        # --- 0. Live Fraud Diagnostics Pie Chart ---
        if 'fraud_signals' in live_history_df.columns:
            fraud_totals = {"AI_Probability": 0, "Burstiness": 0, "Similarity": 0, "Sentiment_Mismatch": 0, "Stylometric_Anomaly": 0}
            valid_counts = 0
            for sig_json in live_history_df['fraud_signals']:
                if pd.isna(sig_json) or not sig_json: continue
                try:
                    sigs = json.loads(sig_json)
                    if not sigs: continue
                    for k in fraud_totals.keys():
                        fraud_totals[k] += sigs.get(k, 0)
                    valid_counts += 1
                except: pass
                
            if valid_counts > 0:
                # Calculate averages
                fraud_avgs = {k: v / valid_counts for k, v in fraud_totals.items()}
                # Create Pie Chart
                df_fraud = pd.DataFrame(list(fraud_avgs.items()), columns=["Signal", "Intensity"])
                # Filter out zeroes for a cleaner pie
                df_fraud = df_fraud[df_fraud["Intensity"] > 0]
                
                if not df_fraud.empty:
                    fig_fraud = px.pie(df_fraud, values="Intensity", names="Signal", 
                                       title="Live Session: Fake Review Vectors (Avg Intensity)",
                                       hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_fraud.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_fraud, width='stretch')
                    st.markdown("<br>", unsafe_allow_html=True)

        # --- 1. Live Aspects Graph ---
        aspect_data = []
        for aspect in ['Taste', 'Service', 'Ambience', 'Quantity', 'Price']:
            if aspect in live_history_df.columns:
                valid_sentiments = live_history_df[live_history_df[aspect] != 'Not Mentioned'][aspect]
                if len(valid_sentiments) > 0:
                    counts = valid_sentiments.value_counts(normalize=True) * 100
                    aspect_data.append({"Aspect": aspect, "Sentiment": "Positive", "Percentage": counts.get('Positive', 0)})
                    aspect_data.append({"Aspect": aspect, "Sentiment": "Neutral", "Percentage": counts.get('Neutral', 0)})
                    aspect_data.append({"Aspect": aspect, "Sentiment": "Negative", "Percentage": counts.get('Negative', 0)})
                else:
                    aspect_data.append({"Aspect": aspect, "Sentiment": "Positive", "Percentage": 0})
                    aspect_data.append({"Aspect": aspect, "Sentiment": "Neutral", "Percentage": 0})
                    aspect_data.append({"Aspect": aspect, "Sentiment": "Negative", "Percentage": 0})
                    
        df_aspects_live = pd.DataFrame(aspect_data)
        fig_aspects = px.bar(df_aspects_live, x="Aspect", y="Percentage", color="Sentiment", 
                     title="Live Session: Aspect Sentiment Polarity", barmode='group',
                     color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"})
        st.plotly_chart(fig_aspects, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- 2. Live Dishes Graph ---
        import json
        dish_counts_live = {}
        stop_dishes = {"bite", "cuisine", "zomato gold", "mutton haleem", "pista house", "shah ghouse", "mutton haleem dish", "soumen das", "soumen", "culinary traditions", "sustenance", "flavors", "culinary masterpieces", "masterpieces", "tapestry", "symphony"}
        for json_str in live_history_df['extracted_dishes']:
            if pd.isna(json_str): continue
            try:
                dishes_list = json.loads(json_str)
                for dish in dishes_list:
                    if dish.lower().strip() in stop_dishes:
                        continue
                    if dish not in dish_counts_live:
                        dish_counts_live[dish] = 0
                    dish_counts_live[dish] += 1
            except:
                pass
                
        if dish_counts_live:
            df_dishes_live = pd.DataFrame(list(dish_counts_live.items()), columns=["Dish", "Mentions"])
            df_dishes_live = df_dishes_live.sort_values(by="Mentions", ascending=False).head(15)
            fig_dishes = px.bar(df_dishes_live, x="Dish", y="Mentions", title="Live Session: Extracted Dishes Frequency", color="Mentions", color_continuous_scale="Blues")
            st.plotly_chart(fig_dishes, width='stretch')

        st.markdown("---")
        
        # Partition data: 'is_fake' is stored as a string "True" / "False" in SQLite
        fake_mask = live_history_df['is_fake'].astype(str).str.lower() == 'true'
        fake_df = live_history_df[fake_mask].copy()
        genuine_df = live_history_df[~fake_mask].copy()

        # 1. Genuine Reviews Table
        st.subheader("Genuine Review History")
        st.markdown("This table displays mathematically verified authentic reviews with full sentiment and entity extraction.")
        
        if 'restaurant_name' in genuine_df.columns:
            display_genuine = genuine_df.drop(columns=['restaurant_name'])
        else:
            display_genuine = genuine_df
            
        if not display_genuine.empty:
            st.dataframe(display_genuine, hide_index=True, width='stretch')
        else:
            st.info("No genuine reviews have been analyzed yet.")

        # 2. Fake Reviews Quarantine Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Isolated Fake Reviews")
        st.markdown("This table quarantines reviews detected as SPAM/Bot-generated by the Authenticity engine.")
        
        if not fake_df.empty:
            # Filter exactly to the 4 requested columns
            fake_cols = ['created_at', 'raw_review', 'is_fake', 'fraud_signals']
            fake_cols_existing = [c for c in fake_cols if c in fake_df.columns]
            display_fake = fake_df[fake_cols_existing]
            st.dataframe(display_fake, hide_index=True, width='stretch')
        else:
            st.success("No fake reviews detected in this session!")
    else:
        st.info("No live reviews have been analyzed yet. Upload one above to start!")
