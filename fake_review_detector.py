import os
import math
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

class StylometricAuthenticityDetector:
    """
    Phase 4 V3: Hybrid 5-Signal Fraud Detection.
    Combines AI classification, grammatical stylometrics (variance/entropy),
    semantic duplicate detection, and sentiment-rating logic to calculate a mathematical FraudScore.
    """
    def __init__(self, roberta_model="Hello-SimpleAI/chatgpt-detector-roberta", sim_model="all-MiniLM-L6-v2"):
        os.environ['HF_HOME'] = r'd:\MAJOR\ml_cache\huggingface'
        
        # INCREASE NETWORK TIMEOUT LIMITS FOR HUGE MODEL DOWNLOADS (Evades WinError 10060)
        os.environ['HF_HUB_REQUEST_TIMEOUT'] = '300'
        os.environ['REQUESTS_TIMEOUT'] = '300'
        
        print("Loading V3 Stylometric Signals...")
        self.ai_classifier = pipeline("text-classification", model=roberta_model, top_k=None)
        
        print(f"Loading Semantic Similarity Space '{sim_model}'...")
        self.encoder = SentenceTransformer(sim_model)
        
        # In-memory vector database representing the recent review history
        # Used for detecting duplicate/bot-farm copy-paste reviews
        self.history_embeddings = []
        
        self.ai_buzzwords = [
            "delve", "tapestry", "testament", "impeccable", "paramount", "culinary mastery",
            "symphony", "exquisite", "absolute pleasure", "highly recommend", "game-changer",
            "delectable", "unforgettable", "meticulous", "exemplary", "nestled", "elevate",
            "bustling", "vibrant", "realm", "profound", "anticipating", "beacon", "gem", "testament to"
        ]

    def _get_ai_prob(self, text):
        prob = 0.0
        try:
            outputs = self.ai_classifier(text[:2000], truncation=True, max_length=512)
            if isinstance(outputs, list):
                iterator = outputs[0] if len(outputs) > 0 and isinstance(outputs[0], list) else outputs
            elif isinstance(outputs, dict):
                iterator = [outputs]
            else:
                iterator = []
                
            for res in iterator:
                if isinstance(res, dict) and res.get('label') in ['ChatGPT', 'Fake']:
                    prob = res.get('score', 0.0)
        except:
            pass
            
        # Add regex/rule-based detection for competitor spam bots which AI models miss
        spam_phrases = ['worst place ever', 'food poisoning', 'bug in my', 'go to', 'much cheaper', 'do not eat here', 'refund', 'terrible']
        lower = text.lower()
        spam_count = sum(1 for p in spam_phrases if p in lower)
        
        # If it severely contains defamatory/spam phrases, artificially boost AI/Bot probability
        if spam_count > 0:
            prob = max(prob, min(1.0, 0.4 + (spam_count * 0.30)))
            
        return prob

    def _get_burstiness(self, text):
        import re
        burst_points = 0.0
        
        # 1. Punctuation clusters (!!!, ???)
        if re.search(r'[!?.]{2,}', text): burst_points += 0.3
        if re.search(r'[!?.]{4,}', text): burst_points += 0.3
            
        # 2. Capitalization Shouting
        words = text.split()
        if words:
            caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
            if caps_ratio > 0.05: burst_points += 0.4
            if caps_ratio > 0.15: burst_points += 0.4
                
        # 3. Buzzwords fallback
        lower = text.lower()
        count = sum(1 for w in self.ai_buzzwords if w in lower)
        burst_points += min(count / 3.0, 1.0)
        
        return min(burst_points, 1.0)

    def _get_similarity(self, text):
        if not self.history_embeddings:
            return 0.0
        try:
            emb = self.encoder.encode([text])
            sims = cosine_similarity(emb, self.history_embeddings)[0]
            # Get highest match out of past history pool
            return float(np.max(sims))
        except:
            return 0.0

    def _get_mismatch(self, rating, aspect_sentiments):
        if not rating or rating == 'Live' or not aspect_sentiments:
            return 0.0
        
        try:
            r = int(float(rating))
        except:
            return 0.0
            
        # Using safely parsed V3 aspect geometries
        pos = sum(1 for k, v in aspect_sentiments.items() if isinstance(v, dict) and v.get('sentiment') == 'Positive')
        neg = sum(1 for k, v in aspect_sentiments.items() if isinstance(v, dict) and v.get('sentiment') == 'Negative')
        
        # 4/5 star but literally only complaining -> Mathematical mismatch 1.0 fraud
        if r >= 4 and neg > 0 and pos == 0:
            return 1.0
        # 1/2 star but universally praising -> Mathematical mismatch 1.0 fraud
        if r <= 2 and pos > 0 and neg == 0:
            return 1.0
            
        return 0.0

    def _get_stylometric_anomaly(self, text):
        import re
        # AI texts often have perfectly uniform sentence lengths and predictable punctuation
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) <= 1:
            variance_score = 0.8 # Anomaly: Real reviews usually have > 1 sentence
        else:
            lengths = [len(s.split()) for s in sentences]
            var = np.var(lengths)
            # Extremely tight variance indicates uniform generated length
            variance_score = 1.0 if var < 5.0 else max(0.0, 1.0 - (var / 50.0))
            
        punct_counts = {p: text.count(p) for p in ['.', ',', '!', '?', ';', '-']}
        total_p = sum(punct_counts.values())
        if total_p == 0:
            entropy = 0.0
        else:
            probs = [c/total_p for c in punct_counts.values() if c > 0]
            entropy = -sum(p * math.log2(p) for p in probs)
            
        # Low punctuation entropy = entirely uniform ending patterns (robotic)
        entropy_score = 1.0 if entropy < 0.5 else max(0.0, 1.0 - entropy)
        
        # Explicit rage-bot anomaly: extremely high density of exclamation marks over other punctuation
        if total_p > 0 and punct_counts.get('!', 0) / total_p > 0.5:
            entropy_score = 1.0
            
        return min((variance_score + entropy_score) / 2.0, 1.0)

    def analyze_authenticity(self, text, rating=None, aspect_sentiments=None):
        """
        Calculates the requested V3 Hybrid 5-Signal FraudScore Formula.
        """
        if not text or str(text).strip() == "":
            return {"fake_probability": 0.0, "fraud_score": 0.0, "classification": "GENUINE"}
            
        ai_prob = self._get_ai_prob(text)
        burst_score = self._get_burstiness(text)
        sim_score = self._get_similarity(text)
        mismatch_score = self._get_mismatch(rating, aspect_sentiments)
        anomaly_score = self._get_stylometric_anomaly(text)
        
        # Weighting requested by System Architect
        fraud_score = (0.35 * ai_prob) + (0.20 * burst_score) + (0.20 * sim_score) + (0.15 * mismatch_score) + (0.10 * anomaly_score)
        
        # Extreme attack mathematical overdrive: If review hits BOTH high spam burstiness & bot signals
        if ai_prob > 0.70 and burst_score > 0.70:
             fraud_score = max(fraud_score, 0.88)
        elif burst_score == 1.0 and anomaly_score == 1.0:
             fraud_score = max(fraud_score, 0.76)
             
        # Strict 55% Fraud Threshold requested by System Architect
        is_fake = fraud_score >= 0.55
        
        # Embed the review natively for the next inference cycle duplication checks
        try:
            emb = self.encoder.encode([text])[0]
            self.history_embeddings.append(emb)
            if len(self.history_embeddings) > 1000:
                self.history_embeddings.pop(0) # Constrain memory mapping bounds
        except:
            pass
            
        return {
            "fake_probability": round(float(ai_prob), 4),
            "fraud_score": round(float(fraud_score), 4),
            "classification": "FAKE_REVIEW" if is_fake else "GENUINE",
            "signals": {
                "AI_Probability": round(float(ai_prob), 4),
                "Burstiness": round(float(burst_score), 4),
                "Similarity": round(float(sim_score), 4),
                "Sentiment_Mismatch": round(float(mismatch_score), 4),
                "Stylometric_Anomaly": round(float(anomaly_score), 4)
            }
        }

if __name__ == "__main__":
    detector = StylometricAuthenticityDetector()
    sample = "I recently had the absolute pleasure of dining at this exquisite establishment. The symphony of flavors was a true testament to their culinary mastery. The meticulous service was paramount. Highly recommend this realm of taste!"
    print(detector.analyze_authenticity(sample))
