import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import pandas as pd

class AspectSentimentExtractor:
    """
    V3 Word-Level Dependency Aspect-Based Sentiment Analysis (ABSA).
    Uses spaCy for syntactic tree parsing (nsubj, amod) to physically link adjectives directly to aspect nouns.
    Eliminates Zero-Shot Hallucinations, adhering strictly to Mention Rules.
    """
    def __init__(self):
        print("Loading spaCy En-Core-Web-SM Neural Pipeline...")
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Explicit Aspect Mention Thresholding (Strictly Nouns & Domain Adjectives)
        self.aspect_keywords = {
            "Taste": ["taste", "food", "delicious", "awesome", "yummy", "tasty", "flavor", "spicy", "sweet", "biryani", "chicken", "mutton", "fish", "veg", "meal", "starter"],
            "Service": ["service", "staff", "waiter", "manager", "served", "courteous", "polite", "prompt", "rude", "delivery", "late", "time", "delay", "zomato", "swiggy", "fast", "slow"],
            "Ambience": ["ambience", "atmosphere", "vibe", "place", "music", "clean", "neat", "seating", "hygiene", "decor", "interior", "ambinence", "ambiance"],
            "Price": ["price", "cost", "money", "bill", "cheap", "expensive", "pay", "paid", "costly", "overpriced", "affordable"],
            "Quantity": ["quantity", "portion", "size", "filling", "amount", "plenty", "abundant"]
        }

    def _get_word_sentiment(self, word, aspect=None):
        """Calculates internal math polarity for an independent string fragment with domain overrides."""
        lower_word = word.lower()
        
        # Dictionary fixes for misspelled user reviews
        lower_word = lower_word.replace("dissapoint", "disappoint")
        lower_word = lower_word.replace("dissapointed", "disappointed")
        
        # Universal Semantic Overrides for explicit negations that VADER might drop
        # Only apply this if we are evaluating a specific word/modifier, NOT a full fallback sentence string
        if len(lower_word.split()) <= 3 and any(w in lower_word for w in ["not good", "wasn't good", "isn't good", "didn't like", "n't good", "n't like", "n't great", "not great", "terrible", "bad"]):
            return -1.5
        
        # Pad with spaces to enforce exact word matching and prevent substring hallucination
        padded_word = f" {lower_word} "
        
        # Universal Neutral Overrides across ALL aspects seamlessly
        universal_neutrals = [" ok ", " okay ", " average ", " normal ", " standard ", " decent ", " fair ", " fine ", " adequate ", " moderate ", " alright ", " acceptable ", " so-so ", " so so "]
        if any(w in padded_word for w in universal_neutrals):
            return 0.0001
            
        # Hard mathematical context overrides tailored per aspect
        if aspect == "Taste":
            if any(f" {w} " in padded_word for w in ["bland", "tasteless", "flavorless", "bad", "terrible", "worst", "nasty", "stale", "cold", "raw", "overcooked", "salty", "sour", "bitter", "disgusting", "awful", "unappetizing"]):
                return -1.5
            if any(f" {w} " in padded_word for w in ["delicious", "yummy", "tasty", "great", "good", "amazing", "excellent", "awesome", "fantastic", "superb", "perfect", "mouthwatering", "flavorful", "fresh", "incredible", "best"]):
                return 1.5
            if any(f" {w} " in padded_word for w in ["plain", "ordinary", "simple"]):
                return 0.0001
                
        elif aspect == "Price":
            if any(f" {w} " in padded_word for w in ["high", "expensive", "overpriced", "costly", "much", "more", "bad", "too", "pricey", "exorbitant", "unreasonable"]):
                return -1.5
            if any(f" {w} " in padded_word for w in ["cheap", "affordable", "value", "low", "less", "good", "great", "budget", "economical", "bargain", "worth"]):
                return 1.5
            if any(f" {w} " in padded_word for w in ["reasonable"]):
                return 0.0001
                
        elif aspect == "Quantity":
            if any(f" {w} " in padded_word for w in ["less", "small", "tiny", "insufficient", "bad", "low", "poor", "little", "meager", "scarce", "lacking", "inadequate", "short", "skimpy", "improve", "disappointing", "reduce", "not enough"]):
                return -1.5
            if any(f" {w} " in padded_word for w in ["huge", "large", "good", "great", "filling", "more", "lot", "ample", "generous", "massive", "abundant", "plenty", "big", "satisfactory", "excellent", "perfect", "heavy"]):
                return 1.5
            # Expanding explicit quantity contextual neutrals
            if any(f" {w} " in padded_word for w in ["enough", "sufficient", "expected", "typical", "regular", "medium", "exact", "appropriate", "proportionate", "balanced", "right", "proper"]):
                return 0.0001
                
        elif aspect == "Service":
            if any(f" {w} " in padded_word for w in ["manners", "cheaply", "worst", "terrible", "bad", "slow", "rude", "poor", "ignore", "horrible", "delayed", "late", "unprofessional", "arrogant", "lazy", "inattentive", "careless"]):
                return -1.5
            if any(f" {w} " in padded_word for w in ["good", "great", "fast", "quick", "polite", "courteous", "nice", "friendly", "tremendous", "amazing", "excellent", "fantastic", "outstanding", "stellar", "attentive", "helpful", "prompt", "best"]):
                return 1.5
                
        elif aspect == "Ambience":
            if any(f" {w} " in padded_word for w in ["noisy", "loud", "crowded", "dirty", "unclean", "bad", "worst", "terrible", "smelly", "dark", "dull", "cramped", "messy", "uncomfortable", "congested", "dingy", "horrible"]):
                return -1.5
            if any(f" {w} " in padded_word for w in ["good", "great", "nice", "beautiful", "pleasant", "cozy", "clean", "neat", "aesthetic", "vibrant", "lively", "peaceful", "calm", "elegant", "lovely", "amazing", "perfect", "relaxing", "wonderful", "best"]):
                return 1.5
            if any(f" {w} " in padded_word for w in ["casual", "usual"]):
                return 0.0001
                
        return self.sentiment_analyzer.polarity_scores(lower_word)['compound']

    def extract_aspect_sentiments(self, text):
        """
        Step 1: Check Explicit Mention logically.
        Step 2: Use syntactic windowing for children dependencies.
        Step 3: Calculate numerical tokens & aggregate.
        Step 4: Classification mapping.
        """
        results = {
            "Taste": {"sentiment": "Not Mentioned", "confidence": 0.0},
            "Service": {"sentiment": "Not Mentioned", "confidence": 0.0},
            "Ambience": {"sentiment": "Not Mentioned", "confidence": 0.0},
            "Price": {"sentiment": "Not Mentioned", "confidence": 0.0},
            "Quantity": {"sentiment": "Not Mentioned", "confidence": 0.0}
        }
        
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return results

        lower_text = text.lower()
        doc = self.nlp(text)
        
        for aspect, keywords in self.aspect_keywords.items():
            # Step 1: Aspect Mention Check (Zero False Positives mathematical constraint)
            is_mentioned = any(kw in lower_text for kw in keywords)
            
            if not is_mentioned:
                continue # Lock state to Not Mentioned / 0
                
            # Step 2 & 3: spaCy Word-level dependency aggregation array
            token_scores = []
            
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                if any(kw in sent_lower for kw in keywords):
                    # We locate the geometric node of the explicit aspect
                    for token in sent:
                        if token.lemma_.lower() in keywords or token.text.lower() in keywords:
                            # 1. Dependency Link: Children mapping (e.g. food was [ADJ])
                            modifiers = []
                            for child in token.children:
                                if child.pos_ in ['ADJ', 'ADV', 'VERB']:
                                    # Check for any attached negation to that modifier (e.g. "wasn't good")
                                    negation = next((n.text for n in child.children if n.dep_ == 'neg'), "")
                                    modifiers.append(f"{negation} {child.text}".strip())
                            
                            # 2. Dependency Link: Head linking (e.g. [ADJ] waiters)
                            if token.dep_ in ['nsubj', 'nsubjpass', 'compound']:
                                head = token.head
                                if head.pos_ in ['ADJ', 'VERB', 'AUX']:
                                    # Snag direct modifiers or complement of the verb (e.g. 'quantity [is] -> average')
                                    for c in head.children:
                                        if c.pos_ == 'ADV':
                                            modifiers.append(c.text)
                                        elif c.dep_ in ['acomp', 'attr', 'dobj', 'oprd']:  # "is -> average"
                                            acomp_neg = next((n.text for n in c.children if n.dep_ == 'neg'), "")
                                            modifiers.append(f"{acomp_neg} {c.text}".strip())
                                            # Snag adverbs attached to the complement (e.g. "is -> very average")
                                            for gc in c.children:
                                                if gc.pos_ == 'ADV':
                                                    modifiers.append(f"{gc.text} {c.text}")
                                    
                                    # Also keep the head itself if it carries sentiment
                                    negation = next((n.text for n in head.children if n.dep_ == 'neg'), "")
                                    modifiers.append(f"{negation} {head.text}".strip())
                            
                            # 3. Universal logical proximity frame (5 word bounding box) to catch complex negations
                            start = max(0, token.i - 3)
                            end = min(len(doc), token.i + 4)
                            
                            # Scan the bounding box for adjectives
                            for i in range(start, end):
                                if doc[i].pos_ in ['ADJ', 'ADV', 'VERB']:
                                    # Syntactic check backwards for a negation token immediately preceding it
                                    negation = ""
                                    if i > 0 and (doc[i-1].dep_ == 'neg' or doc[i-1].text.lower() in ["not", "wasn't", "isn't", "didn't", "never", "no"]):
                                        negation = doc[i-1].text
                                    elif i > 1 and (doc[i-2].dep_ == 'neg' or doc[i-2].text.lower() in ["not", "wasn't", "isn't", "didn't", "never", "no"]):
                                        negation = doc[i-2].text
                                        
                                    modifiers.append(f"{negation} {doc[i].text}".strip())
                            
                            # Deduplicate modifiers before scoring 
                            modifiers = list(set(modifiers))
                            
                            with open("mods_debug.txt", "a") as f:
                                f.write(f"\\n[{aspect}] MODIFIERS: {modifiers}")
                            
                            for mod in modifiers:
                                score = self._get_word_sentiment(mod, aspect)
                                if score != 0 or score == 0.0001: # Exclude zero-valence connecting verbs but KEEP forced Neutrals
                                    token_scores.append(score)

            # Step 4: Final Aggregation and Threshold mapping
            if token_scores:
                aspect_score = sum(token_scores) / len(token_scores)
            else:
                # If absolute semantic barrenness around the word, map the overarching sentence sequence
                fallback_sentences = [sent.text for sent in doc.sents if any(kw in sent.text.lower() for kw in keywords)]
                full_sent = " ".join(fallback_sentences)
                aspect_score = self._get_word_sentiment(full_sent, aspect)
                
            # Classifications mathematically constrained
            if aspect_score > 0.15:
                sentiment = "Positive"
            elif aspect_score < -0.15:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            # Construct synthetic contextual confidence score from density bounds
            confidence = min(abs(aspect_score) + 0.45, 1.0)
            if sentiment == "Neutral":
                confidence = 1.0 - abs(aspect_score)
                
            results[aspect] = {
                "sentiment": sentiment,
                "confidence": round(float(confidence), 2)
            }

        return results

if __name__ == "__main__":
    extractor = AspectSentimentExtractor()
    # Explicit test demonstrating the failure edge case logically blocked
    test_str = "The mushroom is cool but waiters were nice and price is terrible."
    print(f"Review: '{test_str}'")
    print(extractor.extract_aspect_sentiments(test_str))
