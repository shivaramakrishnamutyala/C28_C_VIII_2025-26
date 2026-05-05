import pandas as pd
from gliner import GLiNER
from thefuzz import process, fuzz
import os
import warnings
import re

warnings.filterwarnings('ignore', category=UserWarning)

class DynamicDishExtractor:
    """
    V2 High-Intelligence Dish Extractor using GLiNER (Generalist Model for NER).
    Zero-shot extraction directly from deep semantic representations.
    Eliminates rigid grammatical dependency hallucination (e.g. 'good taste' is ignored natively).
    """
    def __init__(self, model_name='urchade/gliner_base'):
        # Load SOTA zero-shot NER model
        print(f"Loading GLiNER zero-shot model '{model_name}'...")
        try:
            self.model = GLiNER.from_pretrained(model_name)
        except Exception as e:
            print(f"Fallback to small base model due to: {e}")
            self.model = GLiNER.from_pretrained("urchade/gliner_small")
            
        # The specific entity categories we want GLiNER to conceptually identify
        # We explicitly scan for 'person' to aggressively block server/staff names from being marked as dishes
        self.labels = ["food", "dish", "beverage", "drink", "person", "name"]
        
        # Standardize misspelled incoming dishes securely to correct database groups
        self.standard_menu_items = [
            'biryani', 'pizza', 'pasta', 'burger', 'paneer', 'chicken', 
            'mutton', 'naan', 'roti', 'fried rice', 'noodles', 'dosa', 
            'idli', 'waffle', 'ice cream', 'cake', 'curry', 'fish',
            'bisibelebath', 'upma', 'poori', 'vada', 'samosa', 'momos', 'ramen', 'vadapav', 'vada pav',
            'crispy corn', 'corn', 'tandoori chicken', 'chicken biryani', 'basket biryani', 'mutton biryani', 'veg biryani', 'egg biryani', 'prawn biryani'
        ]

    def extract_dishes(self, text):
        """
        Dynamically extracts food entities using Zero-Shot NLP understanding with Contextual Preservation.
        """
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return []

        # NLP Pre-processing: Symmetrical Ellipsis Resolution (Compound Dish Extraction)
        # e.g., "chicken and mutton biryani" -> natively rewritten to "chicken biryani and mutton biryani"
        text = re.sub(
            r'\b(chicken|mutton|veg|egg|prawn|fish|paneer)\s+(and|&)\s+(one\s+|two\s+)?(chicken|mutton|veg|egg|prawn|fish|paneer)\s+(biryani|curry|rice|noodles|pizza|burger)\b',
            r'\1 \5 \2 \3\4 \5', text, flags=re.IGNORECASE
        )

        # Predict entities natively conceptually
        entities = self.model.predict_entities(text, self.labels, threshold=0.45)
        
        extracted = set()
        for entity in entities:
            # High-intelligence strict exclusion: If GLiNER identifies a human/server name, discard immediately
            if entity["label"] in ["person", "name"]:
                continue
                
            clean_text = entity["text"].lower().strip()
            # Hard block on descriptive hallucination & generic restaurant/staff taxonomy
            generic_blocks = [
                "taste", "flavor", "flavors", "good taste", "quality", "quantity", "money", "price", 
                "experience", "gastronomic experience", "paradise", "food", "foods", "dish", "dishes", "item", "items", 
                "beverage", "beverages", "drink", "drinks", "buffet", "starters", "starter", 
                "main course", "dessert", "desserts", "meal", "meals", "dinner", "lunch", "breakfast", "menu",
                "service", "servive", "ambience", "atmosphere", "place", "restaurant", "nizamis", "nizami", "chinese", "italian",
                "music", "song", "songs", "dj", "band", "bite", "bites",
                "zomato gold", "zomatogold", "swiggy dineout", "dineout", "online", "cuisine", "cuisines",
                "staff", "manager", "server", "waiter", "captain", "chef", "owner", "soumen", "das", "soumen das",
                "culinary traditions", "sustenance", "culinary masterpieces", "tapestry", "symphony",
                "culinary", "masterpieces", "traditions", "mastery", "palate", "venue", "establishment"
            ]
            
            # Reject exact standalone generic words, OR phrases that are just describing generic things
            is_invalid_phrase = (
                clean_text in generic_blocks or 
                clean_text.endswith(" food") or 
                clean_text.endswith(" foods") or 
                "culinary" in clean_text or 
                "experience" in clean_text or 
                "tapestry" in clean_text or 
                "symphony" in clean_text or 
                "sustenance" in clean_text
            )
            
            if not is_invalid_phrase:
                # Suffix Stripping for Delivery and Generic Taxonomy (e.g. "burmese parcel" -> "burmese")
                for delivery_term in [" parcel", " takeaway", " delivery", " dishes", " dish", " items", " item"]:
                    if clean_text.endswith(delivery_term):
                        clean_text = clean_text[:-len(delivery_term)].strip()
                        break
                        
                if clean_text: # Ensure it didn't collapse entirely
                    extracted.add(clean_text)
                
        # --- Pipeline Step 1: High-Intelligence Fallback & Typo Consolidation ---
        extracted = self._fallback_ngram_extraction(text, extracted)
        extracted_list = list(self._standardize_dish_names(extracted))
                
        # --- Pipeline Step 2: Context-Aware Sub-String Preservation ---
        # Solves the "Tawa Fish vs Fish" distinction limitation.
        # Everything here is operating on the cleanly standardized names (from the DB).
        # We check the raw user text to see if the sub-string occurs independently of the parent string.
        
        lower_text = text.lower()
        final_contextual_unique = []
        
        # Sort by length descending, so longer phrases like 'chicken biryani' process first
        extracted_list_sorted = sorted(extracted_list, key=len, reverse=True)
        
        for dish in extracted_list_sorted:
            dish_lower = dish.lower().strip()
            is_redundant_substring = False
            
            for parent_dish in final_contextual_unique:
                parent_lower = parent_dish.lower().strip()
                # If the current dish (e.g. 'chicken') is entirely contained within an already-approved 
                # longer dish (e.g. 'chilli chicken' or 'chicken biryani')
                if dish_lower in parent_lower:
                    is_redundant_substring = True
                    break
                        
            if not is_redundant_substring and dish_lower != "":
                final_contextual_unique.append(dish)

        final_consumed_dishes = []
        for dish in final_contextual_unique:
            if self._is_genuinely_consumed(dish, lower_text):
                final_consumed_dishes.append(dish)

        return final_consumed_dishes

    def _is_genuinely_consumed(self, dish, text):
        """
        Scans the immediate text-window exactly prior to a dish mention to ensure it was 
        genuinely consumed/recommended and not just used as a theoretical negative comparison 
        (e.g., "people who hate mutton can try this" or "without vegetables").
        """
        # Words indicating the dish was a theoretical comparison OR explicitly absent
        theoretical_rejectors = [
            'hate', 'unlike', 'instead of', 'avoid', 'skip', 'not prefer', 'don\'t like', 'do not like', 'dislike',
            'without', 'missing', 'no', 'lack', 'lacking', 'zero',
            'restaurant for', 'place for', 'famous for', 'known for', 'great for', 'good for', 'best for', 'hub for', 'spot for'
        ]
        
        escaped_dish = re.escape(dish)
        for match in re.finditer(escaped_dish, text):
            # Grab a 35-character syntactic window before the dish mention.
            start_window = max(0, match.start() - 35)
            context_window = text[start_window:match.start()].strip()
            
            # If any of the rejection phrases appear right before the dish, we assume it wasn't consumed
            for rejector in theoretical_rejectors:
                # Use regex to ensure we match the rejector phrase
                if re.search(r'\b' + re.escape(rejector) + r'\b', context_window, re.IGNORECASE):
                    return False
        return True

    def _fallback_ngram_extraction(self, text, extracted_so_far):
        """
        Creates 1-gram to 3-gram slices and forces a Levenshtein match.
        """
        lower_text = text.lower()
        # Split by punctuation to avoid ngram hallucination across boundaries
        phrases = re.split(r'[.,;!]+', lower_text)
        
        for phrase in phrases:
            words = phrase.strip().split()
            if not words:
                continue
                
            ngrams = words.copy()
            for i in range(len(words) - 1):
                ngrams.append(f"{words[i]} {words[i+1]}")
            for i in range(len(words) - 2):
                ngrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                
            for gram in ngrams:
                clean_gram = ''.join(e for e in gram if e.isalnum() or e.isspace()).strip()
                if len(clean_gram) > 3:
                    # Use token_sort_ratio for n-grams to handle multi-word phrase matching perfectly
                    match = process.extractOne(clean_gram, self.standard_menu_items, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 85:
                        extracted_so_far.add(match[0])
                        
        return extracted_so_far

    def _standardize_dish_names(self, extracted_dishes):
        """
        Uses Levenshtein distance (Fuzzy Matching) to securely consolidate 
        user typos (e.g., 'biriyani', 'pizaa') to the exact master DB group.
        """
        standardized = set()
        for dish in extracted_dishes:
            match = process.extractOne(dish, self.standard_menu_items, scorer=fuzz.ratio)
            
            # Require a high 85% confidence score to overwrite a dynamic user string
            if match and match[1] >= 85:
                standardized.add(match[0])  # Use standardized spelling
            else:
                standardized.add(dish)      # Unrecognized or specific sub-dish (like 'basket biryani'), keep original string
        return standardized

    def process_dataframe(self, df, text_column='review_text'):
        """
        Applies the GLiNER extraction to the entire pandas DataFrame.
        """
        print(f"GLiNER zero-shot extracting live dish names from {len(df)} reviews...")
        df['Mentioned_Dishes'] = df[text_column].apply(self.extract_dishes)
        
        # Count how many total unique dishes we found
        all_dishes = [dish for sublist in df['Mentioned_Dishes'].tolist() for dish in sublist]
        print(f"GLiNER discovered {len(set(all_dishes))} unique live dishes across the dataset!")
        
        return df

if __name__ == "__main__":
    extractor = DynamicDishExtractor()
    test_reviews = [
        "We ordered the blueberry cheesecake waffle and it was amazing.",
        "The garlic naan was extremely cold but I tried the spicy mutton biryani anyway.",
        "I highly recommend trying their signature Apollo Fish, it's the best.",
        "Paneer curry is good taste.", # The exact string that hallucinated before
        "BisiBeleBath terrible", # Zero-context edge case
        "pizaa completely burnt" # Extreme typo + zero context
    ]
    
    print("\nTesting V2 GLiNER NER Output:\n" + "-"*30)
    for review in test_reviews:
        print(f"Review: {review}")
        print(f"Dishes: {extractor.extract_dishes(review)}\n")
