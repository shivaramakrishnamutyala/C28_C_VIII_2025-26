# Surveil-Eat
# 🍽️ SurveilEat AI

### Restaurant Review Intelligence System

---

## 📌 Overview

**SurveilEat AI** is an advanced Natural Language Processing (NLP) system designed to analyze restaurant reviews and transform unstructured text into **structured, reliable, and actionable insights**.

The system integrates multiple AI techniques such as **aspect-based sentiment analysis, zero-shot dish extraction, and fraud detection** to provide a **trustworthy and manipulation-resistant restaurant rating system**.

---

## 🎯 Objectives

* Extract fine-grained sentiment across multiple restaurant aspects
* Dynamically identify dish names using zero-shot NER
* Detect fake or suspicious reviews using multi-signal analysis
* Generate a reliable and fair restaurant rating
* Provide interactive visualization for decision-making

---

## 🧠 Key Features

### ✅ Aspect-Based Sentiment Analysis

* Uses **spaCy dependency parsing + VADER**
* Analyzes:

  * Taste
  * Service
  * Ambience
  * Price
  * Quantity

---

### 🍛 Zero-Shot Dish Extraction

* Powered by **GLiNER (Transformer-based NER)**
* Extracts dish names dynamically
* Handles spelling variations and unseen dishes

---

### 🚨 Hybrid Fraud Detection Engine

* Multi-signal model using:

  * RoBERTa (AI detection)
  * Semantic similarity
  * Sentiment-rating mismatch
  * Stylometric analysis

* Fraud Score:

```
FraudScore = 0.35*S1 + 0.20*S2 + 0.20*S3 + 0.15*S4 + 0.10*S5
```

---

### ⭐ Dynamic Rating Engine

* Combines:

  * Sentiment scores
  * Fraud filtering
  * Time decay

* Produces a **fair and reliable rating**

---

### 📊 Interactive Dashboard

* Built using **Streamlit**
* Displays:

  * Aspect-wise sentiment
  * Dish insights
  * Fraud analysis
  * Final rating

---

## 🏗️ System Architecture

```
Input Reviews
     ↓
Semantic Preprocessing
     ↓
Aspect Sentiment Analysis + Dish Extraction
     ↓
Fraud Detection Engine
     ↓
Dynamic Rating Engine
     ↓
SQLite Database
     ↓
Streamlit Dashboard
```

---

## 🛠️ Tech Stack

| Category        | Tools                |
| --------------- | -------------------- |
| Programming     | Python               |
| NLP             | spaCy, VADER         |
| Transformers    | GLiNER, RoBERTa      |
| Embeddings      | SentenceTransformers |
| Database        | SQLite               |
| Visualization   | Streamlit            |
| Data Processing | Pandas, NumPy        |

---

## 📂 Project Structure

```
SurveilEat-AI/
│
├── data/
├── models/
├── src/
├── app.py
├── requirements.txt
├── surveileat_intelligence.db
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/SurveilEat-AI.git
cd SurveilEat-AI
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Application

```
streamlit run app.py
```

---

## 📊 Dataset

* Source: Kaggle Restaurant Review Dataset
* Contains:

  * Review text
  * Ratings
  * Timestamp

---

## 📈 Results

* High accuracy in fraud detection
* Fine-grained sentiment insights
* Reliable rating resistant to manipulation

---

## 🔮 Future Scope

* Multilingual support
* Real-time API integration
* Large-scale deployment
* Enhanced fraud detection models

---

## 👨‍💻 Author

**[shivaramakrishna]**
B.Tech Information Technology

---

## 📜 License

This project is for academic and educational purposes.

---
