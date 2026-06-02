# Fake Review Detection

## Overview
A machine learning project comparing traditional statistical models (Naive Bayes, Logistic Regression) vs. embedding-based approaches (TF-IDF) to detect fake computer-generated reviews from authentic human-written ones.

## Dataset
- **Source**: Maxwell's Fake Reviews Dataset (Kaggle)
- **Size**: 40,000+ reviews
- **Classes**: CG (Computer-Generated) vs OR (Original/Authentic)

## Models & Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **TF-IDF + Logistic Regression** | **87%** | **0.87** ✓ |
| Naive Bayes (Bag-of-Words) | 85% | 0.85 |
| Logistic Regression (BoW) | 81% | 0.81 |

**Winner**: TF-IDF + Logistic Regression — better at capturing subtle patterns in review text.

## Key Preprocessing
1. Text cleaning (lowercase, remove punctuation/URLs)
2. Tokenization
3. Stopword removal
4. Lemmatization

## Quick Start
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Vectorize & train
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(reviews_train)

model = LogisticRegression()
model.fit(X_train, labels_train)
```

## Main Finding
**Term weighting (TF-IDF) outperforms raw word frequency** — rare but informative words matter more than common ones for detecting fake reviews.

## Requirements
```
pandas, numpy, scikit-learn, nltk, matplotlib, seaborn
```
