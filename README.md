# Kindle Store Review Sentiment Analysis

## Overview
This project analyzes customer reviews from the **Amazon Kindle Store category** dataset. The dataset consists of product reviews collected from **May 1996 to July 2014**, containing a total of **982,619 reviews**. The primary focus of this analysis is **sentiment classification** and understanding the factors influencing review helpfulness using **Bag of Words (BoW)** and **TF-IDF (Term Frequency-Inverse Document Frequency)** techniques.

## Dataset Information
The dataset contains the following columns:
- `asin`: Product ID (e.g., B000FA64PK)
- `helpful`: Helpfulness rating of the review (e.g., 2/3)
- `overall`: Rating of the product (1-5 stars)
- `reviewText`: Full text of the review
- `reviewTime`: Time of the review
- `reviewerID`: Unique ID of the reviewer
- `reviewerName`: Name of the reviewer
- `summary`: Short summary of the review
- `unixReviewTime`: Unix timestamp of the review

## Objectives
- **Sentiment Analysis**: Classify reviews as **positive, negative, or neutral** based on their textual content.
- **Feature Engineering**: Extract features using **BoW and TF-IDF** for text representation.
- **Review Helpfulness Prediction**: Identify factors influencing whether a review is marked as helpful.

## Techniques Used
### 1. **Data Preprocessing**
- Removed **punctuation, stopwords, and special characters**.
- Converted text to **lowercase**.
- Performed **tokenization and lemmatization**.

### 2. **Feature Extraction**
#### **Bag of Words (BoW)**
- Created a document-term matrix representing word occurrences.
- Used **CountVectorizer** from `sklearn.feature_extraction.text`.
- Applied feature selection to remove low-frequency words.

#### **TF-IDF (Term Frequency-Inverse Document Frequency)**
- Used `TfidfVectorizer` to transform text into numerical vectors.
- Adjusted `max_df` and `min_df` to filter out too common/rare words.
- Compared **unigrams and bigrams** for improved representation.

### 3. **Sentiment Classification**
- Mapped `overall` ratings:
  - **1-2 stars** → Negative
  - **3 stars** → Neutral
  - **4-5 stars** → Positive
- Trained classifiers such as **Logistic Regression, Naïve Bayes, and SVM**.
- Evaluated performance using **accuracy, precision, recall, and F1-score**.

### 4. **Review Helpfulness Analysis**
- Defined `helpful_score` as the ratio of helpful votes to total votes.
- Explored the correlation between **review length, sentiment, and helpfulness**.
- Used **Regression Models** to predict helpfulness scores.

## Results & Insights
- **TF-IDF performed better than BoW** for sentiment classification due to its ability to reduce the weight of common words.
- **Positive reviews were more likely to be marked helpful** compared to negative reviews.
- **Longer and detailed reviews** had higher helpfulness scores.
- **Logistic Regression with TF-IDF achieved the best accuracy (~85%)**.

## Future Work
- Apply **Deep Learning models (LSTMs, BERT)** for improved classification.
- Experiment with **word embeddings (Word2Vec, FastText)**.
- Detect **fake/spam reviews** using text similarity techniques.

## References
- [Amazon Product Data - Julian McAuley, UCSD](http://jmcauley.ucsd.edu/data/amazon/)
- Scikit-learn documentation for NLP preprocessing and classification.

## Author
**Bhavya Jaggi**

