# Financial Sentiment Analysis: News vs Stock Market

## Overview
This project investigates the relationship between daily news sentiment and stock market movements.
The goal is to evaluate whether simple Natural Language Processing (NLP) techniques can help predict market direction.

---

## Datasets

### 1. News Dataset
Source: Kaggle  
Link: https://www.kaggle.com/datasets/aaron7sun/stocknews  
Description: Daily top news headlines collected from Reddit WorldNews.

### 2. Financial Market Data
Source: Kaggle (same dataset)  
Data: Dow Jones Industrial Average (DJIA)

### 3. Additional Dataset (Enrichment)
Source: Yahoo Finance (via yfinance)  
Data: S&P 500 Index (SP500)

---

## Project Structure

- data_preparation.ipynb → Data cleaning & feature engineering
- eda.ipynb → Exploratory data analysis
- hypothesis_testing.ipynb → Statistical testing (t-test)
- model_training.ipynb → Machine learning models

---

## Methodology

### Feature Engineering
- Sentiment score (TextBlob)
- Headline length
- Word count
- SP500 index

### Target Variable
- 1 → Market goes up next day
- 0 → Market goes down next day

---

## Hypothesis Testing

A t-test was conducted to compare sentiment between upward and downward market days.

Result:
p-value ≈ 0.32

Interpretation:
There is no statistically significant difference between the two groups.

---

## Model Results

Models used:
- Decision Tree
- Random Forest

Performance:
- Accuracy ≈ 0.47 – 0.50
- ROC-AUC ≈ 0.46 – 0.50

Interpretation:
Model performance is close to random guessing, indicating weak predictive power.

---

## Feature Importance

Most important features:
1. SP500
2. Sentiment
3. Headline length
4. Word count

Insight:
SP500 and sentiment have similar importance, but neither provides strong predictive power alone.

---

## Conclusion

- News sentiment alone is not sufficient to predict market movements
- Financial markets are influenced by many complex factors
- Simple NLP methods are limited in predictive capability

---

## Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- TextBlob
- Matplotlib / Seaborn
- SciPy
- yfinance

---

## AI Usage Statement

AI tools such as ChatGPT were used for:

- Debugging code
- Structuring notebooks
- Writing documentation

Example prompts:
- Fix pandas merge error
- Write README for sentiment analysis project

All outputs were reviewed and verified before use.

---

## Requirements

See requirements.txt for dependencies.
