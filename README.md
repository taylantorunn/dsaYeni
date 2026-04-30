# Financial Sentiment Analysis: News vs Stock Market

## Overview  
This project analyzes the relationship between news sentiment and stock market movements using statistical analysis and machine learning techniques.

---

## Project Overview  

The goal of this project is to investigate whether the sentiment of daily news headlines can help predict stock market direction. The project combines textual data (news headlines) with financial data (DJIA index).

---

## Datasets  

This project uses two datasets:

- RedditNews.csv  
- upload_DJIA_table.csv  

The datasets are merged using the Date column.

---

## Data Preparation  

- Converted Date columns to datetime format  
- Grouped news by date  
- Combined headlines into a single text column  
- Extracted sentiment scores using TextBlob  
- Created additional features:
  - headline_length  
  - word_count  
- Final dataset saved as: final_dataset.csv  

---

## Exploratory Data Analysis  

- Market movement distribution (Up vs Down)  
- Sentiment distribution  
- Sentiment vs market movement comparison  

---

## Hypothesis Testing  

A t-test was applied to compare sentiment scores between market up and down days.

Result:  
- p-value > 0.05  
- No statistically significant relationship found  

---

## Machine Learning Models  

- Decision Tree  
- Random Forest  

Evaluate performance using:  
- Accuracy  
- ROC-AUC  

---

## Results  

- Accuracy ≈ 0.50  
- ROC-AUC ≈ 0.50–0.52  

---

## Expected Outcome  

It is expected that sentiment may have some influence on market movements, but likely not strong enough to fully explain market behavior.

---

## Contribution  

This project aims to:  

- Combine textual sentiment analysis with financial data  
- Evaluate the predictive power of simple NLP techniques  
- Provide insight into the limitations of sentiment-based market prediction  

---

## Tools and Libraries  

- Python  
- Pandas  
- Scikit-learn  
- TextBlob  
- Matplotlib / Seaborn  

---

## AI Usage Statement  

AI tools such as ChatGPT were used for:  

- Code debugging  
- Structural guidance  

All outputs were reviewed and verified.
