# Financial Sentiment Analysis: News vs Stock Market

This project analyzes the relationship between news sentiment and stock market movements using machine learning and statistical analysis techniques.

## 📌 Project Overview

The goal of this project is to investigate whether the sentiment of daily news headlines can help predict stock market direction. The project combines **text data (news headlines)** with **financial data (DJIA index)** to perform data enrichment.

## 📊 Dataset

The dataset used is:

**Daily News for Stock Market Prediction (Kaggle)**  
https://www.kaggle.com/datasets/aaron7sun/stocknews

It includes:
- News headlines (Top 25 per day)
- Stock market labels (0 = down, 1 = up)
- Date range: 2008 – 2016

## ⚙️ Methods Used

- Data Cleaning & Preprocessing
- Sentiment Analysis (TextBlob)
- Exploratory Data Analysis (EDA)
- Hypothesis Testing (Correlation)
- Machine Learning Models:
  - Decision Tree
  - Random Forest

## 📈 Results

- Correlation between sentiment and market movement: **very weak**
- p-value > 0.05 → No statistically significant relationship
- Model Performance:
  - AUC Score ≈ **0.53**

## 🧠 Conclusion

The results suggest that news sentiment alone is not a strong predictor of stock market movement. Financial markets are complex systems influenced by many factors beyond daily news headlines.

## 🚀 How to Run

1. Download dataset from Kaggle
2. Upload `Combined_News_DJIA.csv` to Colab
3. Run the notebook step-by-step

## 👤 Author

Taylan Torun  
DSA210 Project
