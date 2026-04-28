
# Financial Sentiment Analysis Using News Headlines and Stock Market Data

This project focuses on analyzing financial news headlines and examining their relationship with stock market movements. Instead of relying solely on numerical financial data, this study incorporates textual data to better understand how public sentiment may influence market behavior.

The dataset used in this project is the “Daily News for Stock Market Prediction” dataset obtained from Kaggle. It combines Reddit news headlines from the WorldNews channel with Dow Jones Industrial Average (DJIA) stock data. The dataset includes daily news headlines ranked by popularity, along with a binary label indicating whether the market increased or decreased on that day.

The goal of this project is to determine whether the sentiment of daily news headlines has any predictive power over stock market direction. To achieve this, Natural Language Processing (NLP) techniques are applied to convert text data into sentiment scores. These scores are then analyzed in relation to market movements.

The project follows a structured data science workflow. First, the data is loaded and preprocessed by combining multiple news headlines into a single textual feature for each day. Then, exploratory data analysis (EDA) is conducted to understand the distribution and trends within the dataset. After that, sentiment analysis is performed using the TextBlob library to extract polarity scores from the news text.

In the next step, a statistical hypothesis test is conducted to examine whether there is a significant relationship between sentiment scores and market direction. Finally, a machine learning model (Decision Tree Classifier) is trained to predict whether the market will go up or down based on sentiment features. The model is evaluated using the ROC-AUC metric.

This project demonstrates how combining textual data with financial data can provide deeper insights into market behavior and highlights the importance of data enrichment in modern data science applications.

This project is developed as part of the DSA210 course at Sabancı University.
