# Project Proposal: Financial Sentiment Analysis

## Objective
The objective of this project is to analyze whether the sentiment of daily news headlines has a measurable impact on stock market movements.

## Research Question
Can sentiment extracted from news headlines be used to predict whether the stock market will go up or down?

## Data Sources
The project uses publicly available datasets:
- Reddit news headlines dataset
- DJIA stock market dataset

These datasets will be merged based on date to create a unified dataset.

## Methodology

### 1. Data Preparation
- Clean and merge datasets
- Combine multiple news headlines into a single text column
- Extract sentiment scores using TextBlob
- Create additional features (word count, text length)

### 2. Exploratory Data Analysis
- Analyze distribution of sentiment values
- Compare sentiment across market movements
- Visualize patterns using plots

### 3. Hypothesis Testing
- Perform a t-test to compare sentiment between:
  - Market up days
  - Market down days

### 4. Machine Learning
- Train classification models:
  - Decision Tree
  - Random Forest
- Evaluate performance using:
  - Accuracy
  - ROC-AUC

## Expected Outcome
It is expected that sentiment may have some influence on market movements, but likely not strong enough to fully explain market behavior.

## Contribution
This project aims to:
- Combine textual sentiment analysis with financial data
- Evaluate the predictive power of simple NLP techniques
- Provide insight into the limitations of sentiment-based market prediction

## Tools and Libraries
- Python
- Pandas
- Scikit-learn
- TextBlob
- Matplotlib / Seaborn

## AI Usage Statement
AI tools such as ChatGPT may be used for:
- Code debugging
- Structural guidance

All outputs will be reviewed and verified.
