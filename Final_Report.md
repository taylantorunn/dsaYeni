# DSA 210 Final Report: Financial Sentiment Analysis

**Student:** Taylan Torun  
**Course:** DSA 210 Introduction to Data Science  
**Term:** 2025-2026 Spring Term  
**GitHub Repository:** [https://github.com/taylantorunn/dsaYeni](https://github.com/taylantorunn/dsaYeni)

---

## 1. Motivation
The stock market is a highly dynamic environment driven by various factors, including public perception and global news. Often, investors react to daily headlines, causing fluctuations in market indices. My motivation for this project is to investigate the exact relationship between daily news sentiment and stock market movements. By combining Natural Language Processing (NLP) techniques with financial time-series data, I aim to evaluate whether analyzing the sentiment of top news headlines can provide meaningful insights or predictive power regarding the market's direction for the following day. This project bridges the gap between text analysis and real-world financial decision-making, testing the "Efficient-market hypothesis" which states that asset prices reflect all available information.

## 2. Data source
The data used in this project is collected from multiple publicly available sources and APIs spanning an 8-year period from **2008 to 2016**.

1. **News Dataset (`RedditNews.csv`):** Obtained from Kaggle (Aaron7sun's StockNews dataset). It contains daily top 25 news headlines collected directly from the Reddit WorldNews channel.
2. **Financial Market Data (`upload_DJIA_table.csv`):** Also obtained from the same Kaggle dataset, which tracks the historical daily values of the Dow Jones Industrial Average (DJIA), including Open, High, Low, Close, and Volume.
3. **Additional Dataset (Enrichment via API):** To enrich the analysis and provide a broader macroeconomic context, I utilized the `yfinance` library in Python to pull the S&P 500 Index (`^GSPC`) daily close prices.

```python
# Pulling Enrichment Data from Yahoo Finance
import yfinance as yf
sp500 = yf.download("^GSPC", start="2008-06-01", end="2016-07-01")
sp500.reset_index(inplace=True)
sp500 = sp500[["Date", "Close"]]
sp500.rename(columns={"Close": "SP500"}, inplace=True)
```
**Code Explanation & Data Source Insight:** 
- `yf.download()`: Connects to Yahoo Finance to fetch the S&P 500 (`^GSPC`) data precisely matching our Kaggle dataset's timeline (2008-2016).
- `reset_index()`: Converts the date index back into a standard column so it can be merged with the other datasets later.
- `rename()`: Changes the generic "Close" column name to "SP500" to prevent confusion with the DJIA's "Close" column.
- *Insight:* The S&P 500 data serves as a broader indicator of the U.S. economy compared to the DJIA, which only tracks 30 prominent companies. Including this enriches our feature set for the machine learning models.

## 3. Data analysis

### 3.1 Data Preprocessing & Feature Engineering
Data in the real world is rarely ready for modeling. The first stage involved grouping all daily headlines into a single paragraph per day, and then merging our three datasets (News, DJIA, SP500) into one cohesive dataframe based on the `Date` column.

```python
from textblob import TextBlob

# Feature Engineering
df["sentiment"] = df["News"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["headline_length"] = df["News"].apply(len)
df["word_count"] = df["News"].apply(lambda x: len(x.split()))

# Defining the Target Variable (Binary Classification)
df["Label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()
```
**Code Explanation & Feature Insight:** 
- `TextBlob(x).sentiment.polarity`: Processes the raw text of the combined daily headlines and returns a numerical score between -1 (highly negative) and 1 (highly positive). This acts as our primary NLP feature.
- `apply(len)` and `len(x.split())`: Extracts the total number of characters and words respectively. This helps test if the *volume* of news (regardless of sentiment) affects the market.
- `shift(-1)`: This is the most crucial step for forecasting. It shifts the DJIA close price backward by one day. We then compare it to today's close `> df["Close"]`. If tomorrow's price is higher, it returns `True` (converted to `1` by `astype(int)`). This sets up a true predictive forecasting problem rather than just analyzing the same day.

### 3.2 Exploratory Data Analysis (EDA)
To understand the underlying patterns in the data, I conducted Exploratory Data Analysis using `seaborn`, `matplotlib`, and `wordcloud`.

```python
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Distribution of Target Variable
sns.countplot(x=df["Label"])
plt.show()

# 2. Correlation Heatmap
numerical_cols = df[['sentiment', 'headline_length', 'word_count', 'SP500', 'Label']]
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# 3. Wordcloud for Positive Market Days
positive_news = " ".join(text for text in df[df['Label'] == 1]['News'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_news)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.show()
```
**Code Explanation & EDA Interpretations:** 
1. **Countplot:** Calculates and visualizes the total count of `0`s and `1`s. It verified that our classes (Market Up vs Market Down) are relatively balanced.
2. **Correlation Heatmap:** The `heatmap` visually plots the linear correlation coefficients between variables. *Interpretation:* It confirmed that individual features like `sentiment` or `SP500` have very low linear correlation (close to 0.00) with the target variable, proving that stock market movements are highly non-linear and cannot be predicted by a single variable alone.
3. **Word Cloud:** Highlights the most dominant words in the news before positive market days. While common political terms appeared frequently, it showed that actionable financial signals require more sophisticated NLP techniques than basic word frequencies.

### 3.3 Hypothesis Testing
To scientifically validate our visual findings, I performed independent two-sample t-tests.

```python
from scipy.stats import ttest_ind

# T-Test 1: Sentiment
up = df[df["Label"] == 1]["sentiment"]
down = df[df["Label"] == 0]["sentiment"]
t_stat, p_value = ttest_ind(up, down)

# T-Test 2: Headline Length (News Volume)
up_len = df[df["Label"] == 1]["headline_length"]
down_len = df[df["Label"] == 0]["headline_length"]
t_stat_len, p_value_len = ttest_ind(up_len, down_len)
```
**Code Explanation & Statistical Interpretation:** 
- `ttest_ind()` computes the T-test for the means of two independent samples.
- **Sentiment Test:** The resulting p-value was **0.3282**. Because $0.3282 > 0.05$, we fail to reject the null hypothesis. This proves that mathematically, there is no significant difference in the daily news sentiment between positive and negative market days.
- **Headline Length Test:** The secondary t-test on the volume of news also yielded a high p-value, indicating that the sheer amount of news output is not a statistically significant predictor of the market's direction.

### 3.4 Machine Learning Models
To improve upon the baseline Random Forest models (which achieved ~50% accuracy), I built an advanced NLP pipeline to bypass simple sentiment scoring.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_text = df["News"]
y = df["Label"]
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Convert Text to Numerical TF-IDF Matrix
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Train Logistic Regression Model
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_tfidf, y_train)
y_pred_log = log_model.predict(X_test_tfidf)
```
**Code Explanation:** 
- `train_test_split(test_size=0.2)`: Reserves 80% of the historical data for training, holding out 20% to test predictive accuracy on unseen data.
- `TfidfVectorizer()`: Instead of reducing the daily news into a single arbitrary sentiment float, this extracts a sparse matrix of 5,000 specific keywords from the dataset, weighing words by their frequency and rarity (Term Frequency-Inverse Document Frequency).
- `LogisticRegression()`: A linear model trained on this high-dimensional text data to predict the binary target.

## 4. Findings
The machine learning phase yielded the most crucial findings regarding market predictability.

- **Baseline Models (Decision Tree & Random Forest):** 
  - Accuracy: **~50%** (Performed no better than random guessing)
- **Advanced NLP Model (TF-IDF + Logistic Regression):** 
  - Accuracy: **~55%** 
  - ROC-AUC: **~0.53**

**Interpretation of Metrics:**
By upgrading to TF-IDF, the model proved that the presence of *specific keywords* holds more predictive power than basic lexicon-based sentiment analysis (`TextBlob`). The accuracy jumped from 50% to roughly 55%. 

**Feature Importance Analysis:**
However, despite this improvement, predicting the stock market using only daily news headlines remains incredibly difficult. The Random Forest feature importances from earlier baselines confirmed this by showing an almost equal weight split between S&P 500 (0.267), Sentiment (0.266), and text volume (0.250). 

**Overall Conclusion:** Daily news sentiment and keyword frequencies provide a weak predictive signal. While advanced NLP techniques (TF-IDF) perform slightly better, they are insufficient as standalone indicators for financial forecasting.

## 5. Limitations and future work
**Limitations:**
1. **Simplicity of NLP Tools:** Even with TF-IDF, the model lacks financial context. For example, "Central bank cuts interest rates" might be scored as negative due to the word "cuts," but financially, it is usually a positive signal for the stock market.
2. **Market Complexity:** The stock market is influenced by countless interconnected macroeconomic factors (earnings reports, global events, institutional trading algorithms) that cannot be fully captured by just scanning the top 25 daily news headlines on Reddit.

**Future Work:**
1. **Advanced NLP Models:** Future extensions should replace `TextBlob` and `TF-IDF` with sophisticated, domain-specific large language models like **FinBERT**, which is explicitly trained on financial text and understands market terminology.
2. **Broader Feature Set:** Adding more technical and economic indicators, such as trading volume, inflation rates, moving averages, or the VIX (volatility index).
3. **Deep Learning Algorithms:** Experimenting with time-series forecasting models like LSTMs (Long Short-Term Memory networks) could capture sequential patterns over time much better than static tree-based models.

---

## AI Usage Statement
In accordance with the academic integrity guidelines, AI tools (ChatGPT) were used to assist in the completion of this project. The specific usages are documented below:
- **Code Debugging:** Used to fix `pandas` merge errors encountered when joining the S&P 500 dataset with the main dataset. *(Prompt: "Fix pandas merge error with SP500 dataset")*
- **Statistical Analysis:** Assisted in formatting the hypothesis testing code and interpreting the resulting p-value. *(Prompt: "Write t-test code to compare two groups in pandas")*
- **Model Building Boilerplate:** Used to generate the initial structural code for the TF-IDF Vectorizer and Logistic Regression models. *(Prompt: "Create machine learning model training code with sklearn")*
- **Documentation:** Assisted in outlining and structuring the README and project proposal files. *(Prompt: "Write a structured README for a data science project")*

All AI-generated outputs were thoroughly reviewed, manually tested, and customized to fit the exact needs and context of this project before submission.
