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

After merging, I engineered new features to translate text data into numerical formats that a machine learning model can understand:
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
- `dropna()`: Cleans up the dataset by removing the final row which gets a `NaN` value due to the shift operation.

### 3.2 Exploratory Data Analysis (EDA)
To understand the underlying patterns in the data, I conducted Exploratory Data Analysis using `seaborn` and `matplotlib`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Distribution of Target Variable
sns.countplot(x=df["Label"])
plt.show()

# 2. Sentiment Score Distribution
sns.histplot(df["sentiment"], bins=30, kde=True)
plt.show()

# 3. Sentiment spread across positive vs negative market days
sns.boxplot(x=df["Label"], y=df["sentiment"])
plt.show()
```
**Code Explanation & EDA Interpretations:** 
1. **Countplot:** Calculates and visualizes the total count of `0`s and `1`s in our target variable. *Interpretation:* It verified that our classes (Market Up `1` vs Market Down `0`) are relatively balanced, preventing the model from becoming biased towards a majority class.
2. **Histplot with KDE:** Plots a histogram of the sentiment scores divided into 30 bins, with a Kernel Density Estimate (KDE) line to show the continuous probability curve. *Interpretation:* It revealed that the daily sentiment scores follow a near-normal distribution but lean slightly towards the negative side, which makes sense given the often pessimistic nature of global news.
3. **Boxplot:** Compares the median, quartiles, and outliers of the sentiment scores between the `0` group and `1` group. *Interpretation:* It visually demonstrated that the medians and interquartile ranges of sentiment scores for both groups are nearly identical, providing an early hint that sentiment alone might not be a strong differentiator.

### 3.3 Hypothesis Testing
To scientifically validate the observation from the boxplot, I performed an independent two-sample t-test. The goal was to check if the mean sentiment on days before the market goes up is statistically different from the mean sentiment on days before it goes down.

```python
from scipy.stats import ttest_ind

up = df[df["Label"] == 1]["sentiment"]
down = df[df["Label"] == 0]["sentiment"]

t_stat, p_value = ttest_ind(up, down)
print("p-value:", p_value)
# Output: p-value: 0.3282004042073472
```
**Code Explanation & Statistical Interpretation:** 
- The data is split into two arrays: `up` (sentiment scores when market goes up) and `down` (sentiment scores when market goes down).
- `ttest_ind()` computes the T-test for the means of two independent samples of scores.
- *Interpretation:* The resulting p-value is **0.3282**. In statistics, an alpha level of 0.05 is the standard threshold. Because $0.3282 > 0.05$, we fail to reject the null hypothesis. This proves that mathematically, there is no significant difference in the daily news sentiment between positive and negative market days.

### 3.4 Machine Learning Models
Despite the weak statistical correlation, I trained Machine Learning models to see if a combination of features (`sentiment`, `headline_length`, `word_count`, `SP500`) could find complex, non-linear patterns to predict the market.

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

features = ["sentiment", "headline_length", "word_count", "SP500"]
X = df[features]
y = df["Label"]

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Training
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_rf))
```
**Code Explanation:** 
- `train_test_split(test_size=0.2)`: Reserves 80% of the historical data for training the model, and holds out 20% to test its predictive accuracy on unseen data.
- `RandomForestClassifier()`: Initializes an ensemble model made of many decision trees to prevent overfitting.
- `fit()` and `predict()`: The model learns the relationship between the features (X) and the market direction (y), and then predicts outcomes on the test set.

## 4. Findings
The machine learning phase solidified the findings from our statistical tests. 

- **Decision Tree Classifier:** 
  - Accuracy: **46.98%** 
  - ROC-AUC: **0.4687**
- **Random Forest Classifier:** 
  - Accuracy: **50.50%** 
  - ROC-AUC: **0.4999**

**Interpretation of Metrics:**
An accuracy of ~50% and a ROC-AUC score of ~0.5 in a balanced binary classification problem indicates that the models are performing no better than random guessing (flipping a coin). The models could not extract a reliable predictive signal from the provided features.

**Feature Importance Analysis:**
Extracting the feature importances from the Random Forest model provided the following weights:
1. **SP500:** 0.267
2. **Sentiment:** 0.266
3. **Headline Length:** 0.250
4. **Word Count:** 0.215

*Insight:* Although the S&P 500 index and the Sentiment Score were the strongest predictors, their predictive power was almost equal to arbitrary text metrics like headline length and word count. This confirms the overall conclusion: **Daily news sentiment alone, when analyzed through simple NLP methods, is completely insufficient to predict stock market movements.**

## 5. Limitations and future work
**Limitations:**
1. **Simplicity of NLP Tools:** `TextBlob` uses a basic lexicon-based approach (counting positive vs negative words). It fails to capture nuanced financial context. For example, "Central bank cuts interest rates" might be scored as negative due to the word "cuts," but financially, it is usually a positive signal for the stock market.
2. **Market Complexity:** The stock market is influenced by countless interconnected macroeconomic factors (earnings reports, global events, institutional trading algorithms) that cannot be fully captured by just scanning the top 25 daily news headlines on Reddit.

**Future Work:**
1. **Advanced NLP Models:** Future extensions should replace `TextBlob` with sophisticated, domain-specific large language models like **FinBERT**, which is explicitly trained on financial text and understands market terminology.
2. **Broader Feature Set:** Adding more technical and economic indicators, such as trading volume, inflation rates, moving averages, or the VIX (volatility index).
3. **Deep Learning Algorithms:** Experimenting with time-series forecasting models like LSTMs (Long Short-Term Memory networks) could capture sequential patterns over time much better than static tree-based models.

---

## AI Usage Statement
In accordance with the academic integrity guidelines, AI tools (ChatGPT) were used to assist in the completion of this project. The specific usages are documented below:
- **Code Debugging:** Used to fix `pandas` merge errors encountered when joining the S&P 500 dataset with the main dataset. *(Prompt: "Fix pandas merge error with SP500 dataset")*
- **Statistical Analysis:** Assisted in formatting the hypothesis testing code and interpreting the resulting p-value. *(Prompt: "Write t-test code to compare two groups in pandas")*
- **Model Building Boilerplate:** Used to generate the initial structural code for the models and extract feature importances. *(Prompt: "Create machine learning model training code with sklearn")*
- **Documentation:** Assisted in outlining and structuring the README and project proposal files. *(Prompt: "Write a structured README for a data science project")*

All AI-generated outputs were thoroughly reviewed, manually tested, and customized to fit the exact needs and context of this project before submission.
