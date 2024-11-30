# Import
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

# Download
nltk.download('stopwords')

# Paths
train_data_path = r"E:\Research\PRODIGY_DS_04\Dataset\twitter_training.csv"
validation_data_path = r"E:\Research\PRODIGY_DS_04\Dataset\twitter_validation.csv"

# Load
column_names = ["tweetid", "topic", "sentiment", "comment"]
train_data = pd.read_csv(train_data_path, header=None, names=column_names)
validation_data = pd.read_csv(validation_data_path, header=None, names=column_names)

# Clean
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"#", "", text)  # Remove hashtags symbol
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return text.lower().strip()
    else:
        return ""  

# Columns
text_column = 'comment'  
sentiment_column = 'sentiment'  

# Clean text (train)
train_data['clean_text'] = train_data[text_column].apply(clean_text)
train_data = train_data.dropna(subset=['clean_text', sentiment_column])
train_data = train_data[train_data['clean_text'] != ""]

# Clean text (validation)
validation_data['clean_text'] = validation_data[text_column].apply(clean_text)
validation_data = validation_data.dropna(subset=['clean_text', sentiment_column])
validation_data = validation_data[validation_data['clean_text'] != ""]

# Stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words and word != 'game'])

# Remove stopwords (train)
train_data['clean_text'] = train_data['clean_text'].apply(remove_stopwords)

# Remove stopwords (validation)
validation_data['clean_text'] = validation_data['clean_text'].apply(remove_stopwords)

# Positive and Negative (train)
positive_tweets = train_data[train_data[sentiment_column] == 'Positive']['clean_text']
negative_tweets = train_data[train_data[sentiment_column] == 'Negative']['clean_text']

# Positive and Negative (validation)
positive_tweets_val = validation_data[validation_data[sentiment_column] == 'Positive']['clean_text']
negative_tweets_val = validation_data[validation_data[sentiment_column] == 'Negative']['clean_text']

# Combine positive text (train)
positive_text = " ".join(positive_tweets)

# Combine negative text (train)
negative_text = " ".join(negative_tweets)

# Combine positive text (validation)
positive_text_val = " ".join(positive_tweets_val)

# Combine negative text (validation)
negative_text_val = " ".join(negative_tweets_val)

# Word Cloud (positive train)
positive_wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(positive_text)

# Word Cloud (negative train)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(negative_text)

# Word Cloud (positive validation)
positive_wordcloud_val = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(positive_text_val)

# Word Cloud (negative validation)
negative_wordcloud_val = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(negative_text_val)

# Plot positive word cloud (train)
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title("Word Cloud - Positive Sentiment (Training Data)")
plt.axis('off')
plt.show()

# Plot negative word cloud (train)
plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title("Word Cloud - Negative Sentiment (Training Data)")
plt.axis('off')
plt.show()

# Sentiment distribution (train)
plt.figure(figsize=(8, 5))
sns.countplot(data=train_data, x=sentiment_column, order=train_data[sentiment_column].value_counts().index)
plt.title("Sentiment Distribution - Training Data")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Plot positive word cloud (validation)
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud_val, interpolation='bilinear')
plt.title("Word Cloud - Positive Sentiment (Validation Data)")
plt.axis('off')
plt.show()

# Plot negative word cloud (validation)
plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud_val, interpolation='bilinear')
plt.title("Word Cloud - Negative Sentiment (Validation Data)")
plt.axis('off')
plt.show()

# Sentiment distribution (validation)
plt.figure(figsize=(8, 5))
sns.countplot(data=validation_data, x=sentiment_column, order=validation_data[sentiment_column].value_counts().index)
plt.title("Sentiment Distribution - Validation Data")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
