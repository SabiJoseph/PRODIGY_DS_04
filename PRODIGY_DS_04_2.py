import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import re

# Paths
train_data_path = r"E:\Research\PRODIGY_DS_04\Dataset\twitter_training.csv"
validation_data_path = r"E:\Research\PRODIGY_DS_04\Dataset\twitter_validation.csv"

# Load
train_data = pd.read_csv(train_data_path, header=None)
validation_data = pd.read_csv(validation_data_path, header=None)

# Columns
train_data.columns = ['tweet_id', 'topic', 'emotion', 'tweet']
validation_data.columns = ['tweet_id', 'topic', 'emotion', 'tweet']

# Check missing values
print(f"Missing values in training data:\n{train_data.isna().sum()}")
print(f"Missing values in validation data:\n{validation_data.isna().sum()}")

# Fill missing
train_data['tweet'].fillna("", inplace=True)
validation_data['tweet'].fillna("", inplace=True)

# Check after filling
print(f"Missing values after filling in training data:\n{train_data.isna().sum()}")
print(f"Missing values after filling in validation data:\n{validation_data.isna().sum()}")

# Clean text
def clean_text(text):
    if isinstance(text, str):  
        text = text.lower()
        text = re.sub(r"http\S+", "", text)  
        text = re.sub(r"\s+", " ", text)  
        text = re.sub(r"[^\w\s]", "", text)  
    return text

# Apply cleaning
train_data['clean_text'] = train_data['tweet'].apply(clean_text)
validation_data['clean_text'] = validation_data['tweet'].apply(clean_text)

# Encode labels
le = LabelEncoder()
train_data['encoded_sentiment'] = le.fit_transform(train_data['emotion'])
validation_data['encoded_sentiment'] = le.transform(validation_data['emotion'])

# Prepare data
X_train = train_data['clean_text']
y_train = train_data['encoded_sentiment']
X_val = validation_data['clean_text']
y_val = validation_data['encoded_sentiment']

# Check NaN in features
print(f"NaN values in X_train: {X_train.isna().sum()}")
print(f"NaN values in X_val: {X_val.isna().sum()}")

# Build model
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Classification report
train_class_report = classification_report(y_train, y_train_pred, target_names=le.classes_, output_dict=True)
val_class_report = classification_report(y_val, y_val_pred, target_names=le.classes_, output_dict=True)

# Print reports
print("Training Classification Report:")
print(train_class_report)

print("\nValidation Classification Report:")
print(val_class_report)

# Plot metrics
metrics = ['precision', 'recall', 'f1-score']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(metrics):
    axes[i].bar(
        ['Irrelevant', 'Negative', 'Neutral', 'Positive'],
        [train_class_report['Irrelevant'][metric], 
         train_class_report['Negative'][metric], 
         train_class_report['Neutral'][metric], 
         train_class_report['Positive'][metric]], color='b', alpha=0.6, label='Train'
    )
    axes[i].bar(
        ['Irrelevant', 'Negative', 'Neutral', 'Positive'],
        [val_class_report['Irrelevant'][metric], 
         val_class_report['Negative'][metric], 
         val_class_report['Neutral'][metric], 
         val_class_report['Positive'][metric]], color='r', alpha=0.6, label='Validation'
    )
    axes[i].set_title(f'{metric.capitalize()} Comparison')
    axes[i].set_xlabel('Sentiment')
    axes[i].set_ylabel(metric.capitalize())
    axes[i].legend()

plt.tight_layout()
plt.show()

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)

sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('Training Confusion Matrix')

sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title('Validation Confusion Matrix')

plt.tight_layout()
plt.show()

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
