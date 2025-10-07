import pandas as pd

df = pd.read_csv('misinformation_detection.csv')
bots = pd.read_csv('activity_botscore.csv')

#print("Misinformation data shape:", df.shape)
#print("Bot data shape:", bots.shape)
#print("\nMisinformation columns:\n", df.columns)
#print("\nBot columns:\n", bots.columns)

# Inspecting structure of the data.
#print("\n------Misinformation data-------")
#df.info()
#print(df.describe(include='all'))
#print(df.head())

#print("\n-------Bot data-------")
#bots.info()
#print(bots.head(3))

# Clean Column Names
df.columns = df.columns.str.lower().str.strip() # make everything lowercase and remove extra spaces

# Check for any missing values.
#print('\nMissing values per columns')
#print(df.isna().sum())
# Handling these missing values.
df['fact_check_source'] = df['fact_check_source'].fillna('not_fact_checked') # fill missing fact_check_source with 'not_fact_checked'

# Removal of duplicates.
before = len(df)
df = df.drop_duplicates(subset='post_id', keep='first') # drop duplicates based on post_id
after = len(df)
#print(f'\nRemoved {before - after} duplicate posts')

# Validate ranges and fix outliers.
numeric_cols = ['sentiment_score', 'toxicity_score', 'viral_score', 'engagement_score']
#print(df[numeric_cols].describe())

# clip to valid ranges.
df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
df['toxicity_score'] = df['toxicity_score'].clip(0, 1)
df['viral_score'] = df['viral_score'].clip(0, 1)

# Fix data and Language Formatiing
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') # convert to datetime, coerce errors to NaT
df['platform'] = df['platform'].str.lower().str.strip() # standardize platform names
df['language'] = df['language'].str.lower().str.strip() # standardize language
df = df[df['language'] == 'english'] # keep only English posts

# Final confirmation
#print("\nFinal dataset check:")
#print("Shape:", df.shape)
#print("Duplicates:", df.duplicated('post_id').sum())
#print("Missing values left:", df.isna().sum().sum())

#Save cleaned Data set.
#df.to_csv('misinformation_cleaned.csv', index=False)
print("Cleaned dataset saved as misinformation_cleaned.csv")

# NATURAL LANGUAGE PROCESSING (NLP) PREPROCESSING

import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    # Ensure text is string
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Keep %, numbers, and decimal points
    punctuation = string.punctuation.replace('%', '').replace('.', '')
    text = text.translate(str.maketrans('', '', punctuation))

    # Replace multiple dots (...) with a single space
    text = re.sub(r'\.{2,}', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove standalone non-alphanumeric characters except % and digits
    text = re.sub(r'[^a-z0-9%\s]', '', text)

    # Remove stopwords (common words like “the”, “and”, “is”)
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]

    # Rejoin
    return ' '.join(words)

# Apply cleaning function
df['cleaned_text'] = df['content_text'].apply(clean_text)
#print("\nSample cleaned text:")
#print(df[['content_text', 'cleaned_text']].head())  # show sample comparison

# Add a text length column
df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

# Save processed dataset
#df.to_csv('misinformation_textcleaned.csv', index=False)
print("\n Text preprocessing complete. Saved as misinformation_textcleaned.csv")



# Feature Engineering


import pandas as pd
import numpy as np

# Ensure engagement columns ('likes', 'shares', 'comments') are numeric.
# If any non-numeric or missing values exist, they are coerced to NaN and then replaced with 0.
# Ensure engagement columns are numeric (match the actual column names used below)
for col in ['like_count', 'share_count', 'comment_count']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)



# Social engagement metrics

# Create a column representing the total amount of engagement for each post.
# Adds up all likes, shares, and comments to capture overall reach or interaction.
df['total_engagement'] = df['like_count'] + df['share_count'] + df['comment_count']

# Create ratios showing the *proportion* of each type of engagement.
# These highlight whether a post is more likely to be shared, commented on, or liked.
# 1e-5 prevents division by zero when total_engagement = 0.
df['shares_ratio']   = df['share_count']   / (df['total_engagement'] + 1e-5)
df['comments_ratio'] = df['comment_count'] / (df['total_engagement'] + 1e-5)
df['likes_ratio']    = df['like_count']    / (df['total_engagement'] + 1e-5)


# Text-based features


# Count the number of words in each cleaned text post.
# The lambda takes each row's 'cleaned_text', converts to string (if missing),
# splits by spaces, and returns the length of the resulting list.
df['word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

# Compute the average word length for each post.
# The lambda builds a list of all token lengths using a list comprehension,
# then calculates the mean word length using np.mean.
# If there are no words in a post, it returns 0 to avoid division errors.
df['avg_word_length'] = df['cleaned_text'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
)

# Count how many words in the original text are fully uppercase.
# This helps detect posts that use capitalization for emphasis or aggression.
# The lambda loops through each word, checks if it's uppercase, and sums how many are True.
df['capital_word_count'] = df['content_text'].apply(
    lambda x: sum(1 for w in str(x).split() if w.isupper())
)

# Sentiment and toxicity category labels

# Function converts continuous sentiment scores into descriptive categories.
# Positive sentiment > 0.3, negative < -0.3, neutral otherwise.
def sentiment_label(val):
    if val > 0.3:
        return 'positive'
    elif val < -0.3:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_category'] = df['sentiment_score'].apply(sentiment_label)

# Function converts continuous toxicity scores into categories.
# High toxicity > 0.6, medium between 0.3–0.6, low otherwise.
def toxicity_label(val):
    if val > 0.6:
        return 'high'
    elif val > 0.3:
        return 'medium'
    else:
        return 'low'

df['toxicity_category'] = df['toxicity_score'].apply(toxicity_label)


# Platform moderation and fact-check indicators

# Create a binary flag showing whether a post was moderated.
# The lambda converts the 'moderation_action' to lowercase and checks if it’s one of
# the key moderation types. If yes, return 1 (True); otherwise, 0 (False).
df['was_moderated'] = df['moderation_action'].apply(
    lambda x: 1 if str(x).lower() in ['removed', 'warning label', 'demonetized'] else 0
)

# Create a binary flag showing whether a post has been fact-checked.
# If 'fact_check_source' is 'not_fact_checked', the lambda returns 0; otherwise, 1.
df['was_factchecked'] = df['fact_check_source'].apply(
    lambda x: 0 if str(x) == 'not_fact_checked' else 1
)


# Quick progress check
# Display a subset of key engineered columns to verify output.
# Helps confirm that numeric and categorical values look reasonable before saving.
print(df[['total_engagement', 'shares_ratio', 'comments_ratio', 'likes_ratio',
          'word_count', 'avg_word_length', 'capital_word_count',
          'sentiment_category', 'toxicity_category',
          'was_moderated', 'was_factchecked']].head())


#  Save final feature-engineered dataset

# Export the DataFrame with all original and engineered features to a new CSV file
# for the next stage of encoding and modeling.
df.to_csv('misinformation_features.csv', index=False)
print("\nFeature engineering complete. Saved as misinformation_features.csv")
