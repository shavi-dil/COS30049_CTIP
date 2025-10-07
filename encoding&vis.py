import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encoding and SPlitting.

# Load dataset
df = pd.read_csv('misinformation_features.csv')

#define target(y - dependent variable) and features (x - independent variables)

# Create target column (misinformation_flag), values: 1 = fact checked and moderated, likely misinformation: 0 niethier, likely normal

df['misinformation_flag'] = (
    (df['was_factchecked'] == 1) | (df['was_moderated'] == 1)
).astype(int)

target = 'misinformation_flag'

#Define feature set (X) and target (y)
drop_cols = ['post_id', 'content_text', 'cleaned_text', target]
X = df.drop(columns=drop_cols, errors='ignore')
y = df[target]

# Encode categorical columns

cat_cols = X.select_dtypes(include=['object']).columns

# use label encoders
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}")

# Split dataset (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# checks
#print("\n Data split complete:")
#print(f"Training samples: {X_train.shape[0]}")
#print(f"Testing samples:  {X_test.shape[0]}")
#print(f"Features:         {X_train.shape[1]}")


# Plotting

# Step 7: Exploratory Data Analysis & Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#  Ensure target column exists 
if 'misinformation_flag' not in df.columns:
    df['misinformation_flag'] = (
        (df['was_factchecked'] == 1) | (df['was_moderated'] == 1)
    ).astype(int)

#  Basic info 
print("\nData shape:", df.shape)
print("\nColumn overview:")
print(df.columns.tolist())
print("\nMissing values per column:\n", df.isna().sum())

#  Class balance 
plt.figure(figsize=(6,4))
sns.countplot(x='misinformation_flag', data=df, palette='coolwarm')
plt.title('Class Balance: Misinformation vs Normal')
plt.xlabel('Misinformation Flag (1 = likely misinformation)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

print("\nClass distribution:")
print(df['misinformation_flag'].value_counts(normalize=True) * 100)

#  Correlation heatmap (numerical only) 
num_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,8))
sns.heatmap(num_df.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# elationship between key features and misinformation 
plt.figure(figsize=(6,4))
sns.boxplot(x='misinformation_flag', y='toxicity_score', data=df, palette='viridis')
plt.title('Toxicity Score vs Misinformation Flag')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='misinformation_flag', y='sentiment_score', data=df, palette='mako')
plt.title('Sentiment Score vs Misinformation Flag')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='misinformation_flag', y='total_engagement', data=df, palette='crest')
plt.title('Engagement vs Misinformation Flag')
plt.tight_layout()
plt.show()

