from datasets import load_dataset
import pandas as pd
import numpy as np

# 
# Load data
# 
bots_df = pd.read_csv("bot_detection_data.csv")  # your tweet-level bot dataset
npl = load_dataset("junaid1993/Dataset_Bot_Detection")
twitter = load_dataset("airt-ml/twitter-human-bots")

twitter_df = twitter["train"].to_pandas()
npl_df = npl["train"].to_pandas()

# 
# Standardise column names
#
def standardise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

bots_df   = standardise_cols(bots_df)
twitter_df= standardise_cols(twitter_df)
npl_df    = standardise_cols(npl_df)

# After standardisation, expected columns:
# bots_df: user_id, username, tweet, retweet_count, mention_count,
#          follower_count, verified, bot_label, location, created_at, hashtags
# twitter_df: created_at, default_profile, default_profile_image, description,
#             favourites_count, followers_count, friends_count, geo_enabled, id,
#             lang, location, profile_background_image_url, profile_image_url,
#             screen_name, statuses_count, verified, average_tweets_per_day,
#             account_age_days, account_type
# npl_df: text_data, label

# 
#  Harmonise key names
#
# Twitter user table: align to user_id / username
twitter_df = twitter_df.rename(
    columns={
        "id": "user_id",
        "screen_name": "username",
    }
)

# Bots table often already aligned after step 2.
# If your CSV used e.g. "User ID" or "Bot Label" they are now "user_id" / "bot_label".

#
#  Drop bad rows & strip strings
# 
# Keep rows with essential fields
bots_df   = bots_df.dropna(subset=["user_id", "username", "tweet"])
twitter_df= twitter_df.dropna(subset=["user_id"])
npl_df    = npl_df.dropna(subset=["text_data", "label"])

# Strip whitespace from all object columns
def strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    return df

bots_df    = strip_object_cols(bots_df)
twitter_df = strip_object_cols(twitter_df)
npl_df     = strip_object_cols(npl_df)

#
#  Convert dtypes
# 
# IDs to string for safe joins
bots_df["user_id"]    = bots_df["user_id"].astype(str)
twitter_df["user_id"] = twitter_df["user_id"].astype(str)

# Numeric conversions with safety (coerce then fill)
def to_int_safe(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

bots_numeric = ["retweet_count", "mention_count", "follower_count"]
for c in bots_numeric:
    if c in bots_df.columns:
        bots_df[c] = to_int_safe(bots_df[c])

tw_numeric = ["followers_count", "friends_count", "favourites_count", "statuses_count", "account_age_days"]
for c in tw_numeric:
    if c in twitter_df.columns:
        # account_age_days may be float; keep as float then later cast if needed
        if c == "account_age_days":
            twitter_df[c] = pd.to_numeric(twitter_df[c], errors="coerce")
        else:
            twitter_df[c] = to_int_safe(twitter_df[c])

# Booleans
if "verified" in bots_df.columns:
    # verified in bots_df could be bool or 'True'/'False'
    bots_df["verified"] = bots_df["verified"].astype(str).str.lower().isin(["true", "1", "yes"])

if "verified" in twitter_df.columns:
    twitter_df["verified"] = twitter_df["verified"].astype(str).str.lower().isin(["true", "1", "yes"])

# Dates
for df_, col in [(bots_df, "created_at"), (twitter_df, "created_at")]:
    if col in df_.columns:
        df_[col] = pd.to_datetime(df_[col], errors="coerce")

# Labels
# bots_df: expect "bot_label" as 0/1; ensure int 0/1
if "bot_label" in bots_df.columns:
    bots_df["bot_label"] = pd.to_numeric(bots_df["bot_label"], errors="coerce").fillna(0).astype(int)

# npl_df: expect "label" as 'bot'/'human' -> map to 1/0
npl_df["label"] = npl_df["label"].str.lower().map({"bot": 1, "human": 0})
npl_df["label"] = npl_df["label"].fillna(0).astype(int)

# 
# Feature engineering - bots_df (tweet-level)
# 
# Text features (vectorised)
bots_df["tweet"] = bots_df["tweet"].astype(str)
bots_df["tweet_length_chars"] = bots_df["tweet"].str.len()
bots_df["word_count"]         = bots_df["tweet"].str.split().str.len()
bots_df["has_link"]           = bots_df["tweet"].str.contains(r"http", case=False, na=False).astype(int)
bots_df["has_mention"]        = bots_df["tweet"].str.contains(r"@",    case=False, na=False).astype(int)
bots_df["has_hashtag"]        = bots_df["tweet"].str.contains(r"#",    case=False, na=False).astype(int)

# Behaviour ratios
bots_df["engagement_ratio"] = (
    (bots_df.get("retweet_count", 0) + bots_df.get("mention_count", 0))
    / (bots_df.get("follower_count", 0) + 1)
)

# Verified as int for simple models
bots_df["is_verified_int"] = bots_df["verified"].astype(int) if "verified" in bots_df.columns else 0

#
# Feature engineering - twitter_df (user-level)
# 
# Some rows may have NaN account_age_days; guard div-by-zero
twitter_df["followers_to_friends"] = (
    twitter_df.get("followers_count", 0) / (twitter_df.get("friends_count", 0) + 1)
)

if "account_age_days" in twitter_df.columns:
    twitter_df["tweets_per_day"] = (
        twitter_df.get("statuses_count", 0) / (twitter_df["account_age_days"].fillna(0) + 1)
    )
else:
    twitter_df["tweets_per_day"] = np.nan

twitter_df["verified_int"]       = twitter_df["verified"].astype(int)
twitter_df["description_length"] = twitter_df["description"].astype(str).str.len() if "description" in twitter_df.columns else 0

# Keep a slim user feature table for optional enrichment
user_features = twitter_df[[
    "user_id",
    "followers_to_friends",
    "tweets_per_day",
    "verified_int",
    "description_length"
]].drop_duplicates()

#
# merge user features into bots_df (if user_id overlaps)
# 
merged_df = bots_df.merge(user_features, on="user_id", how="left")

# Fill any remaining numeric NaNs to keep models happy
num_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[num_cols] = merged_df[num_cols].fillna(0)

# 
#  Save cleaned outputs
# 
merged_df.to_csv("bot_processed.csv", index=False)       # tweet-level, with engineered features
#twitter_df.to_csv("twitter_processed.csv", index=False)  # user-level, engineered
#npl_df.to_csv("npl_processed.csv", index=False)          # text-only, labels mapped to 0/1

print("\n\n---------------------------NEW LINE---------------------------- \n\n")

print(f'Processed twitter_df head:\n {twitter_df.head(2)}\n\n npl_df head: {npl_df.head(2)}\n\n merged_df head:\n\n {merged_df.head(2)}')