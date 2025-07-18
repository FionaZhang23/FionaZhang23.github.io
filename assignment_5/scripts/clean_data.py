import os
import json
import gzip
import pandas as pd
from utils import sample_for_testing


input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
dataset = []
for l in input_file:
    d = eval(l)
    dataset.append(d)
input_file.close()

print(f"Loaded {len(dataset)} raw records.")

df = pd.DataFrame(dataset)
cleaned_df = df.dropna(subset=["hours"])
cleaned_df = cleaned_df.copy()
cleaned_df["year"] = cleaned_df["date"].apply(lambda x: int(str(x)[:4]))

print(f"Remaining after dropping missing 'hours': {cleaned_df.shape[0]} rows.")
cleaned_df_sampled = sample_for_testing(cleaned_df)
filtered_df = cleaned_df_sampled[["text", "hours", "year"]]

output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/data"
os.makedirs(output_dir, exist_ok=True)

csv_main = os.path.join(output_dir, "steam_reviews_cleaned.csv")


filtered_df.to_csv(csv_main, index=False)
'''

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import re

# Load the cleaned dataset
data_path = "/deac/csc/classes/csc373/zhanx223/assignment_5/data/steam_reviews_cleaned.csv"
df = pd.read_csv(data_path)

# Ensure no missing text
df = df.dropna(subset=["text"])

# Preprocess: lowercase, remove non-alphabetic chars, split, remove stop words
def clean_text(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)  # only a-z words
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return words

# Apply cleaning
df["tokens"] = df["text"].apply(clean_text)

# Flatten all words into one list
all_words = [word for tokens in df["tokens"] for word in tokens]

# Count frequency
word_counts = Counter(all_words)

# Show top 20 most frequent words
print("🔝 Top 100 most frequent words (excluding stop words):")
for word, freq in word_counts.most_common(100):
    print(f"{word}: {freq}")
'''