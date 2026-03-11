import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")[["v1", "v2"]]
df.columns = ["label", "text"]

# Clean labels
df["label"] = df["label"].str.strip().str.lower()

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")