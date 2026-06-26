import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

train = pd.read_csv("train_data.txt", sep=" ::: ", engine="python", names=["Title", "Genre", "Plot"])
print(train.head())
print(train.isnull().sum())

X = train["Title"] + " " + train["Plot"]
y = train["Genre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')),
    ('clf', LinearSVC(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
])

print("Training Pipeline...")
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

import pandas as pd

test_df = pd.read_csv("test_data.txt", sep=" ::: ", engine="python", header=None)
sol_df = pd.read_csv("test_data_solution.txt", sep=" ::: ", engine="python", header=None)

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

test_df = pd.read_csv("test_data.txt", sep=" ::: ", engine="python", header=None)
sol_df = pd.read_csv("test_data_solution.txt", sep=" ::: ", engine="python", header=None)

test_df[1] = test_df[1].fillna('').astype(str)
test_df[2] = test_df[2].fillna('').astype(str)
test_data_features = test_df[1] + " " + test_df[2]

print("Predicting on Test Data...")
test_preds = pipeline.predict(test_data_features)

actual_genres = sol_df[2].fillna('').astype(str).str.strip().str.lower()

print("Test Set Accuracy:", accuracy_score(actual_genres, test_preds))
print("\nClassification Report:\n")
print(classification_report(actual_genres, test_preds))