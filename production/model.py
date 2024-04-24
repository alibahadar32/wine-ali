# %% [markdown]
# # Wines classification model

# %%

import mlflow
mlflow.autolog()

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()

# %% [markdown]
# ## load and read data 

# %%
# Load the dataset with error handling to skip problematic rows
try:
    data = pd.read_csv(args.trainingdata)
except pd.errors.ParserError:
    data = pd.read_csv(args.trainingdata, error_bad_lines=False)

# %% [markdown]
# ## Drop rows with missing values in the 'description' and 'points' columns

# %%

data = data.dropna(subset=['description', 'points'])

# %% [markdown]
# ## Split the data into features (X) and target variable (y)

# %%

X = data['description']
y = data['points']

# %% [markdown]
# ## Split the data into training and testing sets

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Convert text data into numerical features using TF-IDF vectorization

# %%
# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# %% [markdown]
# ## Train a Multinomial Naive Bayes classifier

# %%

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)


# %% [markdown]
# ## Evaluate model

# %%
# Predict the test set labels
y_pred = clf.predict(X_test_vectorized)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


