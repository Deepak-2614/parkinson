# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pickle

# Load dataset
df = pd.read_csv("Parkinson/parkinsons.csv")
# Explore the dataset
#print(df.head())
#print(df.shape)
#print(df.isnull().sum())
#print(df.info())
#print(df.describe())
#print(df['status'].value_counts())

# Separate features and target
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define base classifiers
svm_clf = SVC(kernel='linear', probability=True)  # SVM with probability for voting
rf_clf = RandomForestClassifier(random_state=2)
gb_clf = GradientBoostingClassifier(random_state=2)

# Create the ensemble model using VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('rf', rf_clf), ('gb', gb_clf)],
    voting='soft'  # Use soft voting to consider the predicted probabilities
)

# Train the ensemble model
voting_clf.fit(X_train, Y_train)

# Evaluate the model on the training data
x_pred = voting_clf.predict(X_train)
training_data_accuracy = accuracy_score(x_pred, Y_train)
#print(f"Training Data Accuracy: {training_data_accuracy * 100:.2f}%")

# Evaluate the model on the test data
x_tst_pred = voting_clf.predict(X_test)
test_data_accuracy = accuracy_score(x_tst_pred, Y_test)
#print(f"Test Data Accuracy: {test_data_accuracy * 100:.2f}%")

# Save the model and scaler as a pickle file
with open('Parkinson/parkinsons_voting_clf.pkl', 'wb') as model_file:
    pickle.dump((voting_clf, scaler, X.columns.tolist()), model_file)

#print("Model saved as 'parkinsons_voting_clf.pkl'.")
