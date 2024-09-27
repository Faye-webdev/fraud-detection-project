# Importing necessary libraries
import pandas as pd

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the data
data = pd.read_csv("C:\\Users\\sowem\\Downloads\\creditcard.csv\\creditcard.csv")
print("Data loaded successfully.")
print(data.head())  # View the first few rows of the dataset

# Split the dataset into features and target variable
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate the model
print("Evaluating the model...")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
