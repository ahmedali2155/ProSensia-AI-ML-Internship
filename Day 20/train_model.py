import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("titanic.csv")

# Drop unnecessary columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Display updated dataset information
print(df.info())

# Check remaining missing values
print(df.isnull().sum())

# Display first five rows
print(df.head())

# Label Encoding
label_encoders = {}

categorical_columns = ["Sex", "Embarked"]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

# Save encoders
joblib.dump(label_encoders, "label_encoders.pkl")

print("\nLabel Encoding Completed Successfully!")
print(df.head())

print("\nEncoded Data:")
# -----------------------------
# Train/Test Split
# -----------------------------

from sklearn.model_selection import train_test_split

# Features
X = df.drop("Survived", axis=1)

# Target
y = df["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain/Test Split Completed Successfully!")

print("Training Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)
# -----------------------------
# Train Logistic Regression Model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Trained Successfully!")
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Trained Model
# -----------------------------

joblib.dump(model, "best_titanic_model.pkl")

print("\nModel saved successfully!")
