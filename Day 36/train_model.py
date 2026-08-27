import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from preprocessing import create_preprocessing_pipeline


# Load dataset
df = pd.read_csv("../datasets/titanic.csv")

# Keep only required columns
df = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived",
    ]
]

# Remove rows with missing target
df = df.dropna(subset=["Survived"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Create preprocessing pipeline
pipeline = create_preprocessing_pipeline()

# Fit ONLY on training data
X_train_processed = pipeline.fit_transform(X_train)

# Transform test data
X_test_processed = pipeline.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_processed, y_train)

# Save artifacts
joblib.dump(model, "model/best_titanic_model.pkl")
joblib.dump(pipeline, "model/pipeline.pkl")

print("Training completed successfully.")
print("Model saved to model/best_titanic_model.pkl")
print("Pipeline saved to model/pipeline.pkl")