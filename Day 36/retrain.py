import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def retrain_model():
    """
    Retrains the Titanic model and saves it as model_v2.pkl.
    """

    # Load dataset
    df = pd.read_csv("../datasets/titanic.csv")

    # Features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Load preprocessing pipeline
    pipeline = joblib.load("model/pipeline.pkl")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Transform data
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100
    )

    model.fit(X_train_processed, y_train)

    # Evaluate model
    predictions = model.predict(X_test_processed)

    score = f1_score(y_test, predictions)

    print(f"New Model F1 Score: {score:.4f}")

    # Save new version
    joblib.dump(model, "model/model_v2.pkl")

    print("New model saved as model/model_v2.pkl")

    return score


if __name__ == "__main__":
    retrain_model()