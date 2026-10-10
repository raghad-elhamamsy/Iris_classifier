import joblib 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset

X, y = load_iris(return_X_y=True)

# Create and train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "trained_model.pkl")


