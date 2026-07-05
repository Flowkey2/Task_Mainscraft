import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Prepare features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Save model as pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model pickled successfully to model.pkl")
