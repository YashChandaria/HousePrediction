from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import pickle
import numpy as np

# Load Dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

feature_names = housing.feature_names

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=41
)

# Scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=41
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=41,
        n_jobs=-1
    )
}

results = []

best_model = None
best_score = -999

for name, model in models.items():

    if name == "Linear Regression":
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_test_scaled)
    else:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append([
        name,
        round(r2, 4),
        round(rmse, 4)
    ])

    if r2 > best_score:
        best_score = r2
        best_model = model

# Results DataFrame
results_df = pd.DataFrame(
    results,
    columns=["Model", "R2 Score", "RMSE"]
)

print(results_df)

# Save Metrics
results_df.to_csv(
    "model_metrics.csv",
    index=False
)

# Save Models
pickle.dump(
    models["Linear Regression"],
    open("linear_model.pkl", "wb")
)

pickle.dump(
    models["Decision Tree"],
    open("decision_tree.pkl", "wb")
)

pickle.dump(
    models["Random Forest"],
    open("random_forest.pkl", "wb")
)

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

# Save Best Model
pickle.dump(
    best_model,
    open("best_model.pkl", "wb")
)

print("\nTraining Completed Successfully!")
print("Best Model Saved As: best_model.pkl")

