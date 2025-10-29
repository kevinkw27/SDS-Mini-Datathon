import pandas as pd
df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# Removes duplicates and keeps unique rows
df_clean = df.drop_duplicates().copy()  # creates a new dataframe without the duplicates
df_clean.info()

############################
#######DECISION TREE########
############################
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Features & target
X = df_clean[['age', 'bmi', 'children', 'sex', 'smoker', 'region']].copy()
y = df_clean['charges']

# Encode categorical variables
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Range of depths to test
depth_values = list(range(1, 21))
train_rmse = []
test_rmse = []

for d in depth_values:
    # Create regression tree
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # RMSE
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Plot RMSE vs Depth
plt.figure(figsize=(8,5))
plt.plot(depth_values, train_rmse, marker='o', label='Train RMSE', color='teal')
plt.plot(depth_values, test_rmse, marker='o', label='Test RMSE', color='orange')
plt.title("Decision Tree RMSE vs Max Depth")
plt.xlabel("Tree Depth")
plt.ylabel("RMSE")
plt.legend()
plt.show()

#Optimal depth = 4
best_depth = depth_values[np.argmin(test_rmse)]
print(f"Optimal max_depth: {best_depth}")


# Create Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42, max_depth=4)
dt_reg.fit(X_train, y_train)

# Predict & evaluate RMSE = 4326.98
y_pred = dt_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")

# Replace node value arrays with scalar mean for readable plotting
for i in range(dt_reg.tree_.node_count):
    dt_reg.tree_.value[i] = np.mean(dt_reg.tree_.value[i])

# Plot the regression tree
plt.figure(figsize=(20,10))
plot_tree(
    dt_reg,
    feature_names=X.columns,
    rounded=True,
    filled=True,
    max_depth=2
)
plt.title("Decision Tree Regressor: Predicting Charges")
plt.show()
"""
############################
####### RANDOM FOREST ######
############################
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Features & target
X = df_clean[['age', 'bmi', 'children', 'sex', 'smoker', 'region']].copy()
y = df_clean['charges']

# Encode categorical variables
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Range of depths to test
depth_values = list(range(2, 21)) + [None]
train_rmse, test_rmse = [], []

for d in depth_values:
    model = RandomForestRegressor(
        n_estimators=200,       # number of trees
        max_depth=d,            # per-tree depth limit
        max_samples=0.5,        # use 50% of rows per tree
        min_samples_leaf=5,     # bigger leaves to prevent overfitting
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # RMSE
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Find optimal depth - Optimal max_depth = 5
best_depth = depth_values[np.argmin(test_rmse)]
print(f"Optimal max_depth: {best_depth}")

# Plot RMSE vs max_depth
labels = [str(d) for d in depth_values[:-1]] + ["None"]
plt.figure(figsize=(9,5))
plt.plot(labels, train_rmse, marker='o', label='Train RMSE')
plt.plot(labels, test_rmse, marker='o', label='Test RMSE')
plt.title("Random Forest Regression: RMSE vs max_depth")
plt.xlabel("max_depth (None = fully grown trees)")
plt.ylabel("RMSE")
plt.legend()
plt.show()


# Train final Random Forest with optimal depth
rf_final = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    max_samples=0.5,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X_train, y_train)

# Predict & evaluate - RMSE = 4193.08
y_pred_final = rf_final.predict(X_test)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
print(f"Final Random Forest RMSE: {rmse_final:.2f}")

#Feature importance
fi = pd.Series(rf_final.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
fi.plot(kind='bar')
plt.title("Random Forest — Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
"""

############################
########### XG BOOST #######
############################
"""
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Features & target
X = df_clean[['age', 'bmi', 'children', 'sex', 'smoker', 'region']].copy()
y = df_clean['charges']

# Encode categorical variables
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

max_depth_values = list(range(2, 11))      # tree depth
learning_rates = [0.01, 0.05, 0.1, 0.2]   # learning rates

results = []
# Loop over all combinations
for depth in max_depth_values:
    for lr in learning_rates:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'max_depth': depth, 'learning_rate': lr, 'rmse': rmse})

# Convert to DataFrame for easy inspection
results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['rmse'].idxmin()]
print("Best Parameters:")
print(best_row)


# Train final model with best params
xgb_final = XGBRegressor(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)
xgb_final.fit(X_train, y_train)

# Predict & evaluate - RMSE = 4285.32
y_pred_final = xgb_final.predict(X_test)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
print(f"\nFinal XGBoost RMSE: {rmse_final:.2f}")

# ----------------------------
# Feature importance plot
# ----------------------------
feature_importance = pd.Series(xgb_final.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(8,5))
plt.bar(feature_importance.index, feature_importance.values, color='teal')
plt.title("XGBoost Regressor Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()
"""
