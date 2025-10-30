import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")


##################################
###########RANDOM FOREST #########
##################################

##################################
#####RANDOMZIEDCV SEARCH #########
##################################

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Encode categorical variables
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

y = df['charges']  # Original scale

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Random Forest and parameter grid
rf = RandomForestRegressor(random_state=1, n_jobs=-1)

param_dist = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 5, 10, 15, 20],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'bootstrap': [True, False],
    'max_samples': [0.5, 0.7, 1.0]
}

# Randomized Search CV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,              # number of parameter combinations to try
    scoring='neg_mean_squared_error',
    cv=10,
    random_state=1,
    n_jobs=-1,
    verbose=1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best hyperparameters
print("\nBest Random Forest Parameters:")
print(random_search.best_params_)

# Evaluate on Test Set
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mean_actual = np.mean(y_test)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nRandom Forest Metrics (Test Set, Original Scale):")
print(f"RMSE: {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R²: {r2:.4f}")

# Feature Importance
fi = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
fi.plot(kind='bar', color='skyblue')
plt.title("Random Forest — Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()



##################################
#####LOG CHARGES #################
##################################
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Encode categorical variables
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# Log-transform target to reduce skew
y = np.log(df['charges'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Random Forest and parameter grid
rf = RandomForestRegressor(random_state=1, n_jobs=-1)

param_dist = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 5, 10, 15, 20],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'bootstrap': [True, False],
    'max_samples': [0.5, 0.7, 1.0]
}

# Randomized Search CV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,              # number of parameter combinations to try
    scoring='neg_mean_squared_error',
    cv=10,
    random_state=1,
    n_jobs=-1,
    verbose=1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best hyperparameters
print("\nBest Random Forest Parameters:")
print(random_search.best_params_)

# Evaluate on Test Set
best_rf = random_search.best_estimator_
y_pred_log = best_rf.predict(X_test)
y_pred = np.exp(y_pred_log)        
y_test_orig = np.exp(y_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test, y_pred_log)
mean_actual = np.mean(y_test_orig)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nRandom Forest Metrics (Test Set):")
print(f"RMSE: {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R² (log scale): {r2:.4f}")

# Feature Importance
fi = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
fi.plot(kind='bar', color='skyblue')
plt.title("Random Forest — Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


##################################
#####INTERACTION FEATURES ########
##################################

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Encode categorical variables
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# Add interaction features
X['age_smoker'] = X['age'] * X['smoker_binary']
X['bmi_smoker'] = X['bmi'] * X['smoker_binary']
X['age_bmi'] = X['age'] * X['bmi']

# Log-transform target to reduce skew
y = np.log(df['charges'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Random Forest and focused parameter grid
rf = RandomForestRegressor(random_state=1, n_jobs=-1)

param_dist = {
    'n_estimators': [200, 400, 600],
    'max_depth': [10, 15, 20, None],
    'max_features': ['sqrt', 0.5, 0.7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'bootstrap': [True],
    'max_samples': [0.5, 0.7, 1.0]
}

# Randomized Search CV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,              # number of parameter combinations to try
    scoring='neg_mean_squared_error',
    cv=10,
    random_state=1,
    n_jobs=-1,
    verbose=1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best hyperparameters
print("\nBest Random Forest Parameters:")
print(random_search.best_params_)

# Evaluate on Test Set
best_rf = random_search.best_estimator_
y_pred_log = best_rf.predict(X_test)
y_pred = np.exp(y_pred_log)        
y_test_orig = np.exp(y_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test, y_pred_log)
mean_actual = np.mean(y_test_orig)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nRandom Forest Metrics (Test Set):")
print(f"RMSE: {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R² (log scale): {r2:.4f}")

# Feature Importance
fi = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
fi.plot(kind='bar', color='skyblue')
plt.title("Random Forest — Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


##################################
#################XG BOOST#########
##################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##################################
#####LOG CHARGES #################
##################################

# --- Encode categorical variables ---
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# --- Concatenate features ---
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# --- Log-transform the target variable ---
y = np.log(df['charges'])

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# --- GridSearchCV with 10-fold CV ---
param_grid = {
    'max_depth': list(range(2, 11)),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [200],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb = XGBRegressor(random_state=1, objective='reg:squarederror', n_jobs=-1)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=10,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters (10-fold CV):", grid_search.best_params_)
print("Best CV RMSE (log-scale):", np.sqrt(-grid_search.best_score_))

# --- Train final model on full training set ---
xgb_final = grid_search.best_estimator_
y_train_pred_log = xgb_final.predict(X_train)
y_test_pred_log = xgb_final.predict(X_test)

# --- Convert back to original scale ---
y_train_pred = np.exp(y_train_pred_log)
y_test_pred = np.exp(y_test_pred_log)
y_train_actual = np.exp(y_train)
y_test_actual = np.exp(y_test)

# --- Evaluation Metrics (original scale) ---
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
mae = mean_absolute_error(y_test_actual, y_test_pred)
r2 = r2_score(y_test_actual, y_test_pred)
mean_actual = np.mean(y_test_actual)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nXGBoost Test Metrics (with log-transform):")
print(f"RMSE (original scale): {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE (original scale): {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R² (original scale): {r2:.4f}")

# --- Feature Importance Plot ---
feature_importance = pd.Series(xgb_final.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(8,5))
plt.bar(feature_importance.index, feature_importance.values, color='teal')
plt.title("XGBoost Regressor Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Residual Plot (log scale) ---
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
x_min = min(y_train_pred_log.min(), y_test_pred_log.min())
x_max = max(y_train_pred_log.max(), y_test_pred_log.max())

plt.scatter(y_train_pred_log, y_train_pred_log - y_train, color="green", s=15, alpha=0.6, label='Train data')
plt.scatter(y_test_pred_log, y_test_pred_log - y_test, color="blue", s=15, alpha=0.6, label='Test data')
plt.hlines(y=0, xmin=x_min, xmax=x_max, colors='red', linewidth=2, linestyles='dashed')

plt.xlabel("Predicted log(Charges)", fontsize=12)
plt.ylabel("Residuals (Predicted - Actual)", fontsize=12)
plt.title("Residuals (XGBoost Regression on log-transformed Charges)", fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#############################
######RANDOMISEDSEARCHCV#####
#############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Encode categorical variables ---
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# --- Concatenate features ---
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# --- Log-transform the target variable ---
y = np.log(df['charges'])

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# --- Expanded parameter space for RandomizedSearchCV ---
param_dist = {
    'n_estimators': [200, 400, 600, 800],
    'max_depth': [2, 3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0.1, 1, 5],
    'reg_alpha': [0, 0.1, 0.5]
}

# --- RandomizedSearchCV with 10-fold CV ---
xgb = XGBRegressor(objective='reg:squarederror', random_state=1, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,
    cv=10,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=1
)

search.fit(X_train, y_train)

print("Best parameters (10-fold CV):", search.best_params_)
print("Best CV RMSE (log-scale):", abs(search.best_score_))

# --- Train final model with best parameters ---
xgb_final = search.best_estimator_
xgb_final.fit(X_train, y_train)

# --- Predictions ---
y_train_pred_log = xgb_final.predict(X_train)
y_test_pred_log = xgb_final.predict(X_test)

# --- Convert predictions back to original scale ---
y_train_pred = np.exp(y_train_pred_log)
y_test_pred = np.exp(y_test_pred_log)
y_train_actual = np.exp(y_train)
y_test_actual = np.exp(y_test)

# --- Evaluation Metrics ---
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
mae = mean_absolute_error(y_test_actual, y_test_pred)
r2 = r2_score(y_test_actual, y_test_pred)
mean_actual = np.mean(y_test_actual)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nXGBoost Test Metrics (with log-transform + expanded tuning):")
print(f"RMSE (original scale): {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE (original scale): {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R² (original scale): {r2:.4f}")

# --- Feature Importance ---
feature_importance = pd.Series(xgb_final.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(8,5))
plt.bar(feature_importance.index, feature_importance.values, color='teal')
plt.title("XGBoost Feature Importance (Expanded Search, Log-Transformed Target)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Residual Plot (log-scale) ---
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
x_min = min(y_train_pred_log.min(), y_test_pred_log.min())
x_max = max(y_train_pred_log.max(), y_test_pred_log.max())

plt.scatter(y_train_pred_log, y_train_pred_log - y_train, color="green", s=15, alpha=0.6, label='Train data')
plt.scatter(y_test_pred_log, y_test_pred_log - y_test, color="blue", s=15, alpha=0.6, label='Test data')
plt.hlines(y=0, xmin=x_min, xmax=x_max, colors='red', linewidth=2, linestyles='dashed')

plt.xlabel("Predicted log(Charges)", fontsize=12)
plt.ylabel("Residuals (Predicted - Actual)", fontsize=12)
plt.title("Residuals (XGBoost Regression with Log-Transformed Charges)", fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


##############################
######INTERACTION FEATURES####
##############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Encode categorical variables ---
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# --- Base features ---
X_base = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# --- Add interaction terms ---
X_base['age_smoker'] = df['age'] * df['smoker_binary']
X_base['bmi_smoker'] = df['bmi'] * df['smoker_binary']
X_base['age_bmi'] = df['age'] * df['bmi']
X_base['bmi_children'] = df['bmi'] * df['children']

X = X_base
y = np.log(df['charges'])

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# --- GridSearchCV setup ---
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1]
}

xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=1,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# --- Fit model ---
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE (log-scale):", abs(grid_search.best_score_))

# --- Final model ---
xgb_final = grid_search.best_estimator_
xgb_final.fit(X_train, y_train)

# --- Predictions ---
y_test_pred_log = xgb_final.predict(X_test)
y_test_pred = np.exp(y_test_pred_log)
y_test_actual = np.exp(y_test)

# --- Evaluation ---
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
mae = mean_absolute_error(y_test_actual, y_test_pred)
r2 = r2_score(y_test_actual, y_test_pred)
mean_actual = np.mean(y_test_actual)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nXGBoost with Interaction Effects (Log Transform):")
print(f"RMSE (original scale): {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE (original scale): {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R² (original scale): {r2:.4f}")

# --- Feature Importance ---
feature_importance = pd.Series(xgb_final.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(9,5))
plt.bar(feature_importance.index, feature_importance.values, color='darkcyan')
plt.title("XGBoost Feature Importance (with Interaction Terms)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
