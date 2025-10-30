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

# =============================================================================
# FAIRNESS ANALYSIS FOR RANDOM FOREST - GENDER & REGION ONLY
# =============================================================================

# Get Random Forest predictions on test set (using your trained model)
rf_test_predictions = best_rf.predict(X_test)

# Convert predictions back to original scale if needed (since you used log-transform)
if 'y_pred_log' in locals():
    rf_test_predictions_original = np.exp(rf_test_predictions)
    y_test_original = np.exp(y_test)
else:
    rf_test_predictions_original = rf_test_predictions
    y_test_original = y_test

# Add predictions and actual values to test set for fairness analysis
test_set_rf = X_test.copy()
test_set_rf['actual_charges'] = y_test_original
test_set_rf['predicted_charges'] = rf_test_predictions_original
test_set_rf['prediction_error'] = test_set_rf['predicted_charges'] - test_set_rf['actual_charges']
test_set_rf['absolute_error'] = np.abs(test_set_rf['prediction_error'])
test_set_rf['relative_error'] = (test_set_rf['prediction_error'] / test_set_rf['actual_charges']) * 100

# 1. Gender Fairness Analysis
print("\n1. GENDER FAIRNESS ANALYSIS")
print("-" * 30)

male_errors = test_set_rf[test_set_rf['sex_binary'] == 1]['prediction_error']
female_errors = test_set_rf[test_set_rf['sex_binary'] == 0]['prediction_error']

# T-test for gender differences
from scipy import stats
t_stat_gender, p_value_gender = stats.ttest_ind(male_errors, female_errors, equal_var=False)

print(f"Male count: {len(male_errors)}")
print(f"Female count: {len(female_errors)}")
print(f"Mean error - Male: ${male_errors.mean():.2f}")
print(f"Mean error - Female: ${female_errors.mean():.2f}")
print(f"Std error - Male: ${male_errors.std():.2f}")
print(f"Std error - Female: ${female_errors.std():.2f}")
print(f"T-statistic: {t_stat_gender:.4f}")
print(f"P-value: {p_value_gender:.4f}")

# Disparate impact analysis
male_mean_pred = test_set_rf[test_set_rf['sex_binary'] == 1]['predicted_charges'].mean()
female_mean_pred = test_set_rf[test_set_rf['sex_binary'] == 0]['predicted_charges'].mean()
male_mean_actual = test_set_rf[test_set_rf['sex_binary'] == 1]['actual_charges'].mean()
female_mean_actual = test_set_rf[test_set_rf['sex_binary'] == 0]['actual_charges'].mean()

disparate_impact_gender = female_mean_pred / male_mean_pred
actual_ratio_gender = female_mean_actual / male_mean_actual

print(f"\nDisparate Impact (Female/Male predictions): {disparate_impact_gender:.4f}")
print(f"Actual Ratio (Female/Male actual): {actual_ratio_gender:.4f}")
print(f"Bias Ratio (Predicted/Actual): {disparate_impact_gender/actual_ratio_gender:.4f}")

# 2. Regional Fairness Analysis
print("\n2. REGIONAL FAIRNESS ANALYSIS")
print("-" * 30)

# Get region columns
region_cols = [col for col in test_set_rf.columns if col.startswith('region_')]
region_errors = []

print("Regional Error Analysis:")
for region in region_cols:
    region_data = test_set_rf[test_set_rf[region] == 1]
    if len(region_data) > 0:
        mean_error = region_data['prediction_error'].mean()
        mean_absolute_err = region_data['absolute_error'].mean()
        region_errors.append((region, mean_error, len(region_data)))
        print(f"{region}: Mean error = ${mean_error:>8.2f}, MAE = ${mean_absolute_err:>8.2f} (n={len(region_data)})")

# ANOVA test for regional differences
region_groups = []
region_names = []
for region in region_cols:
    region_errors_data = test_set_rf[test_set_rf[region] == 1]['prediction_error']
    if len(region_errors_data) > 0:
        region_groups.append(region_errors_data)
        region_names.append(region.replace('region_', ''))

if len(region_groups) >= 2:
    f_stat_region, p_value_region = stats.f_oneway(*region_groups)
    print(f"\nANOVA F-statistic: {f_stat_region:.4f}")
    print(f"ANOVA P-value: {p_value_region:.4f}")
else:
    print("\nInsufficient regional groups for ANOVA")

# 3. Visualization - Gender Differences
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
gender_means = [male_errors.mean(), female_errors.mean()]
gender_labels = ['Male', 'Female']
colors = ['blue', 'pink']
bars = plt.bar(gender_labels, gender_means, color=colors, alpha=0.7)
plt.title('Random Forest: Mean Prediction Error by Gender')
plt.ylabel('Mean Error ($)')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(gender_means):
    plt.text(i, v, f'${v:.1f}', ha='center', va='bottom', fontweight='bold')

plt.subplot(1, 3, 2)
gender_abs_means = [np.abs(male_errors).mean(), np.abs(female_errors).mean()]
bars = plt.bar(gender_labels, gender_abs_means, color=colors, alpha=0.7)
plt.title('Random Forest: Mean Absolute Error by Gender')
plt.ylabel('Mean Absolute Error ($)')
plt.grid(True, alpha=0.3)

for i, v in enumerate(gender_abs_means):
    plt.text(i, v, f'${v:.1f}', ha='center', va='bottom', fontweight='bold')

plt.subplot(1, 3, 3)
gender_relative_means = [
    test_set_rf[test_set_rf['sex_binary'] == 1]['relative_error'].mean(),
    test_set_rf[test_set_rf['sex_binary'] == 0]['relative_error'].mean()
]
bars = plt.bar(gender_labels, gender_relative_means, color=colors, alpha=0.7)
plt.title('Random Forest: Mean Relative Error by Gender')
plt.ylabel('Mean Relative Error (%)')
plt.grid(True, alpha=0.3)

for i, v in enumerate(gender_relative_means):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 4. Visualization - Regional Differences
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
region_error_means = [test_set_rf[test_set_rf[col] == 1]['prediction_error'].mean() for col in region_cols]
region_display_names = [col.replace('region_', '').title() for col in region_cols]

bars = plt.bar(region_display_names, region_error_means, color='lightgreen', alpha=0.7)
plt.title('Random Forest: Mean Prediction Error by Region')
plt.ylabel('Mean Error ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

for i, v in enumerate(region_error_means):
    plt.text(i, v, f'${v:.1f}', ha='center', va='bottom', fontsize=9)

plt.subplot(1, 2, 2)
region_abs_means = [test_set_rf[test_set_rf[col] == 1]['absolute_error'].mean() for col in region_cols]
bars = plt.bar(region_display_names, region_abs_means, color='lightcoral', alpha=0.7)
plt.title('Random Forest: Mean Absolute Error by Region')
plt.ylabel('Mean Absolute Error ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

for i, v in enumerate(region_abs_means):
    plt.text(i, v, f'${v:.1f}', ha='center', va='bottom', fontsize=9)

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
