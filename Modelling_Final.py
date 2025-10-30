import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

##########################################################
#################LINEAR REGRESSION########################
##########################################################

# Load the library for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

# One-hot encode region 
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate all features 
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# Take log of target variable
y = np.log(df['charges'])

# Add constant for intercept - OLS does not add an intercept automatically
X = sm.add_constant(X)

# Ensure all predictors are numeric
X = X.astype(float)

# Backward elimination function
def backward_elimination(X, y):
    features = list(X.columns)
    full_model = sm.OLS(y, X).fit()
    best_adj_r2 = full_model.rsquared_adj
    best_model = full_model  
    improved = True
    
    while improved and len(features) > 1:  # at least constant + 1 predictor
        improved = False
        models = []
        
        for f in features:
            if f == 'const':  # don’t remove the intercept
                continue
            trial_features = [feat for feat in features if feat != f]
            model = sm.OLS(y, X[trial_features]).fit()
            models.append((model.rsquared_adj, f, model))
        
        if not models:
            break

        best_trial_adj_r2, feature_removed, best_trial_model = max(models, key=lambda x: x[0])
        
        if best_trial_adj_r2 > best_adj_r2:
            best_adj_r2 = best_trial_adj_r2
            best_model = best_trial_model
            features.remove(feature_removed)
            improved = True
            print(f"Removed '{feature_removed}' | New adjusted R²: {best_adj_r2:.5f}")
    
    print("\nFinal Model Features:", features)
    return best_model

# Run backward elimination
best_model = backward_elimination(X, y)

# Summary of the final model
print(best_model.summary())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualising coefficients
print("Intercept:", model.intercept_)

coeff_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print("\nFeature Coefficients:\n", coeff_df)

coef_df_sorted = coeff_df.sort_values(by="Coefficient", ascending=False)

# Create plot with numeric labels
plt.figure(figsize=(10,6))
bars = plt.barh(coef_df_sorted["Feature"], coef_df_sorted["Coefficient"], color="skyblue")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance (Linear Regression Coefficients)")

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01 if width >= 0 else width - 0.01, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# Plotting Residual Errors
plt.style.use('fivethirtyeight')

plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10,
            label='Train data')

plt.scatter(model.predict(X_test),
            model.predict(X_test) - y_test,
            color="blue", s=10,
            label='Test data')

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual Errors")
plt.show()

# Model Evaluation Metrics
rmse_test_orig = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))
mae_test_orig = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
r2_test = r2_score(y_test, y_pred)  # R² in log scale

# Normalized error rates (% of mean actual charges)
mean_actual = np.mean(np.exp(y_test))
normalized_rmse = rmse_test_orig / mean_actual * 100
normalized_mae = mae_test_orig / mean_actual * 100

print("\nModel Evaluation Metrics (Test Set):")
print(f"Test RMSE (original scale): {rmse_test_orig:.2f}")
print(f"Test MAE (original scale): {mae_test_orig:.2f}")
print(f"Test R² (log scale): {r2_test:.4f}") #0.7605
print(f"Normalized RMSE: {normalized_rmse:.2f}%") #64.27%
print(f"Normalized MAE: {normalized_mae:.2f}%") #32.47%

############################################
# FAIRNESS ANALYSIS ON TEST SET
############################################

# Make predictions (using your existing model)
y_pred = model.predict(X_test)

# Convert back to original scale for fairness analysis
y_test_orig = np.exp(y_test)
y_pred_orig = np.exp(y_pred)

# Get test set indices to match with original dataframe
test_indices = X_test.index
df_test = df.loc[test_indices].copy()
df_test['predicted_charges'] = y_pred_orig
df_test['actual_charges'] = y_test_orig

# FAIRNESS ANALYSIS
print("=" * 60)
print("FAIRNESS ANALYSIS ON TEST SET")
print("=" * 60)

# 1. Gender Fairness Analysis
print("\n1. GENDER FAIRNESS ANALYSIS")
print("-" * 40)

male_actual = df_test[df_test['sex'] == 'male']['actual_charges']
female_actual = df_test[df_test['sex'] == 'female']['actual_charges']
male_pred = df_test[df_test['sex'] == 'male']['predicted_charges']
female_pred = df_test[df_test['sex'] == 'female']['predicted_charges']

# Statistical tests
from scipy import stats
t_stat_actual, p_actual = stats.ttest_ind(male_actual, female_actual, equal_var=False)
t_stat_pred, p_pred = stats.ttest_ind(male_pred, female_pred, equal_var=False)

print(f"Actual charges - Male vs Female:")
print(f"  Male mean: ${male_actual.mean():.2f}, Female mean: ${female_actual.mean():.2f}")
print(f"  T-test: t={t_stat_actual:.4f}, p={p_actual:.4f}")

print(f"Predicted charges - Male vs Female:")
print(f"  Male mean: ${male_pred.mean():.2f}, Female mean: ${female_pred.mean():.2f}")
print(f"  T-test: t={t_stat_pred:.4f}, p={p_pred:.4f}")

# Disparate Impact
disparate_impact_actual = female_actual.mean() / male_actual.mean()
disparate_impact_pred = female_pred.mean() / male_pred.mean()
print(f"Disparate Impact Ratio (Female/Male):")
print(f"  Actual: {disparate_impact_actual:.4f}")
print(f"  Predicted: {disparate_impact_pred:.4f}")

# 2. Region Fairness Analysis
print("\n2. REGION FAIRNESS ANALYSIS")
print("-" * 40)

region_actual_means = df_test.groupby('region')['actual_charges'].mean()
region_pred_means = df_test.groupby('region')['predicted_charges'].mean()

print("Average Actual Charges by Region:")
for region, mean_charge in region_actual_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

print("\nAverage Predicted Charges by Region:")
for region, mean_charge in region_pred_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

# ANOVA test for regional differences
regions = df_test['region'].unique()
region_groups_actual = [df_test[df_test['region'] == region]['actual_charges'] for region in regions]
region_groups_pred = [df_test[df_test['region'] == region]['predicted_charges'] for region in regions]

f_stat_actual, p_region_actual = stats.f_oneway(*region_groups_actual)
f_stat_pred, p_region_pred = stats.f_oneway(*region_groups_pred)

print(f"\nANOVA - Regional Differences:")
print(f"  Actual charges: F={f_stat_actual:.4f}, p={p_region_actual:.4f}")
print(f"  Predicted charges: F={f_stat_pred:.4f}, p={p_region_pred:.4f}")

# 3. Model Bias Analysis
print("\n3. MODEL BIAS ANALYSIS")
print("-" * 40)

# Calculate prediction errors by gender
male_error = male_pred.mean() - male_actual.mean()
female_error = female_pred.mean() - female_actual.mean()

print(f"Prediction Bias by Gender:")
print(f"  Male bias: ${male_error:.2f} ({male_error/male_actual.mean()*100:.2f}%)")
print(f"  Female bias: ${female_error:.2f} ({female_error/female_actual.mean()*100:.2f}%)")

# 4 Prediction Error Bar Plots
print("\n4. PREDICTION ERROR ANALYSIS")
print("-" * 40)

# Calculate prediction errors
df_test['prediction_error'] = df_test['actual_charges'] - df_test['predicted_charges']

# Create error comparison plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Prediction errors by gender
gender_errors = df_test.groupby('sex')['prediction_error'].mean()
gender_errors.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
axes[0].set_title('Average Prediction Error by Gender\n(Actual - Predicted)')
axes[0].set_ylabel('Prediction Error ($)')
axes[0].set_xlabel('Gender')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=0)

# Add value labels on bars
for i, v in enumerate(gender_errors):
    axes[0].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

# Plot 2: Prediction errors by region
region_errors = df_test.groupby('region')['prediction_error'].mean()
region_errors.plot(kind='bar', ax=axes[1], color=['lightgreen', 'lightyellow', 'lightblue', 'lightpink'])
axes[1].set_title('Average Prediction Error by Region\n(Actual - Predicted)')
axes[1].set_ylabel('Prediction Error ($)')
axes[1].set_xlabel('Region')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for i, v in enumerate(region_errors):
    axes[1].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.show()

# Print error statistics
print("Average Prediction Errors (Actual - Predicted):")
print(f"\nBy Gender:")
for gender, error in gender_errors.items():
    print(f"  {gender}: ${error:.2f}")

print(f"\nBy Region:")
for region, error in region_errors.items():
    print(f"  {region}: ${error:.2f}")


##########################################################
#################POLYNOMIAL REGRESSION####################
##########################################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

# One-hot encode region
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate all features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)

# Log-transform target variable
y = np.log(df['charges'])

# Polynomial Feature Transformation
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

# Get feature names for interpretation
poly_feature_names = poly.get_feature_names_out(X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=1)

# --- Train polynomial regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
r2 = r2_score(y_test, y_pred)
n, p = X_test.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Convert predictions back to original scale
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred)

# Compute main error metrics
rmse_test_orig = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
mae_test_orig = mean_absolute_error(y_test_exp, y_pred_exp)

# Normalized errors (% of mean actual charges)
mean_actual = np.mean(y_test_exp)
normalized_rmse = rmse_test_orig / mean_actual * 100
normalized_mae = mae_test_orig / mean_actual * 100

# Display Results
print(f"\nPolynomial Regression (degree={degree}) Evaluation Metrics:")
print(f"Test RMSE (original scale): {rmse_test_orig:.2f}")
print(f"Test MAE (original scale): {mae_test_orig:.2f}")
print(f"Test R² (log scale): {r2:.4f}")
print(f"Adjusted R² (log scale): {adj_r2:.4f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"Normalized MAE: {normalized_mae:.2f}%")

# Residual Error Plot
plt.style.use('fivethirtyeight')

plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

plt.scatter(model.predict(X_test),
            model.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linewidth=2)
plt.legend(loc='upper right')
plt.title(f"Residual Errors (Polynomial Regression, degree={degree})")
plt.show()

############################################
# FAIRNESS ANALYSIS ON TEST SET - POLYNOMIAL REGRESSION
############################################

# Convert predictions back to original scale
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred)

# FIX: Get test set indices properly
# Option 1: If y_test has index (most reliable)
if hasattr(y_test, 'index'):
    test_indices = y_test.index
    df_test = df.loc[test_indices].copy()
# Option 2: If no index available, create matching test set
else:
    # Create a test set that matches our split
    _, test_indices = train_test_split(df.index, test_size=0.4, random_state=1)
    df_test = df.loc[test_indices].copy()

df_test['predicted_charges'] = y_pred_exp
df_test['actual_charges'] = y_test_exp


# FAIRNESS ANALYSIS
print("=" * 60)
print("FAIRNESS ANALYSIS - POLYNOMIAL REGRESSION (degree={})".format(degree))
print("=" * 60)

# 1. Gender Fairness Analysis
print("\n1. GENDER FAIRNESS ANALYSIS")
print("-" * 40)

male_actual = df_test[df_test['sex'] == 'male']['actual_charges']
female_actual = df_test[df_test['sex'] == 'female']['actual_charges']
male_pred = df_test[df_test['sex'] == 'male']['predicted_charges']
female_pred = df_test[df_test['sex'] == 'female']['predicted_charges']

# Statistical tests
from scipy import stats
t_stat_actual, p_actual = stats.ttest_ind(male_actual, female_actual, equal_var=False)
t_stat_pred, p_pred = stats.ttest_ind(male_pred, female_pred, equal_var=False)

print(f"Actual charges - Male vs Female:")
print(f"  Male mean: ${male_actual.mean():.2f}, Female mean: ${female_actual.mean():.2f}")
print(f"  T-test: t={t_stat_actual:.4f}, p={p_actual:.4f}")

print(f"Predicted charges - Male vs Female:")
print(f"  Male mean: ${male_pred.mean():.2f}, Female mean: ${female_pred.mean():.2f}")
print(f"  T-test: t={t_stat_pred:.4f}, p={p_pred:.4f}")

# Disparate Impact
disparate_impact_actual = female_actual.mean() / male_actual.mean()
disparate_impact_pred = female_pred.mean() / male_pred.mean()
print(f"Disparate Impact Ratio (Female/Male):")
print(f"  Actual: {disparate_impact_actual:.4f}")
print(f"  Predicted: {disparate_impact_pred:.4f}")

# 2. Region Fairness Analysis
print("\n2. REGION FAIRNESS ANALYSIS")
print("-" * 40)

region_actual_means = df_test.groupby('region')['actual_charges'].mean()
region_pred_means = df_test.groupby('region')['predicted_charges'].mean()

print("Average Actual Charges by Region:")
for region, mean_charge in region_actual_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

print("\nAverage Predicted Charges by Region:")
for region, mean_charge in region_pred_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

# ANOVA test for regional differences
regions = df_test['region'].unique()
region_groups_actual = [df_test[df_test['region'] == region]['actual_charges'] for region in regions]
region_groups_pred = [df_test[df_test['region'] == region]['predicted_charges'] for region in regions]

f_stat_actual, p_region_actual = stats.f_oneway(*region_groups_actual)
f_stat_pred, p_region_pred = stats.f_oneway(*region_groups_pred)

print(f"\nANOVA - Regional Differences:")
print(f"  Actual charges: F={f_stat_actual:.4f}, p={p_region_actual:.4f}")
print(f"  Predicted charges: F={f_stat_pred:.4f}, p={p_region_pred:.4f}")

# 3. Prediction Error Analysis
print("\n3. PREDICTION ERROR ANALYSIS")
print("-" * 40)

# Calculate prediction errors
df_test['prediction_error'] = df_test['actual_charges'] - df_test['predicted_charges']

# Create error comparison plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Prediction errors by gender
gender_errors = df_test.groupby('sex')['prediction_error'].mean()
gender_errors.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
axes[0].set_title('Average Prediction Error by Gender\n(Actual - Predicted)')
axes[0].set_ylabel('Prediction Error ($)')
axes[0].set_xlabel('Gender')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=0)

# Add value labels on bars
for i, v in enumerate(gender_errors):
    axes[0].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

# Plot 2: Prediction errors by region
region_errors = df_test.groupby('region')['prediction_error'].mean()
region_errors.plot(kind='bar', ax=axes[1], color=['lightgreen', 'lightyellow', 'lightblue', 'lightpink'])
axes[1].set_title('Average Prediction Error by Region\n(Actual - Predicted)')
axes[1].set_ylabel('Prediction Error ($)')
axes[1].set_xlabel('Region')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for i, v in enumerate(region_errors):
    axes[1].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.show()

# Print error statistics
print("Average Prediction Errors (Actual - Predicted):")
print(f"\nBy Gender:")
for gender, error in gender_errors.items():
    print(f"  {gender}: ${error:.2f}")

print(f"\nBy Region:")
for region, error in region_errors.items():
    print(f"  {region}: ${error:.2f}")


# Residual Error Plot (your existing code)
plt.style.use('fivethirtyeight')
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, color="green", s=10, label='Train data')
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linewidth=2)
plt.legend(loc='upper right')
plt.title(f"Residual Errors (Polynomial Regression, degree={degree})")
plt.show()

##########################################################
#################K NEAREST NEIGHBOURS#####################
##########################################################

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Binary encode categorical variables
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

# One-hot encode region (drop_first=True to avoid dummy trap)
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Combine features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)
y = df['charges']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_neighbors': list(range(1, 40, 2)),  # odd numbers 1-39
    'weights': ['uniform', 'distance'],    # optional: uniform vs weighted
    'p': [1, 2]                            # Manhattan (1) or Euclidean (2)
}

grid = GridSearchCV(
    estimator=KNeighborsRegressor(),
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=10,
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

# Best hyperparameters
best_params = grid.best_params_
best_k = best_params['n_neighbors']
print("Best hyperparameters:", best_params)

# Final model evaluation
best_knn = grid.best_estimator_  
y_pred = best_knn.predict(X_test_scaled)

# Evaluate final model
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Normalized metrics (% of mean charge)
mean_y_test = np.mean(y_test)
nrmse = (rmse_test / mean_y_test) * 100
nmae = (mae_test / mean_y_test) * 100

# Error Metrics
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test MAE: {mae_test:.2f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Normalized RMSE: {nrmse:.2f}%")
print(f"Normalized MAE: {nmae:.2f}%")

############################################
# FAIRNESS ANALYSIS ON TEST SET - KNN
############################################

# Get test set indices to match with original dataframe
test_indices = X_test.index
df_test = df.loc[test_indices].copy()
df_test['predicted_charges'] = y_pred
df_test['actual_charges'] = y_test

# FAIRNESS ANALYSIS
print("=" * 60)
print("FAIRNESS ANALYSIS - K-NEAREST NEIGHBORS (k={})".format(best_k))
print("=" * 60)

# 1. Gender Fairness Analysis
print("\n1. GENDER FAIRNESS ANALYSIS")
print("-" * 40)

male_actual = df_test[df_test['sex'] == 'male']['actual_charges']
female_actual = df_test[df_test['sex'] == 'female']['actual_charges']
male_pred = df_test[df_test['sex'] == 'male']['predicted_charges']
female_pred = df_test[df_test['sex'] == 'female']['predicted_charges']

# Statistical tests
from scipy import stats
t_stat_actual, p_actual = stats.ttest_ind(male_actual, female_actual, equal_var=False)
t_stat_pred, p_pred = stats.ttest_ind(male_pred, female_pred, equal_var=False)

print(f"Actual charges - Male vs Female:")
print(f"  Male mean: ${male_actual.mean():.2f}, Female mean: ${female_actual.mean():.2f}")
print(f"  T-test: t={t_stat_actual:.4f}, p={p_actual:.4f}")

print(f"Predicted charges - Male vs Female:")
print(f"  Male mean: ${male_pred.mean():.2f}, Female mean: ${female_pred.mean():.2f}")
print(f"  T-test: t={t_stat_pred:.4f}, p={p_pred:.4f}")

# Disparate Impact
disparate_impact_actual = female_actual.mean() / male_actual.mean()
disparate_impact_pred = female_pred.mean() / male_pred.mean()
print(f"Disparate Impact Ratio (Female/Male):")
print(f"  Actual: {disparate_impact_actual:.4f}")
print(f"  Predicted: {disparate_impact_pred:.4f}")

# 2. Region Fairness Analysis
print("\n2. REGION FAIRNESS ANALYSIS")
print("-" * 40)

region_actual_means = df_test.groupby('region')['actual_charges'].mean()
region_pred_means = df_test.groupby('region')['predicted_charges'].mean()

print("Average Actual Charges by Region:")
for region, mean_charge in region_actual_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

print("\nAverage Predicted Charges by Region:")
for region, mean_charge in region_pred_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

# ANOVA test for regional differences
regions = df_test['region'].unique()
region_groups_actual = [df_test[df_test['region'] == region]['actual_charges'] for region in regions]
region_groups_pred = [df_test[df_test['region'] == region]['predicted_charges'] for region in regions]

f_stat_actual, p_region_actual = stats.f_oneway(*region_groups_actual)
f_stat_pred, p_region_pred = stats.f_oneway(*region_groups_pred)

print(f"\nANOVA - Regional Differences:")
print(f"  Actual charges: F={f_stat_actual:.4f}, p={p_region_actual:.4f}")
print(f"  Predicted charges: F={f_stat_pred:.4f}, p={p_region_pred:.4f}")

# 3. Prediction Error Analysis
print("\n3. PREDICTION ERROR ANALYSIS")
print("-" * 40)

# Calculate prediction errors
df_test['prediction_error'] = df_test['actual_charges'] - df_test['predicted_charges']

# Create error comparison plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Prediction errors by gender
gender_errors = df_test.groupby('sex')['prediction_error'].mean()
gender_errors.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
axes[0].set_title('Average Prediction Error by Gender\n(Actual - Predicted)')
axes[0].set_ylabel('Prediction Error ($)')
axes[0].set_xlabel('Gender')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=0)

# Add value labels on bars
for i, v in enumerate(gender_errors):
    axes[0].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

# Plot 2: Prediction errors by region
region_errors = df_test.groupby('region')['prediction_error'].mean()
region_errors.plot(kind='bar', ax=axes[1], color=['lightgreen', 'lightyellow', 'lightblue', 'lightpink'])
axes[1].set_title('Average Prediction Error by Region\n(Actual - Predicted)')
axes[1].set_ylabel('Prediction Error ($)')
axes[1].set_xlabel('Region')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for i, v in enumerate(region_errors):
    axes[1].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.show()

##########################################################
#################SUPPORT VECTOR MACHINE###################
##########################################################
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

# One-hot encode region
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate all features
X = pd.concat([df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']], region_dummies], axis=1)
y = df['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Log-transform target to reduce skew
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'C': [1, 10, 100, 500],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_log)

print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE (log-scale):", np.sqrt(-grid_search.best_score_))

# Train best model
best_svr = grid_search.best_estimator_
y_pred_log = best_svr.predict(X_test_scaled)

# Convert predictions back to original scale
y_pred_orig = np.exp(y_pred_log)

# Evaluation
# RMSE and MAE
rmse_test_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
mae_test_orig = mean_absolute_error(y_test, y_pred_orig)

# Normalized RMSE and MAE (divide by mean of actuals)
rmse_test_norm = rmse_test_orig / y_test.mean()
mae_test_norm = mae_test_orig / y_test.mean()

# R-squared
r2_test = r2_score(y_test, y_pred_orig)

print(f"Test RMSE (original scale): {rmse_test_orig:.2f}")
print(f"Test MAE (original scale): {mae_test_orig:.2f}")
print(f"Normalized RMSE: {rmse_test_norm:.2%}")
print(f"Normalized MAE: {mae_test_norm:.2%}")
print(f"R²: {r2_test:.4f}")

# Residual plot
plt.style.use('fivethirtyeight')
plt.scatter(y_pred_orig, y_pred_orig - y_test, color='blue', s=10, label='Test data')
plt.hlines(y=0, xmin=min(y_pred_orig), xmax=max(y_pred_orig), linewidth=2)
plt.xlabel("Predicted charges")
plt.ylabel("Residuals")
plt.title("Residuals (SVR with RBF Kernel)")
plt.legend(loc='upper right')
plt.show()

############################################
# FAIRNESS ANALYSIS ON TEST SET - SVM ######
############################################

# Get test set indices to match with original dataframe
test_indices = X_test.index
df_test = df.loc[test_indices].copy()
df_test['predicted_charges'] = y_pred_orig
df_test['actual_charges'] = y_test

# FAIRNESS ANALYSIS
print("=" * 60)
print("FAIRNESS ANALYSIS - SUPPORT VECTOR REGRESSION")
print("=" * 60)

# 1. Gender Fairness Analysis
print("\n1. GENDER FAIRNESS ANALYSIS")
print("-" * 40)

male_actual = df_test[df_test['sex'] == 'male']['actual_charges']
female_actual = df_test[df_test['sex'] == 'female']['actual_charges']
male_pred = df_test[df_test['sex'] == 'male']['predicted_charges']
female_pred = df_test[df_test['sex'] == 'female']['predicted_charges']

# Statistical tests
from scipy import stats
t_stat_actual, p_actual = stats.ttest_ind(male_actual, female_actual, equal_var=False)
t_stat_pred, p_pred = stats.ttest_ind(male_pred, female_pred, equal_var=False)

print(f"Actual charges - Male vs Female:")
print(f"  Male mean: ${male_actual.mean():.2f}, Female mean: ${female_actual.mean():.2f}")
print(f"  T-test: t={t_stat_actual:.4f}, p={p_actual:.4f}")

print(f"Predicted charges - Male vs Female:")
print(f"  Male mean: ${male_pred.mean():.2f}, Female mean: ${female_pred.mean():.2f}")
print(f"  T-test: t={t_stat_pred:.4f}, p={p_pred:.4f}")

# Disparate Impact
disparate_impact_actual = female_actual.mean() / male_actual.mean()
disparate_impact_pred = female_pred.mean() / male_pred.mean()
print(f"Disparate Impact Ratio (Female/Male):")
print(f"  Actual: {disparate_impact_actual:.4f}")
print(f"  Predicted: {disparate_impact_pred:.4f}")

# 2. Region Fairness Analysis
print("\n2. REGION FAIRNESS ANALYSIS")
print("-" * 40)

region_actual_means = df_test.groupby('region')['actual_charges'].mean()
region_pred_means = df_test.groupby('region')['predicted_charges'].mean()

print("Average Actual Charges by Region:")
for region, mean_charge in region_actual_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

print("\nAverage Predicted Charges by Region:")
for region, mean_charge in region_pred_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

# ANOVA test for regional differences
regions = df_test['region'].unique()
region_groups_actual = [df_test[df_test['region'] == region]['actual_charges'] for region in regions]
region_groups_pred = [df_test[df_test['region'] == region]['predicted_charges'] for region in regions]

f_stat_actual, p_region_actual = stats.f_oneway(*region_groups_actual)
f_stat_pred, p_region_pred = stats.f_oneway(*region_groups_pred)

print(f"\nANOVA - Regional Differences:")
print(f"  Actual charges: F={f_stat_actual:.4f}, p={p_region_actual:.4f}")
print(f"  Predicted charges: F={f_stat_pred:.4f}, p={p_region_pred:.4f}")

# 3. Prediction Error Analysis
print("\n3. PREDICTION ERROR ANALYSIS")
print("-" * 40)

# Calculate prediction errors
df_test['prediction_error'] = df_test['actual_charges'] - df_test['predicted_charges']

# Create error comparison plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Prediction errors by gender
gender_errors = df_test.groupby('sex')['prediction_error'].mean()
gender_errors.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
axes[0].set_title('Average Prediction Error by Gender\n(Actual - Predicted)')
axes[0].set_ylabel('Prediction Error ($)')
axes[0].set_xlabel('Gender')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=0)

# Add value labels on bars
for i, v in enumerate(gender_errors):
    axes[0].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

# Plot 2: Prediction errors by region
region_errors = df_test.groupby('region')['prediction_error'].mean()
region_errors.plot(kind='bar', ax=axes[1], color=['lightgreen', 'lightyellow', 'lightblue', 'lightpink'])
axes[1].set_title('Average Prediction Error by Region\n(Actual - Predicted)')
axes[1].set_ylabel('Prediction Error ($)')
axes[1].set_xlabel('Region')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for i, v in enumerate(region_errors):
    axes[1].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.show()

# Print error statistics
print("Average Prediction Errors (Actual - Predicted):")
print(f"\nBy Gender:")
for gender, error in gender_errors.items():
    print(f"  {gender}: ${error:.2f}")

print(f"\nBy Region:")
for region, error in region_errors.items():
    print(f"  {region}: ${error:.2f}")

##############################
######DECISION TREE###########
##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Encode categorical variables
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)
y = df['charges']

# 10-fold CV to select max_depth
depth_values = list(range(1, 21))
cv_rmse = []

for d in depth_values:
    dt = DecisionTreeRegressor(max_depth=d, random_state=1)
    scores = cross_val_score(dt, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    cv_rmse.append(rmse_scores.mean())
    print(f"max_depth={d} | 10-fold CV RMSE: {rmse_scores.mean():.2f}")

# Optimal depth
best_depth = depth_values[np.argmin(cv_rmse)]
print(f"\nOptimal max_depth (10-fold CV): {best_depth}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train best Decision Tree
dt_reg = DecisionTreeRegressor(max_depth=best_depth, random_state=1)
dt_reg.fit(X_train, y_train)
y_train_pred = dt_reg.predict(X_train)
y_test_pred = dt_reg.predict(X_test)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mean_actual = np.mean(y_test)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nDecision Tree Regression Metrics (Test Set):")
print(f"RMSE: {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R²: {r2:.4f}")

# Plot regression tree
for i in range(dt_reg.tree_.node_count):
    dt_reg.tree_.value[i] = np.mean(dt_reg.tree_.value[i])

plt.figure(figsize=(20,10))
plot_tree(dt_reg, feature_names=X.columns, rounded=True, filled=True, max_depth=2)
plt.title("Decision Tree Regressor: Predicting Charges")
plt.show()

# Residual Plot (original scale)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
x_min = min(y_train_pred.min(), y_test_pred.min())
x_max = max(y_train_pred.max(), y_test_pred.max())

plt.scatter(y_train_pred, y_train_pred - y_train, color="green", s=15, alpha=0.6, label='Train data')
plt.scatter(y_test_pred, y_test_pred - y_test, color="blue", s=15, alpha=0.6, label='Test data')
plt.hlines(y=0, xmin=x_min, xmax=x_max, colors='red', linewidth=2, linestyles='dashed')

plt.xlabel("Predicted Charges (original scale)", fontsize=12)
plt.ylabel("Residuals (Predicted - Actual)", fontsize=12)
plt.title("Residuals (Decision Tree Regression)", fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

############################################
# FAIRNESS ANALYSIS ON TEST SET - SVM ######
############################################

# Get test set indices to match with original dataframe
test_indices = X_test.index
df_test = df.loc[test_indices].copy()
df_test['predicted_charges'] = y_pred_orig
df_test['actual_charges'] = y_test

# FAIRNESS ANALYSIS
print("=" * 60)
print("FAIRNESS ANALYSIS - SUPPORT VECTOR REGRESSION")
print("=" * 60)

# 1. Gender Fairness Analysis
print("\n1. GENDER FAIRNESS ANALYSIS")
print("-" * 40)

male_actual = df_test[df_test['sex'] == 'male']['actual_charges']
female_actual = df_test[df_test['sex'] == 'female']['actual_charges']
male_pred = df_test[df_test['sex'] == 'male']['predicted_charges']
female_pred = df_test[df_test['sex'] == 'female']['predicted_charges']

# Statistical tests
from scipy import stats
t_stat_actual, p_actual = stats.ttest_ind(male_actual, female_actual, equal_var=False)
t_stat_pred, p_pred = stats.ttest_ind(male_pred, female_pred, equal_var=False)

print(f"Actual charges - Male vs Female:")
print(f"  Male mean: ${male_actual.mean():.2f}, Female mean: ${female_actual.mean():.2f}")
print(f"  T-test: t={t_stat_actual:.4f}, p={p_actual:.4f}")

print(f"Predicted charges - Male vs Female:")
print(f"  Male mean: ${male_pred.mean():.2f}, Female mean: ${female_pred.mean():.2f}")
print(f"  T-test: t={t_stat_pred:.4f}, p={p_pred:.4f}")

# Disparate Impact
disparate_impact_actual = female_actual.mean() / male_actual.mean()
disparate_impact_pred = female_pred.mean() / male_pred.mean()
print(f"Disparate Impact Ratio (Female/Male):")
print(f"  Actual: {disparate_impact_actual:.4f}")
print(f"  Predicted: {disparate_impact_pred:.4f}")

# 2. Region Fairness Analysis
print("\n2. REGION FAIRNESS ANALYSIS")
print("-" * 40)

region_actual_means = df_test.groupby('region')['actual_charges'].mean()
region_pred_means = df_test.groupby('region')['predicted_charges'].mean()

print("Average Actual Charges by Region:")
for region, mean_charge in region_actual_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

print("\nAverage Predicted Charges by Region:")
for region, mean_charge in region_pred_means.items():
    print(f"  {region}: ${mean_charge:.2f}")

# ANOVA test for regional differences
regions = df_test['region'].unique()
region_groups_actual = [df_test[df_test['region'] == region]['actual_charges'] for region in regions]
region_groups_pred = [df_test[df_test['region'] == region]['predicted_charges'] for region in regions]

f_stat_actual, p_region_actual = stats.f_oneway(*region_groups_actual)
f_stat_pred, p_region_pred = stats.f_oneway(*region_groups_pred)

print(f"\nANOVA - Regional Differences:")
print(f"  Actual charges: F={f_stat_actual:.4f}, p={p_region_actual:.4f}")
print(f"  Predicted charges: F={f_stat_pred:.4f}, p={p_region_pred:.4f}")

# 3. Prediction Error Analysis
print("\n3. PREDICTION ERROR ANALYSIS")
print("-" * 40)

# Calculate prediction errors
df_test['prediction_error'] = df_test['actual_charges'] - df_test['predicted_charges']

# Create error comparison plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Prediction errors by gender
gender_errors = df_test.groupby('sex')['prediction_error'].mean()
gender_errors.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
axes[0].set_title('Average Prediction Error by Gender\n(Actual - Predicted)')
axes[0].set_ylabel('Prediction Error ($)')
axes[0].set_xlabel('Gender')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=0)

# Add value labels on bars
for i, v in enumerate(gender_errors):
    axes[0].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

# Plot 2: Prediction errors by region
region_errors = df_test.groupby('region')['prediction_error'].mean()
region_errors.plot(kind='bar', ax=axes[1], color=['lightgreen', 'lightyellow', 'lightblue', 'lightpink'])
axes[1].set_title('Average Prediction Error by Region\n(Actual - Predicted)')
axes[1].set_ylabel('Prediction Error ($)')
axes[1].set_xlabel('Region')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for i, v in enumerate(region_errors):
    axes[1].text(i, v + (10 if v >= 0 else -30), f'${v:.0f}', 
                ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.show()

############################
####### RANDOM FOREST ######
############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Encode categorical variables
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)
y = df['charges']

# Range of depths to test
depth_values = list(range(2, 21)) + [None]
cv_rmse = []

for d in depth_values:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=d,
        max_samples=0.5,
        min_samples_leaf=5,
        random_state=1,
        n_jobs=-1
    )
    # 10-fold CV with negative MSE
    scores = cross_val_score(
        model, X, y,
        cv=10,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse_scores = np.sqrt(-scores)  # convert to RMSE
    cv_rmse.append(rmse_scores.mean())
    print(f"max_depth={d} | 10-fold CV RMSE: {rmse_scores.mean():.2f}")

# Find best max_depth
best_depth = depth_values[np.argmin(cv_rmse)]
print(f"\nOptimal max_depth (10-fold CV): {best_depth}")

# Train final model with best depth
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

rf_final = RandomForestRegressor(
    n_estimators=200,
    max_depth=best_depth,
    max_samples=0.5,
    min_samples_leaf=5,
    random_state=1,
    n_jobs=-1
)
rf_final.fit(X_train, y_train)
y_test_pred = rf_final.predict(X_test)

# Evaluation Metrics
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mean_actual = np.mean(y_test)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nRandom Forest Metrics (Test Set):")
print(f"RMSE: {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R²: {r2:.4f}")

# Feature Importance
fi = pd.Series(rf_final.feature_importances_, index=X.columns)
fi.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(10,6))
bars = plt.bar(fi.index, fi.values, color='skyblue')
plt.title("Random Forest Regressor Feature Importance", fontsize=14)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

############################
########### XG BOOST #######
############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate features
X = pd.concat([
    df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
    region_dummies
], axis=1)
y = df['charges']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# GridSearchCV with 10-fold CV
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
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))

# Train final model on full training set
xgb_final = grid_search.best_estimator_
y_train_pred = xgb_final.predict(X_train)
y_test_pred = xgb_final.predict(X_test)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mean_actual = np.mean(y_test)
normalized_rmse = rmse / mean_actual * 100
normalized_mae = mae / mean_actual * 100

print("\nXGBoost Test Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"Normalized RMSE: {normalized_rmse:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"Normalized MAE: {normalized_mae:.2f}%")
print(f"R²: {r2:.4f}")

# Feature Importance Plot
feature_importance = pd.Series(xgb_final.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(10,6))
bars = plt.bar(feature_importance.index, feature_importance.values, color='teal')
plt.title("XGBoost Regressor Feature Importance", fontsize=14)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Residual Plot (original scale)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
x_min = min(y_train_pred.min(), y_test_pred.min())
x_max = max(y_train_pred.max(), y_test_pred.max())

plt.scatter(y_train_pred, y_train_pred - y_train, color="green", s=15, alpha=0.6, label='Train data')
plt.scatter(y_test_pred, y_test_pred - y_test, color="blue", s=15, alpha=0.6, label='Test data')
plt.hlines(y=0, xmin=x_min, xmax=x_max, colors='red', linewidth=2, linestyles='dashed')

plt.xlabel("Predicted Charges (original scale)", fontsize=12)
plt.ylabel("Residuals (Predicted - Actual)", fontsize=12)
plt.title("Residuals (XGBoost Regression)", fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
