import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# Removes duplicates and keeps unique rows
df_clean = df.drop_duplicates().copy()  # creates a new dataframe without the duplicates
df_clean.info()


##########################################################
#################FEATURE SELECTION########################
##########################################################


"""
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Keep only numeric columns for correlation and VIF
independent = df_clean.select_dtypes(include=[np.number]).copy()

# Compute correlation matrix
corr_matrix = independent.corr()
print(corr_matrix)

# Identify pairs of features with high collinearity (correlation > 0.8 or < -0.8).
high_corr_features = []
for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.7:
            high_corr_features.append((col1, col2, corr_matrix.loc[col1, col2]))

collinearity_df = pd.DataFrame(high_corr_features, columns=["Feature 1", "Feature 2", "Correlation"]).drop_duplicates()
print("\nHighly Correlated Features:\n", collinearity_df)

# Drop dependent variable if present
if 'charges' in independent.columns:
    independent = independent.drop(columns=['charges'])

# Compute Variance Inflation Factor (VIF) for each feature.
vif_data = pd.DataFrame()
vif_data["Feature"] = independent.columns
vif_data["VIF"] = [variance_inflation_factor(independent.values, i)
                   for i in range(independent.shape[1])]

print("\nVariance Inflation Factor (VIF) for each feature:\n", vif_data)

"""

##########################################################
#################LINEAR REGRESSION########################
##########################################################

"""
# Load the library for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

#Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

#One-hot encode region 
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

def backward_elimination(X, y):
    features = list(X.columns)
    
    full_model = sm.OLS(y, X).fit()
    best_adj_r2 = full_model.rsquared_adj
    best_model = full_model  
    improved = True
    
    while improved and len(features) > 1:  # at least constant + 1 predictor
        improved = False
        models = []
        
        # Try removing each feature once
        for f in features:
            if f == 'const':  # don’t remove the intercept
                continue
            trial_features = [feat for feat in features if feat != f]
            model = sm.OLS(y, X[trial_features]).fit()
            models.append((model.rsquared_adj, f, model))
        
        # Find the best adjusted R² after removing one feature
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Visualising coefficients
print("Intercept:", model.intercept_)


coeff_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print("\nFeature Coefficients:\n", coeff_df)

coef_df_sorted = coeff_df.sort_values(by="Coefficient", ascending=False)

# Create plot.
plt.figure(figsize=(8,6))
plt.barh(coef_df_sorted["Feature"], coef_df_sorted["Coefficient"], color="blue")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.show()


#### Plotting Residual Errors
# plot for residual error

# setting plot style
plt.style.use('fivethirtyeight')

# plotting residual errors in training data
plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10,
            label='Train data')

# plotting residual errors in test data
plt.scatter(model.predict(X_test),
            model.predict(X_test) - y_test,
            color="blue", s=10,
            label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# plotting legend
plt.legend(loc='upper right')

# plot title
plt.title("Residual errors")

# method call for showing the plot
plt.show()

# variance score: Model explains 76% of variance
print('Variance score: {}'.format(model.score(X_test, y_test)))

# RMSE: 8562.02
rmse_test_orig = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))
print("Test RMSE (original scale):", rmse_test_orig)
"""

##########################################################
#################POLYNOMIAL REGRESSION####################
##########################################################

"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

#One-hot encode region 
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate all features 
X = pd.concat([
df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
region_dummies
], axis=1)

y = np.log(df['charges'])

# --- Polynomial Feature Transformation ---
# Choose polynomial degree (try degree= 2 or 3)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Get feature names for later interpretation
poly_feature_names = poly.get_feature_names_out(X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.4, random_state=1)

# Train polynomial regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compute performance metrics
r2 = r2_score(y_test, y_pred)
n, p = X_test.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmse_test_orig = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))

# RMSE degree 2 - 5395, degree 3 - 5281, degree 4 - 34852
print("Test RMSE (original scale):", rmse_test_orig)

# --- Residual Error Plot ---
plt.style.use('fivethirtyeight')

plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

plt.scatter(model.predict(X_test),
            model.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual Errors (Polynomial Regression)")
plt.show()
"""

##########################################################
#################K NEAREST NEIGHBOURS#####################
##########################################################

"""
#Import libraries for KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

#Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

#One-hot encode region 
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate all features 
X = pd.concat([
df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
region_dummies
], axis=1)
y = df['charges']

#Split dataset for train and test set
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Scale features to have mean=0 and std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define odd k values from 1 to 79
k_values = list(range(1, 79, 2))

train_rmse = []
test_rmse = []

# Loop through each k value
for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = knn_model.predict(X_train_scaled)
    y_test_pred = knn_model.predict(X_test_scaled)
    
    # Evaluation metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_rmse.append(rmse_train)
    test_rmse.append(rmse_test)

# Find best k based on test precision
best_k = k_values[np.argmin(test_rmse)]

# Plot
plt.figure(figsize=(9, 5))
plt.plot(k_values, train_rmse, marker='o', color='blue', label='Train RMSE')
plt.plot(k_values, test_rmse, marker='o', color='teal', linestyle='--', label='Test RMSE')
plt.axvline(best_k, color='red', linestyle=':', label=f'Best k (RMSE) = {best_k}')
plt.title("KNN Regression - RMSE vs Number of Neighbors (k)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Best k (based on RMSE): {best_k}")

# Create KNN classifier using best k
knn_model = KNeighborsRegressor(n_neighbors=3)

# Train the model
knn_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = knn_model.predict(X_test_scaled)

# Evaluate the model
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5

#RMSE 5544
print(f'Mean Squared Error: {rmse}')
"""

##########################################################
#################SUPPORT VECTOR MACHINES##################
##########################################################

"""
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Binary encode sex and smoker
df['sex_binary'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker_binary'] = df['smoker'].map({'yes': 1, 'no': 0})

#One-hot encode region 
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Concatenate all features 
X = pd.concat([
df[['age', 'bmi', 'children', 'sex_binary', 'smoker_binary']],
region_dummies
], axis=1)
y = df['charges']

#Split dataset for train and test set
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Log-transform target to reduce skew ---
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Scale features to have mean=0 and std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Hyperparameter tuning using GridSearchCV ---
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

# --- Train best model ---
best_svr = grid_search.best_estimator_
y_pred_log = best_svr.predict(X_test_scaled)

# --- Convert predictions back to original scale ---
y_pred_orig = np.exp(y_pred_log)

# --- Evaluation ---
# RMSE - 4927
rmse_test_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
print("Test RMSE (original scale):", rmse_test_orig)

# --- Residual plot ---
plt.style.use('fivethirtyeight')
plt.scatter(y_pred_orig, y_pred_orig - y_test, color='blue', s=10, label='Test data')
plt.hlines(y=0, xmin=min(y_pred_orig), xmax=max(y_pred_orig), linewidth=2)
plt.xlabel("Predicted charges")
plt.ylabel("Residuals")
plt.title("Residuals (SVR with RBF Kernel)")
plt.legend(loc='upper right')
plt.show()
"""
