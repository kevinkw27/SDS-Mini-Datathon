import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# Removes duplicates and keeps unique rows
df_clean = df.drop_duplicates().copy()  # creates a new dataframe without the duplicates
df_clean.info()

##########################################################
#################LINEAR REGRESSION########################
##########################################################

# Load the library for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

#Label encoding for categorical predictors
label_encoder = LabelEncoder()
df['sex_encoded'] = label_encoder.fit_transform(df['sex'])
df['smoker_encoded'] = label_encoder.fit_transform(df['smoker'])
df['region_encoded'] = label_encoder.fit_transform(df['sex'])


X = df[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']]
y = np.log(df['charges'])

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)


#Visualising coefficients
print("Intercept:", model.intercept_)

# Get actual feature names after preprocessing
X_raw = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
feature_names = X_raw.columns

coeff_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
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

"""
# variance score: Model explains 76% of variance
print('Variance score: {}'.format(model.score(X_test, y_test)))

# Calculate and print errors.
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")

rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
print(f"Root mean squared error: {rmse:.4f}")
"""

##########################################################
#################POLYNOMIAL REGRESSION####################
##########################################################
https://www.geeksforgeeks.org/machine-learning/linear-regression-python-implementation/


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