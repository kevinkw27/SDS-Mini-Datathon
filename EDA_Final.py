##############################################
#### DATA CLEANING############################
##############################################

# Import libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Import dataset
df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# Descriptive statistics
df.describe(include="all")

# No. of unique elements in each column
df.nunique()

# Check unique elements for all columns
# While there seems to be outliers, there are no erronous/unrealistic values.
"""
for col in df.columns:
  print(col)
  print(df[col].unique())
  print()
"""

# Counts the missing values
# No missing values in the dataset.
df.isna().sum()

# Shows the duplicated rows
# There is 1 duplicated row. However, it is reasonable to assume that these 2 people are different since an area has many people.
df[df.duplicated()]

# Concise summary of dataframe
df_clean = df.copy()
"""
df_clean.info()
for col in df_clean.columns:
  print(col)
  print(df_clean[col].unique())
  print()
"""

##############################################
######## ANALYSIS OF DEPENDENT VARIABLE ######
##############################################
from scipy.stats import skew, kurtosis

print(df_clean.describe())
print(f"Skewness: {skew(df_clean['charges']):.2f}")
print(f"Kurtosis: {kurtosis(df_clean['charges']):.2f}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create log-transformed variable
df_clean['log_charges'] = np.log(df_clean['charges'])

# Compare original vs log-scaled
plt.figure(figsize=(10, 5))

# Original
plt.subplot(1, 2, 1)
sns.histplot(df_clean['charges'], bins=30, kde=True, color='teal')
plt.title('Original Charges Distribution')
plt.xlabel('Charges ($)')
plt.ylabel('Count')

# Log-transformed
plt.subplot(1, 2, 2)
sns.histplot(df_clean['log_charges'], bins=30, kde=True, color='coral')
plt.title('Log-Scaled Charges Distribution')
plt.xlabel('log(Charges)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

#############################
####CORRELATION MATRIX#######
#############################

# Encode binary variables first
df_clean['smoker_binary'] = df_clean['smoker'].map({'no': 0, 'yes': 1})
df_clean['sex_binary'] = df_clean['sex'].map({'female': 0, 'male': 1})

# One-hot encode region
region_dummies = pd.get_dummies(df_clean['region'], prefix='region', dtype=int)

# Combine everything
df_combined = pd.concat([df_clean, region_dummies], axis=1)

# Select numeric columns
numeric_cols = df_combined.select_dtypes(include=np.number).columns

# Correlation
correlation_matrix = df_combined[numeric_cols].corr()

# Display
print("Combined Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


#####################################
#ANALYSIS OF SIGNIFICANT PREDICTORS##
#########(AGE, BMI, SMOKING)#########
#####################################


#####################################
#####SMOKING VS CHARGES##############
#####################################
# Boxplot showing smoker vs charges -- Smoker significantly higher median than non-smoker
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.boxplot(x="smoker", y="charges", data=df_clean, palette="Set2", hue="smoker", legend=False)
plt.title("Charges by Smoking")
plt.xlabel("Smoking")
plt.ylabel("Charges")

medians = df_clean.groupby("smoker")["charges"].median()
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    category = text.get_text()
    median_val = medians[category]
    
    ax.text(tick, median_val, f'{median_val:.0f}', 
            ha='center', va='bottom', fontweight='bold', color='black')
plt.show()

#####################################
######### AGE VS CHARGES#############
#####################################

# Histogram of insured ages
plt.figure(figsize=(6,4))
sns.histplot(df_clean["age"], bins = 20 , kde = True)
#kde=Kernel Density Estimation=smooth continuous line (pdf of the data) on the histogram
plt.title("Distribution of Insured Ages")
plt.xlabel("age")
plt.ylabel("Count")
plt.show()

#####################################
######### BMI VS CHARGES#############
#####################################

#Histogram of insured BMI
plt.figure(figsize=(6,4))
sns.histplot(df_clean["bmi"], bins = 20 , kde = True)
#kde=Kernel Density Estimation=smooth continuous line (pdf of the data) on the histogram
plt.title("Distribution of BMI")
plt.xlabel("bmi")
plt.ylabel("Count")
plt.show()

#Approximately normal distribution
from scipy.stats import skew, kurtosis
print('Skewness:', skew(df_clean['bmi']))
print('Kurtosis:', kurtosis(df_clean['bmi']))

# Boxplot of BMI categories based on WHO's classifications
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    elif 30 <= bmi < 35:
        return 'Obesity Class I'
    elif 35 <= bmi < 40:
        return 'Obesity Class II'
    else:
        return 'Obesity Class III'
df_clean['bmi_class'] = df_clean['bmi'].apply(bmi_category)
category_counts = df_clean['bmi_class'].value_counts()
print(category_counts)

# BMI ranges for labels
bmi_ranges = {
    'Underweight': '<18.5',
    'Normal weight': '18.5-24.9',
    'Overweight': '25-29.9',
    'Obesity Class I': '30-34.9',
    'Obesity Class II': '35-39.9',
    'Obesity Class III': '≥40'
}

order = ['Underweight', 'Normal weight', 'Overweight', 
         'Obesity Class I', 'Obesity Class II', 'Obesity Class III']

plt.figure(figsize=(12,6))
ax = sns.boxplot(
    data=df_clean,
    x='bmi_class',
    y='charges',
    order=order,
    palette = "pastel"
)

# Add BMI range to x-axis labels
labels = [tick.get_text() for tick in ax.get_xticklabels()]
new_labels = [f"{label}\n({bmi_ranges[label]})" for label in labels]
ax.set_xticklabels(new_labels)

# Annotate median and sample size
for i, cat in enumerate(order):
    median_val = df_clean[df_clean['bmi_class']==cat]['charges'].median()
    count = df_clean[df_clean['bmi_class']==cat].shape[0]
    ax.text(i, median_val + median_val*0.05, f'Median: {median_val:.0f}\nN={count}', 
            ha='center', va='bottom', fontsize=10, color='black')

plt.title('Insurance Charges by BMI Category')
plt.ylabel('Charges')
plt.xlabel('BMI Category')
plt.xticks(rotation=0)
plt.show()


#####################################
######SMOKING + AGE VS CHARGES#######
#####################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

# Scatterplot: Age vs Charges by Smoker Status
sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
plt.title('Charges vs Age by Smoker Status')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Interaction regression model: Charges ~ Age * Smoker
model_age = smf.ols('charges ~ age * smoker', data=df).fit()
print(model_age.summary())

# Define custom age bins
age_bins = [17, 25, 35, 45, 55, 65]  # meaningful ranges
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65']

# Assign age groups
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True, right=True)

# Compute group means
age_group_means = df.groupby(['age_group', 'smoker'])['charges'].mean().reset_index()

# Pivot to calculate smoker vs non-smoker percentage difference
age_pivot_means = age_group_means.pivot(index='age_group', columns='smoker', values='charges')
age_pivot_means['pct_diff'] = ((age_pivot_means['yes'] - age_pivot_means['no']) / age_pivot_means['no']) * 100

# Boxplot
plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df, x='age_group', y='charges', hue='smoker', showfliers=False)
plt.title('Charges by Age Group (Custom Bins) and Smoker Status')
plt.xlabel('Age Group')
plt.ylabel('Charges')

# Overlay group means as diamond markers
sns.pointplot(
    data=age_group_means,
    x='age_group',
    y='charges',
    hue='smoker',
    markers='D',
    dodge=0.4,
    join=False,
    palette='dark:black',
    ax=ax,
    errorbar=None
)

# Add numeric labels for group means
for i, row in age_group_means.iterrows():
    x = list(df['age_group'].cat.categories).index(row['age_group'])
    offset = -0.2 if row['smoker'] == 'no' else 0.2
    ax.text(x + offset, row['charges'] + 500, f'{row["charges"]:.0f}',
            ha='center', color='black', fontsize=9, fontweight='bold')

# Add percentage difference labels above each age group (for smokers)
for i, (group, vals) in enumerate(age_pivot_means.iterrows()):
    pct = vals['pct_diff']
    ax.text(i, vals['yes'] + 2500, f'+{pct:.0f}%', ha='center',
            color='red', fontsize=10, fontweight='bold')

plt.legend(title='Smoker')
plt.tight_layout()
plt.show()


#####################################
######SMOKING + BMI VS CHARGES#######
#####################################

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Scatterplot: BMI vs Charges by Smoker Status
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
plt.title('Charges vs BMI by Smoker Status')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

# Interaction regression model: Charges ~ BMI * Smoker
# BMI has a far stronger effect on charges for smokers
model_bmi = smf.ols('charges ~ bmi * smoker', data=df).fit()
print(model_bmi.summary())

# Boxplot based on WHO BMI categories
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    elif 30 <= bmi < 35:
        return 'Obesity Class I'
    elif 35 <= bmi < 40:
        return 'Obesity Class II'
    else:
        return 'Obesity Class III'

df['bmi_class'] = df['bmi'].apply(bmi_category)
order = ['Underweight', 'Normal weight', 'Overweight', 
         'Obesity Class I', 'Obesity Class II', 'Obesity Class III']

# Group means and pivot for percentage differences
bmi_group_means = df.groupby(['bmi_class', 'smoker'])['charges'].mean().reset_index()
bmi_pivot_means = bmi_group_means.pivot(index='bmi_class', columns='smoker', values='charges')
bmi_pivot_means['pct_diff'] = ((bmi_pivot_means['yes'] - bmi_pivot_means['no']) / bmi_pivot_means['no']) * 100

plt.figure(figsize=(12,6))
ax = sns.boxplot(
    data=df, x='bmi_class', y='charges', hue='smoker',
    order=order, showfliers=False, palette="pastel"
)

# Overlay group means
sns.pointplot(
    data=bmi_group_means, x='bmi_class', y='charges', hue='smoker',
    markers='D', dodge=0.3, join=False, palette='dark:black',
    ax=ax, errorbar=None
)

n_hue = df['smoker'].nunique()
dodge_amount = 0.3

for i, row in bmi_group_means.iterrows():
    x = order.index(row['bmi_class'])
    offset = -dodge_amount/2 if row['smoker'] == 'no' else dodge_amount/2
    ax.text(x + offset, row['charges'] + 500, f'{row["charges"]:.0f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Percentage difference for smokers only
for i, group in enumerate(order):
    pct = bmi_pivot_means.loc[group, 'pct_diff']
    yes_mean = bmi_pivot_means.loc[group, 'yes']
    ax.text(i + dodge_amount/2, yes_mean + 2500, f'+{pct:.0f}%',
            ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')

plt.title('Insurance Charges by WHO BMI Category and Smoker Status')
plt.xlabel('BMI Category')
plt.ylabel('Charges')
plt.legend(title='Smoker', loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
