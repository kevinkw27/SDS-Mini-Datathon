##############################################
#### DATA CLEANING############################
##############################################

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# Shows the first 5 rows of dataframe and column headers
df.head()

# Data types of each column
df.dtypes

# Concise summary of dataframe
df.info()

# General descriptive statistics
df.describe(include="all")

# Greater detail of descriptive statistics
df["smoker"].value_counts()
    
# No. of unique elements in each column
"""
for col in df.columns:
  print(col)
  print(df[col].unique())
  print()
"""

# Removes duplicates and keeps unique rows
df_clean = df.drop_duplicates().copy()  # creates a new dataframe without the duplicates
df_clean.info()

# Reset indexes
df_clean.reset_index(drop=True, inplace=True) # drop=True ensures that the old index is not added as a column by itself
df_clean.info()

##############################################
#### EXPLORATORY DATA ANALYSIS PART 2 ########
##############################################
"""
import numpy as np #super fast calculator
import seaborn as sns
import matplotlib.pyplot as plt

# Checking if "sex" is imbalanced
sex_percentage = df_clean['sex'].value_counts(normalize=True) * 100
# male 50.49%, female 49.51%

# Checking if "children" is imbalanced
children_percentage = df_clean['children'].value_counts(normalize=True) * 100
# 0 42.9%, 1 24.2%, 2 18%, 3 11.7%, 4 1.87%, 5 1.35%

# Checking if "smoker" is imbalanced
smoker_percentage = df_clean['smoker'].value_counts(normalize=True) * 100
# no 79.5%, yes 20.5%


# reporting the imbalance for 'EverBenched'

ax = sns.countplot(x='smoker', data=df_clean)
plt.title("Class Distribution for 'Smoker'")

for p in ax.patches:
    count = p.get_height()
    percent = 100 * count / len(df)
    ax.annotate(f'{percent:.1f}%',              # format with 1 dp
                (p.get_x() + p.get_width() / 2., count),  # position
                ha='center', va='bottom', fontsize=10, color='black')
plt.show()


# Checking if "region" is imbalanced
region_percentage = df_clean['region'].value_counts(normalize=True) * 100
# southeast 27.2%, southwest 24.3%, northwest 24.2%, northeast 24.2%

# Numerical Univariate Analysis
df_clean.describe()
# Charges: Mean - $13,279 Min - $1121 25% - $4746 Median - $9386 75% - $16,657 Max - $63,770

#Boxplot for charges
import scipy
df_clean['log_charges'] = np.log(df['charges'])
sns.boxplot(x='log_charges', data=df_clean)
plt.title(f'Distribution of charges')
plt.xlabel('charges')
plt.ylabel('Count')
plt.show()
"""

#############################
####CORRELATION MATRIX#######
#############################

"""
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
plt.title("Correlation Matrix Including Encoded Categorical Variables (Smoker, Sex, Region)")
plt.show()

"""

## CATEGORICAL FEATURES VS CHARGES #######
"""
# Boxplot showing smoker vs charges -- Smoker higher median than non-smoker
sns.boxplot(x="smoker", y="charges", data=df_clean, palette="Set2", hue="smoker", legend=False)
plt.title("Charges by Smoking")
plt.xlabel("Smoking")
plt.ylabel("Charges")
plt.show()

# Boxplot showing region vs charges -- Quite similar across 4 regions
sns.boxplot(x="region", y="charges", data=df_clean, palette="Set2", hue="region", legend=False)
plt.title("Charges by Region")
plt.xlabel("Region")
plt.ylabel("Charges")
plt.show()

# Boxplot showing sex vs charges -- Similar across sex
sns.boxplot(x="sex", y="charges", data=df_clean, palette="Set2", hue="sex", legend=False)
plt.title("Charges by Sex")
plt.xlabel("Sex")
plt.ylabel("Charges")
plt.show()
"""

# Boxplot showing children vs charges -- While it's quite even throughout, may be unrepresented because
# limited data for 4 and 5 children
sns.boxplot(x="children", y="charges", data=df_clean, palette="Set2", hue="children", legend=False)
plt.title("Charges by children")
plt.xlabel("Children")
plt.ylabel("Charges")
plt.show()
"""

#### OTHER EDA NOT RELATED TO DEPENDENT VARIABLE ###
"""
# Histogram of insured ages
plt.figure(figsize=(6,4))
sns.histplot(df_clean["age"], bins = 20 , kde = True)
#kde=Kernel Density Estimation=smooth continuous line (pdf of the data) on the histogram
plt.title("Distribution of Insured Ages")
plt.xlabel("age")
plt.ylabel("Count")
plt.show()
"""
"""
# grouped barplot of smokers across region - 1 region got almost double the number of smokers
sns.countplot(x="smoker",hue="region",data=df_clean,palette="colorblind")
plt.title("Smokers by region")
plt.show()
"""

"""
