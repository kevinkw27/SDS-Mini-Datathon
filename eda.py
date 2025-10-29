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

# trying out pairwise relationships (to test out significance of other variables given that smoking status is still the most significant predictor)
sns.pairplot(
    df_clean,
    vars=["age","bmi","children","charges"],
    hue="smoker",
    corner=True,              # show lower triangle only
    diag_kind="kde",          # smooth diag
    plot_kws={"s":18, "alpha":0.45, "edgecolor":"none"},  # smaller, translucent points
    height=2.2, aspect=1.0    # tighter grid
)
plt.suptitle("Pairwise relationships (colored by smoker)", y=1.02)

### Some things i discover:
### 1) Smoking is the most important determining factor for insurance charges
### 2) BMI and age can raise insurance prices. (This is especially true for smokers)
### 3) Number of children, region of residence, gender are minor determining factors
### 4) The combined effect of smoker + high BMI can raise insurance prices much higher together. MUST WATCH OUT FOR THIS COMBO

"""

###############################
######AGE VS SMOKER############
###############################
"""
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
"""

###############################
######BMI VS SMOKER############
###############################
"""
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

# Compute BMI quartiles
bmi_quartiles = np.quantile(df['bmi'], [0, 0.25, 0.5, 0.75, 1.0])
# Round to 1 decimal for readability
bmi_quartiles_rounded = [round(q, 1) for q in bmi_quartiles]

# Create non-overlapping BMI group labels
bmi_labels = [f'{bmi_quartiles_rounded[i]}-{round(bmi_quartiles_rounded[i+1]-0.1, 1)}' 
              for i in range(len(bmi_quartiles_rounded)-1)]

# Assign BMI groups with non-overlapping bins
df['bmi_group'] = pd.cut(
    df['bmi'], 
    bins=bmi_quartiles, 
    labels=bmi_labels, 
    include_lowest=True, 
    right=False  # left-inclusive, right-exclusive
)

# Compute group means
bmi_group_means = df.groupby(['bmi_group', 'smoker'])['charges'].mean().reset_index()

# Pivot to calculate smoker vs non-smoker percentage difference
bmi_pivot_means = bmi_group_means.pivot(index='bmi_group', columns='smoker', values='charges')
bmi_pivot_means['pct_diff'] = ((bmi_pivot_means['yes'] - bmi_pivot_means['no']) / bmi_pivot_means['no']) * 100

# Boxplot
plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df, x='bmi_group', y='charges', hue='smoker', showfliers=False)
plt.title('Charges by BMI Group (Quartiles) and Smoker Status')
plt.xlabel('BMI Group')
plt.ylabel('Charges')

# Overlay group means as diamond markers
sns.pointplot(
    data=bmi_group_means,
    x='bmi_group',
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
for i, row in bmi_group_means.iterrows():
    x = list(df['bmi_group'].cat.categories).index(row['bmi_group'])
    offset = -0.2 if row['smoker'] == 'no' else 0.2
    ax.text(x + offset, row['charges'] + 500, f'{row["charges"]:.0f}',
            ha='center', color='black', fontsize=9, fontweight='bold')

# Add percentage difference labels above each BMI group (for smokers)
for i, (group, vals) in enumerate(bmi_pivot_means.iterrows()):
    pct = vals['pct_diff']
    ax.text(i, vals['yes'] + 2500, f'+{pct:.0f}%', ha='center',
            color='red', fontsize=10, fontweight='bold')

plt.legend(title='Smoker')
plt.tight_layout()
plt.show()
"""
