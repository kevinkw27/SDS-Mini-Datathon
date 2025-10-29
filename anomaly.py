import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1) Load dataset
df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# 2) Boxplots for numeric features
numeric_cols = ['age','bmi','children','charges']
plt.figure(figsize=(14,4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1,4,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# 3) Scatter plot: BMI vs Charges, colored by smoker
plt.figure(figsize=(8,6))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('BMI vs Charges by Smoker Status')
plt.show()

# Optional: log-transform charges for visibility
plt.figure(figsize=(8,6))
sns.scatterplot(x='bmi', y=np.log1p(df['charges']), hue='smoker', data=df)
plt.title('BMI vs log(Charges + 1) by Smoker Status')
plt.show()

# 4) Isolation Forest for anomaly detection
iso = IsolationForest(contamination=0.05, random_state=42)  # detect ~5% anomalies
df['anomaly'] = iso.fit_predict(df[numeric_cols])  # -1 = anomaly, 1 = normal

# Visualize anomalies on scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='bmi', y='charges', 
    hue=df['anomaly'].map({1:'Normal', -1:'Anomaly'}),
    palette={'Normal':'blue','Anomaly':'red'},
    alpha=0.7, data = df
)
plt.title('Isolation Forest Anomalies')
plt.show()

# 5) Table of anomalies
anomalies = df[df['anomaly'] == -1]
print(f"Number of anomalies detected: {len(anomalies)}")
print(anomalies[['age','bmi','children','sex','smoker','region','charges']].sort_values(by='charges', ascending=False))
