
###############################
###CLUSTERING##################
###############################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from matplotlib.table import Table

# Load data
df = pd.read_csv(r"C:\Users\kevin\Documents\NUS\Y2S1\SDS Datathon\insurance.csv")

# Features
X = df[["age", "bmi", "children", "sex", "smoker", "region"]]

# Preprocess numeric and categorical columns
num_cols = ["age", "bmi"]
cat_cols = ["smoker"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

# K-Means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_processed)
df["cluster"] = labels

# Cluster summary
cluster_summary = df.groupby("cluster")[["age","bmi","charges"]].mean().round(2)
cluster_summary["pct_smokers"] = df.groupby("cluster")["smoker"].apply(lambda x: (x=="yes").mean()*100).round(2)
cluster_summary = cluster_summary.reset_index()

# Print cluster summary as console table
print(cluster_summary.to_string(index=False))

# Visualize average charges by cluster
sns.barplot(data=df.groupby("cluster", as_index=False)["charges"].mean(),
            x="cluster", y="charges", palette="Set2")
plt.title("Average Charges by Cluster")
plt.show()

# Create a formatted table figure for presentation
fig, ax = plt.subplots(figsize=(6,2))
ax.set_axis_off()
tbl = Table(ax, bbox=[0,0,1,1])

# Add table headers
for i, col in enumerate(cluster_summary.columns):
    tbl.add_cell(0, i, 1/len(cluster_summary.columns), 0.3, text=col, loc='center', facecolor='lightgrey')

# Add table data
for row in range(len(cluster_summary)):
    for col in range(len(cluster_summary.columns)):
        tbl.add_cell(row+1, col, 1/len(cluster_summary.columns), 0.3, 
                     text=cluster_summary.iloc[row, col], loc='center', facecolor='white')

ax.add_table(tbl)
plt.title("Cluster Summary Table")
plt.show()
