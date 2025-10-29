# ann_insurance.py
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

# 1) Load data ---------------------------------------------------------------
# Expected columns (rename if needed):
# age, sex, bmi, children, smoker, region, charges
df = pd.read_csv("insurance.csv")

# Target + features
y = df["charges"].astype("float32").values
X = df.drop(columns=["charges"])

# Identify column types (edit if your schema differs)
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

# 2) Train/val/test split ----------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 3) Preprocess pipeline -----------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# Fit the preprocessor on training only
preprocess.fit(X_train)

# Transform splits to numpy arrays
X_train_p = preprocess.transform(X_train)
X_val_p   = preprocess.transform(X_val)
X_test_p  = preprocess.transform(X_test)

input_dim = X_train_p.shape[1]

# 4) Build the ANN -----------------------------------------------------------
def build_model(input_dim, lr=1e-3, l2=1e-4, dropout=0.05):
    reg = tf.keras.regularizers.l2(l2)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1)  # regression output
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

model = build_model(input_dim)

# 5) Training setup ----------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=20, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_insurance_ann.keras", monitor="val_mae", save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae", factor=0.5, patience=8, min_lr=1e-5, verbose=1
    ),
]

history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=400,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 6) Evaluation --------------------------------------------------------------
preds = model.predict(X_test_p).ravel()
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
mape = np.mean(np.abs((y_test - preds) / np.maximum(1e-8, y_test))) * 100

print(f"Test MAE : {mae:,.2f}")
print(f"Test RMSE: {rmse:,.2f}")
print(f"Test MAPE: {mape:,.2f}%")

# 7) Save the preprocessor + quick inference helper -------------------------
import joblib
joblib.dump(preprocess, "insurance_preprocessor.joblib")

# Example: scoring a single row
example = pd.DataFrame([{
    "age": 45, "sex": "male", "bmi": 31.2, "children": 2,
    "smoker": "yes", "region": "southeast"
}])
x_ex = preprocess.transform(example)
print("Example prediction:", float(model.predict(x_ex)))
