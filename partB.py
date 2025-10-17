import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- 1. Load and Prepare Data ---
df = pd.read_csv('Concrete_Data.csv')

def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.split('(')[0].strip().replace(' ', '_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

df = clean_col_names(df)
df.rename(columns={'Concrete_compressive_strength': 'Compressive_Strength'}, inplace=True)

train_df = pd.concat([df.iloc[:501], df.iloc[631:]]).copy()
test_df = df.iloc[501:631].copy()

response_variable = 'Compressive_Strength'
predictor_variables = [col for col in df.columns if col != response_variable]

# --- Data Versions ---
# Raw Data
X_train_raw = train_df[predictor_variables]
y_train = train_df[response_variable]
X_test_raw = test_df[predictor_variables]
y_test = test_df[response_variable]

# Standardized Data
X_train_std = (X_train_raw - X_train_raw.mean()) / X_train_raw.std()
X_test_std = (X_test_raw - X_train_raw.mean()) / X_train_raw.std()

# Log-Transformed Data
# Add 1 to avoid log(0) for features like Fly_Ash or Superplasticizer
X_train_log = np.log(X_train_raw + 1)
X_test_log = np.log(X_test_raw + 1)

# --- Helper function for evaluation ---
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # MSE
    mse_train = np.mean((y_train - y_train_pred)**2)
    mse_test = np.mean((y_test - y_test_pred)**2)
    # R-squared
    r2_train = model.rsquared
    ss_total_test = np.sum((y_test - np.mean(y_test))**2)
    ss_residual_test = np.sum((y_test - y_test_pred)**2)
    r2_test = 1 - (ss_residual_test / ss_total_test)
    return mse_train, mse_test, r2_train, r2_test

# --- Analysis for Each Question ---

# == Q1.1 & Q2.1: Model with Raw Data ==
X_train_raw_const = sm.add_constant(X_train_raw)
X_test_raw_const = sm.add_constant(X_test_raw)
model_raw = sm.OLS(y_train, X_train_raw_const).fit()

mse_train_raw, mse_test_raw, r2_train_raw, r2_test_raw = evaluate_model(model_raw, X_train_raw_const, X_test_raw_const, y_train, y_test)

print("--- Part B: Analysis with statsmodels ---")
print("\n--- Q1.1 Performance (Raw Data) ---")
print(f"MSE on training data: {mse_train_raw:.4f}")
print(f"MSE on testing data: {mse_test_raw:.4f}")
print(f"R-squared on training data: {r2_train_raw:.4f}")
print(f"R-squared on testing data: {r2_test_raw:.4f}")

print("\n--- Q2.1 P-values (Raw Data) ---")
print("Statistical test: t-test for each coefficient")
print("P-values for each feature:")
print(model_raw.pvalues.drop('const').to_string())


# == Q2.3: Model with Standardized Data ==
X_train_std_const = sm.add_constant(X_train_std)
model_std = sm.OLS(y_train, X_train_std_const).fit()

print("\n\n--- Q2.3 New P-values (Standardized Data) ---")
print("P-values for each feature after standardization:")
print(model_std.pvalues.drop('const').to_string())

# == Q2.5: Model with Log-Transformed Data ==
X_train_log_const = sm.add_constant(X_train_log)
model_log = sm.OLS(y_train, X_train_log_const).fit()

print("\n\n--- Q2.5 New P-values (Log-Transformed Data) ---")
print("P-values for each feature after log transform:")
print(model_log.pvalues.drop('const').to_string())