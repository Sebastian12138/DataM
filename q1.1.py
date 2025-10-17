import pandas as pd
import numpy as np

# --- 1. Load and Prepare the Data ---

# Load the dataset from the CSV file
file_name = 'Concrete_Data.csv'
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: Make sure the file '{file_name}' is in the same directory as the script.")
    exit()

# Clean column names for easier access
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

# Split the data into training and testing sets as specified
test_df = df.iloc[501:631].copy()
train_df = pd.concat([df.iloc[:501], df.iloc[631:]]).copy()

# --- 2. Preprocessing and Model Functions ---

def standardize(train_data, test_data):
    """Standardizes the predictors based on the training data."""
    mean = train_data.mean()
    std = train_data.std()
    train_scaled = (train_data - mean) / std
    test_scaled = (test_data - mean) / std
    return train_scaled, test_scaled

def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    """
    Your custom gradient descent optimizer for univariate linear regression.
    """
    m, b = 0.0, 0.0
    n = len(x)
    for _ in range(epochs):
        y_pred = m * x + b
        error = y_pred - y
        m_grad = (2/n) * np.sum(x * error)
        b_grad = (2/n) * np.sum(error)
        m -= learning_rate * m_grad
        b -= learning_rate * b_grad
    return m, b

# --- 3. Evaluation Metrics ---

def predict(x, m, b):
    """Makes predictions with the trained model."""
    return m * x + b

def mse(y_true, y_pred):
    """Calculates Mean Squared Error."""
    return np.mean((y_true - y_pred)**2)

def variance_explained(y_true, y_pred):
    """Calculates Variance Explained (R-squared)."""
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean)**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    if ss_total == 0:
        return 1 if ss_residual == 0 else 0
    return 1 - (ss_residual / ss_total)

# --- 4. Main Analysis Loop ---

response_variable = 'Compressive_Strength'
predictor_variables = [col for col in df.columns if col != response_variable]

results = {}

# Hyperparameters for each model (you can tune these)
hyperparams = {
    'Cement': {'learning_rate': 0.1, 'epochs': 1000},
    'Blast_Furnace_Slag': {'learning_rate': 0.1, 'epochs': 1000},
    'Fly_Ash': {'learning_rate': 0.1, 'epochs': 1000},
    'Water': {'learning_rate': 0.1, 'epochs': 1000},
    'Superplasticizer': {'learning_rate': 0.1, 'epochs': 1000},
    'Coarse_Aggregate': {'learning_rate': 0.1, 'epochs': 1000},
    'Fine_Aggregate': {'learning_rate': 0.1, 'epochs': 1000},
    'Age': {'learning_rate': 0.1, 'epochs': 1000}
}

for predictor in predictor_variables:
    # Prepare data for the current predictor
    X_train = train_df[predictor]
    y_train = train_df[response_variable]
    X_test = test_df[predictor]
    y_test = test_df[response_variable]

    # Standardize the predictor variable
    X_train_scaled, X_test_scaled = standardize(X_train, X_test)

    # Train the model using your gradient descent function
    m, b = gradient_descent(X_train_scaled, y_train,
                            learning_rate=hyperparams[predictor]['learning_rate'],
                            epochs=hyperparams[predictor]['epochs'])

    # Make predictions on training and testing sets
    y_train_pred = predict(X_train_scaled, m, b)
    y_test_pred = predict(X_test_scaled, m, b)

    # Evaluate the model and store the results
    results[predictor] = {
        'm': m,
        'b': b,
        'MSE on training data': mse(y_train, y_train_pred),
        'Variance Explained on training data': variance_explained(y_train, y_train_pred),
        'MSE on testing data': mse(y_test, y_test_pred),
        'Variance Explained on testing data': variance_explained(y_test, y_test_pred)
    }

# --- 5. Display the Results ---
for predictor, metrics in results.items():
    print(f"--- Model: {predictor} as predictor ---")
    print(f"m and b values:")
    print(f"  m (slope): {metrics['m']:.4f}")
    print(f"  b (intercept): {metrics['b']:.4f}")
    print(f"MSE on training data: {metrics['MSE on training data']:.4f}")
    print(f"Variance Explained / R-Squared on training data: {metrics['Variance Explained on training data']:.4f}")
    print(f"MSE on testing data: {metrics['MSE on testing data']:.4f}")
    print(f"Variance Explained / R-Squared on testing data: {metrics['Variance Explained on testing data']:.4f}")
    print("\n" + "="*40 + "\n")