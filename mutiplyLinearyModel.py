import pandas as pd
import numpy as np

# --- 1. Load and Prepare the Data ---
df = pd.read_csv('Concrete_Data.csv')

# 清理列名
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

# 按要求分割数据
train_df = pd.concat([df.iloc[:501], df.iloc[631:]]).copy()
test_df = df.iloc[501:631].copy()

# 分离预测变量 (X) 和响应变量 (y)
response_variable = 'Compressive_Strength'
predictor_variables = [col for col in df.columns if col != response_variable]

X_train_raw = train_df[predictor_variables]
y_train = train_df[response_variable]
X_test_raw = test_df[predictor_variables]
y_test = test_df[response_variable]

# --- 2. 预处理 & 模型函数 ---

def standardize_features(train_data, test_data):
    """根据训练数据对所有预测变量进行标准化。"""
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_scaled = (train_data - mean) / std
    test_scaled = (test_data - mean) / std
    return train_scaled, test_scaled

def gradient_descent_multi(X, y, learning_rate=0.01, epochs=10000):
    """多元线性回归的梯度下降算法。"""
    # 初始化参数
    m = np.zeros(X.shape[1]) # m 是一个包含8个系数的向量
    b = 0
    n = len(y)

    for _ in range(epochs):
        y_pred = np.dot(X, m) + b
        error = y_pred - y
        # 计算梯度
        m_grad = (2/n) * np.dot(X.T, error)
        b_grad = (2/n) * np.sum(error)
        # 更新参数
        m -= learning_rate * m_grad
        b -= learning_rate * b_grad
    return m, b

# --- 3. 评估指标函数 ---
def predict_multi(X, m, b):
    return np.dot(X, m) + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def variance_explained(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean)**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

# --- 4. 主分析流程 ---

# 标准化预测变量
X_train, X_test = standardize_features(X_train_raw, X_test_raw)

# 转换为 NumPy 数组以便计算
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

# 训练模型 (超参数 learning_rate=0.01, epochs=10000 是一个不错的起点)
m_coeffs, b_intercept = gradient_descent_multi(X_train, y_train, learning_rate=0.01, epochs=10000)

# 进行预测
y_train_pred = predict_multi(X_train, m_coeffs, b_intercept)
y_test_pred = predict_multi(X_test, m_coeffs, b_intercept)

# 评估模型
mse_train = mse(y_train, y_train_pred)
ve_train = variance_explained(y_train, y_train_pred)
mse_test = mse(y_test, y_test_pred)
ve_test = variance_explained(y_test, y_test_pred)

# --- 5. 显示结果 ---
# (此部分为代码运行后在你的终端上显示的内容)
print("--- 多元回归模型结果 ---")
print("\nm and b values:")
print(f"  b (intercept): {b_intercept:.4f}")
for i, predictor in enumerate(predictor_variables):
    print(f"  m_{i+1} ({predictor}): {m_coeffs[i]:.4f}")

print(f"\n训练集表现:")
print(f"  MSE on training data: {mse_train:.4f}")
print(f"  Variance Explained / R-Squared on training data: {ve_train:.4f}")

print(f"\n测试集表现:")
print(f"  MSE on testing data: {mse_test:.4f}")
print(f"  Variance Explained / R-Squared on testing data: {ve_test:.4f}")