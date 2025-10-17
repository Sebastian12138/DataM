import pandas as pd
import numpy as np

# --- 1. 加载和准备数据 ---
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

# 分割数据
train_df = pd.concat([df.iloc[:501], df.iloc[631:]]).copy()
test_df = df.iloc[501:631].copy()

# 分离预测变量 (X) 和响应变量 (y)
response_variable = 'Compressive_Strength'
predictor_variables = [col for col in df.columns if col != response_variable]

# 重点：直接使用原始（未缩放）的数据
X_train = train_df[predictor_variables]
y_train = train_df[response_variable]
X_test = test_df[predictor_variables]
y_test = test_df[response_variable]


# --- 2. 模型函数 ---

def gradient_descent_multi_raw(X, y, learning_rate=1e-7, epochs=200000):
    """
    针对原始数据的多元回归梯度下降。
    需要非常精细的超参数调整。
    """
    m = np.zeros(X.shape[1])
    b = 0
    n = len(y)

    for i in range(epochs):
        y_pred = np.dot(X, m) + b
        error = y_pred - y

        # 检查是否出现数值溢出 (inf) 或无效值 (NaN)
        if np.isinf(y_pred).any() or np.isnan(y_pred).any():
            print(f"在第 {i} 次迭代时检测到数值溢出。停止训练。")
            return np.full_like(m, np.nan), np.nan

        m_grad = (2 / n) * np.dot(X.T, error)
        b_grad = (2 / n) * np.sum(error)

        m -= learning_rate * m_grad
        b -= learning_rate * b_grad

    return m, b


# --- 3. 评估指标函数 ---
def predict_multi(X, m, b):
    return np.dot(X, m) + b


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def variance_explained(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


# --- 4. 主分析流程 ---

# 转换为 NumPy 数组
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# 使用为原始数据精细调整的超参数进行训练
# 注意：你可能需要根据你的结果进一步减小 learning_rate 或增加 epochs
m_coeffs, b_intercept = gradient_descent_multi_raw(X_train_np, y_train_np, learning_rate=1e-7, epochs=200000)

# --- 5. 显示结果 ---
if np.isnan(m_coeffs).any():
    print("由于数值不稳定，训练失败。请尝试一个更小的学习率。")
else:
    y_train_pred = predict_multi(X_train_np, m_coeffs, b_intercept)
    y_test_pred = predict_multi(X_test_np, m_coeffs, b_intercept)

    mse_train = mse(y_train_np, y_train_pred)
    ve_train = variance_explained(y_train_np, y_train_pred)
    mse_test = mse(y_test_np, y_test_pred)
    ve_test = variance_explained(y_test_np, y_test_pred)

    print("--- 多元回归模型结果 (使用原始数据) ---")
    print("\nm and b values:")
    print(f"  b (intercept): {b_intercept:.4f}")
    for i, predictor in enumerate(predictor_variables):
        print(f"  m_{i + 1} ({predictor}): {m_coeffs[i]:.4f}")

    print(f"\n训练集表现:")
    print(f"  MSE on training data: {mse_train:.4f}")
    print(f"  Variance Explained / R-Squared on training data: {ve_train:.4f}")

    print(f"\n测试集表现:")
    print(f"  MSE on testing data: {mse_test:.4f}")
    print(f"  Variance Explained / R-Squared on testing data: {ve_test:.4f}")