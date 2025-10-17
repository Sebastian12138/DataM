import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 加载和准备数据 ---
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

X_train_raw = train_df[predictor_variables]
y_train = train_df[response_variable]
X_test_raw = test_df[predictor_variables]
y_test = test_df[response_variable]


def standardize_features(train_data, test_data):
    mean = train_data.mean()
    std = train_data.std()
    train_scaled = (train_data - mean) / std
    test_scaled = (test_data - mean) / std
    return train_scaled, test_scaled


X_train, X_test = standardize_features(X_train_raw, X_test_raw)


def gradient_descent_with_loss_tracking(X, y, learning_rate=0.01, epochs=10000):
    """
    修改后的多元梯度下降，会返回每次迭代的损失值。
    """
    m = np.zeros(X.shape[1])
    b = 0
    n = len(y)
    loss_history = []  # 新增：用于存储每次迭代的损失

    for i in range(epochs):
        y_pred = np.dot(X, m) + b
        error = y_pred - y

        # 计算并记录当前迭代的 MSE 损失
        loss = np.mean(error ** 2)
        loss_history.append(loss)

        # 计算梯度
        m_grad = (2 / n) * np.dot(X.T, error)
        b_grad = (2 / n) * np.sum(error)

        # 更新参数
        m -= learning_rate * m_grad
        b -= learning_rate * b_grad

    return m, b, loss_history


# --- 4. 训练模型并获取损失历史 ---
X_train_np = X_train.values
y_train_np = y_train.values

m_coeffs, b_intercept, history = gradient_descent_with_loss_tracking(X_train_np, y_train_np)

# --- 5. 绘制并保存损失曲线图 ---
plt.figure(figsize=(10, 6))
plt.plot(range(len(history)), history)
plt.title('Loss (MSE) over Iterations during Gradient Descent')
plt.xlabel('Iteration Number')
plt.ylabel('Mean Squared Error (Loss)')
plt.grid(True)

# 将图片保存到文件
plt.savefig('loss_curve.png')

print("损失曲线图已生成并保存为 'loss_curve.png'")