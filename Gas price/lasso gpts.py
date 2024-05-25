import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 加载数据
file_path = r"C:\Users\7iCK\Desktop\10_features.csv"
df = pd.read_csv(file_path)

# 设置时间为索引
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 准备特征和目标变量
X = df.drop(columns=['Gas close'])
y = df['Gas close']

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置Lasso模型，使用时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
lasso_model = LassoCV(cv=tscv, random_state=42).fit(X_scaled, y)

# 获取模型系数和最优alpha值
coefficients = lasso_model.coef_
best_alpha = lasso_model.alpha_

# 创建DataFrame来显示所有系数
all_features = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient'])

# 预测和计算性能指标
y_pred = lasso_model.predict(X_scaled)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mse)

# 打印所有特征的系数
print("所有特征的系数:\n", all_features)

# 打印重要特征和性能指标
print("\n重要特征和其系数:\n", all_features[all_features['Coefficient'] != 0])
print("\n最优正则化参数 (alpha):", best_alpha)
print("\n性能指标:")
print("R^2 (决定系数):", r2)
print("MSE (均方误差):", mse)
print("MAE (平均绝对误差):", mae)
print("RMSE (均方根误差):", rmse)

