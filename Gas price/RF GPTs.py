import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 读取数据文件
file_path = r"C:\Users\7iCK\Desktop\10_features.csv"
df = pd.read_csv(file_path)

# 设置"Date"列为索引
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 创建滞后特征
df['Gas close_lag1'] = df['Gas close'].shift(1)
df.dropna(inplace=True)

# 分离特征变量和目标变量
X = df.drop(columns=['Gas close'])  # 特征
y = df['Gas close'].values  # 目标，转换为NumPy数组

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 计算性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy = 100 - mape

# 打印性能指标
print(f'Test MSE: {mse}')
print(f'Test RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}%')
print(f'Accuracy: {accuracy}%')
print(f'R-squared: {r2}')

# 绘制实际与预测的折线图
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Gas Prices', color='red')
plt.plot(y_pred, label='Predicted Gas Prices', color='blue')
plt.title('Actual vs Predicted Gas Prices')
plt.xlabel('Sample')
plt.ylabel('Gas Prices')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  # 将标注放在右上角
plt.grid(True)  # 添加网格线

plt.show()
