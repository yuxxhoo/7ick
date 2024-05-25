import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 文件路径
file_path = r"C:\Users\7iCK\Desktop\10_features.csv"
df = pd.read_csv(file_path)

# 选择特征和目标变量
target = 'Gas close'  # 选择你的目标变量列名
features = df.columns.difference([target, 'Date'])  # 选择特征列，排除时间戳列

X = df[features].values
y = df[target].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # 计算均方根误差(RMSE)
mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差(MAE)
r2 = r2_score(y_test, y_pred)

# 打印MSE, RMSE, MAE 和 R2
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# 打印系数
coefficients = model.coef_  # 获取线性回归模型的系数
intercept = model.intercept_  # 获取截距
print("Coefficients:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef}")
print(f"Intercept: {intercept}")

# 绘制预测与实际值的散点图
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 绘制45度线
plt.xlabel("Actual Natural Gas Prices")
plt.ylabel("Predicted Natural Gas Prices")
plt.title("Natural Gas Prices Prediction (MLR)")
plt.show()

# 绘制实际与预测的折线图
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Gas Prices', color='red')
plt.plot(y_pred, label='Predicted Gas Prices', color='blue')
plt.title('Actual vs Predicted Gas Prices')
plt.xlabel('Sample')
plt.ylabel('Gas Prices')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  # 将标注放在右上角
plt.grid(True)  # 添加网格线
plt.yticks(np.arange(0, 11, 1))  # 设置Y轴刻度从0到10
plt.ylim(0, 10)  # 设置Y轴范围从0到10
plt.show()

