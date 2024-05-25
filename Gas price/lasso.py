import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
file_path = r"C:\Users\7iCK\Desktop\工作\天然气\天然气数据 处理后\18_features.csv"
df = pd.read_csv(file_path)


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 创建滞后特征
features = df.drop(columns='Gas close').shift(1)
target = df['Gas close']

# 删除NaN值
features = features[1:]
target = target[1:]

# 训练Lasso回归模型
lasso = Lasso(alpha=0.03)
lasso.fit(features, target)

# 预测
predictions = lasso.predict(features)

# 计算评估指标
r2 = r2_score(target, predictions)
mse = mean_squared_error(target, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(target, predictions)



# 打印评估指标
print(f'R2 Score: {r2}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# 打印所有系数
coefficients = pd.Series(lasso.coef_, index=features.columns)
print("Model Coefficients:")
print(coefficients)

