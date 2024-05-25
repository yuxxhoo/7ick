
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

# 文件路径
file_path = r"C:\Users\7iCK\Desktop\工作\天然气\天然气数据 处理后\18_features.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 并选择特征和目标变量
X = df.drop(['Gas close'], axis=1).values
y = df['Gas close'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 模型
model = XGBRegressor(seed=100, n_estimators=115, max_depth=5, eval_metric='rmse', learning_rate=0.1,
                     min_child_weight=4, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, gamma=0)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
predict_y = model.predict(X_test)

# 计算评估指标
rmse = math.sqrt(mean_squared_error(y_test, predict_y))
mse = mean_squared_error(y_test, predict_y)
mae = mean_absolute_error(y_test, predict_y)
mape = np.mean(np.abs((y_test - predict_y) / y_test))
accuracy = 1 - mape

# 输出评估指标
print('RMSE:%.14f' % rmse)
print('MSE:%.14f' % mse)
print('MAE:%.14f' % mae)
print('MAPE:%.14f' % mape)
print('Accuracy:%.14f' % accuracy)
print("R2=", r2_score(y_test, predict_y))

# 绘制真实值和预测值的折线图
plt.plot(range(len(y_test)), y_test, color="blue", label='Actual')
plt.plot(range(len(y_test)), predict_y, color="red", label='Prediction')
plt.legend()
plt.show()

# SHAP 分析（可选，如果需要深入分析模型）
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
