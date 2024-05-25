import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import shap

# 定义计算平均绝对百分误差 (MAPE) 的函数
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 读取数据文件
file_path = r"C:\Users\7iCK\Desktop\10_features.csv"
df = pd.read_csv(file_path)

# 选择特征和目标变量
target = df.columns[1]  # 假设第二列为目标变量
cols_list = df.columns[2:]  # 假设从第三列开始到最后的列为特征变量

# 划分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(df[cols_list], df[target], test_size=0.2, random_state=100)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=99, max_features="sqrt", min_samples_leaf=5)
model.fit(x_train, y_train)

# 对测试集进行预测
predict_y = model.predict(x_test)

# 计算误差指标
rmse = math.sqrt(mean_squared_error(y_test, predict_y))
mse = mean_squared_error(y_test, predict_y)
mae = mean_absolute_error(y_test, predict_y)
mape = get_mape(y_test, predict_y)
accuracy = 1 - mape
r2 = r2_score(y_test, predict_y)

# 打印误差指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Accuracy: {accuracy}")
print(f"R-squared (R2): {r2}")

# SHAP值分析
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)

# 绘制SHAP特征重要性图，竖直排列
shap.summary_plot(shap_values, x_train, plot_type="bar", plot_size=(12, 8), color_bar=False)

# 绘制实际与预测的折线图
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Gas Prices', color='red')
plt.plot(predict_y, label='Predicted Gas Prices', color='blue')
plt.title('Actual vs Predicted Gas Prices')
plt.xlabel('Sample')
plt.ylabel('Gas Prices')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  # 将标注放在右上角
plt.grid(True)  # 添加网格线
plt.show()
