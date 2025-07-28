"@A graduate student aiming to apply for a PhD"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
def run_linear_regression(file_path):
    # 读取数据
    df = pd.read_excel(file_path)

    # 确保日期列正确处理
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 创建滞后特征以进行次日预测
    features = df.drop(columns='Gas close').shift(1)  # 除去目标变量的其他所有列滞后一天
    target = df['Gas close']

    # 删除由于滞后操作引入的NaN值
    features = features[1:]  # 删除第一行的NaN
    target = target[1:]      # 同步删除目标变量的第一行

    X = features.values
    y = target.values

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

    # 打印系数和截距
    coefficients = model.coef_  # 获取线性回归模型的系数
    intercept = model.intercept_  # 获取截距
    print("Coefficients:")
    for feature, coef in zip(features.columns, coefficients):
        print(f"{feature}: {coef}")
    print(f"Intercept: {intercept}")

    # 绘制预测与实际值的散点图
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel("Actual Natural Gas Prices")
    plt.ylabel("Predicted Natural Gas Prices")
    plt.title("Next Day Natural Gas Prices Prediction (MLR)")
    plt.show()

# ===== 主程序入口 =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 错误：缺少 Excel 文件路径参数")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"✅ 接收到文件路径参数：{file_path}")
    run_linear_regression(file_path)
