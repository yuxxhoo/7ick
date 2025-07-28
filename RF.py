"@A graduate student aiming to apply for a PhD"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import shap
import sys
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def run_random_forest(file_path):
    # 读取数据
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 创建目标变量（次日价格）
    df['Gas Close'] = df['Gas close'].shift(-1)
    df = df.dropna()

    # 构建特征和标签
    target = 'Gas close'
    cols_list = df.columns[df.columns != target]

    x_train, x_test, y_train, y_test = train_test_split(df[cols_list], df[target], test_size=0.2, random_state=100)

    # 初始化并训练模型
    model = RandomForestRegressor(
        n_estimators=1000,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        max_features="sqrt",
        min_samples_leaf=5
    )
    model.fit(x_train, y_train)

    # 预测
    predict_y = model.predict(x_test)

    # 评估
    rmse = math.sqrt(mean_squared_error(y_test, predict_y))
    mse = mean_squared_error(y_test, predict_y)
    mae = mean_absolute_error(y_test, predict_y)
    r2 = r2_score(y_test, predict_y)

    print('随机森林模型评估结果：')
    print('RMSE:', rmse)
    print('MSE:', mse)
    print('MAE:', mae)
    print(' R-squared:', r2)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_test)), y_test, color="blue", label='Actual')
    plt.plot(range(len(y_test)), predict_y, color="red", label='Predicted')
    plt.title('Actual vs Predicted（RF）')
    plt.xlabel('观测点')
    plt.ylabel('Natural gas price ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 预测下一天
    latest_data = df.iloc[-1:][cols_list]
    next_day_prediction = model.predict(latest_data)
    print("预测下一天的天然气价格：", next_day_prediction[0])

# ===== 主程序入口 =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 错误：缺少 Excel 文件路径参数")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"✅ 接收到文件路径参数：{file_path}")
    run_random_forest(file_path)
