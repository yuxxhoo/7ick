"@A graduate student aiming to apply for a PhD"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def xgboost_model(file_path):
    # 读取数据
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 构造目标变量，预测次日价格
    df['Gas Close'] = df['Gas close'].shift(-1)
    X = df.drop(['Gas close'], axis=1).values
    y = df['Gas close'].values

    # 去除最后一行的缺失
    X = X[:-1]
    y = y[:-1]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化模型
    model = XGBRegressor(
        seed=100,
        n_estimators=115,
        max_depth=5,
        eval_metric='rmse',
        learning_rate=0.1,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        gamma=0
    )
    model.fit(X_train, y_train)

    # 预测并评估
    predict_y = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predict_y))
    mse = mean_squared_error(y_test, predict_y)
    mae = mean_absolute_error(y_test, predict_y)
    mape = np.mean(np.abs((y_test - predict_y) / y_test))
    accuracy = 1 - mape
    r2 = r2_score(y_test, predict_y)

    print('XGBoost模型评估结果：')
    print('RMSE:', rmse)
    print('MSE:', mse)
    print('MAE:', mae)
    print('MAPE:', mape)
    print('Accuracy:', accuracy)
    print(' R-squared:', r2)
    mse = mean_squared_error(y_test, predict_y)
    rmse = np.sqrt(mse)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'RMSE^2: {rmse ** 2}')

    # 可视化结果
    plt.plot(range(len(y_test)), y_test, color="blue", label='Actual')
    plt.plot(range(len(y_test)), predict_y, color="red", label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted（XGBoost）')
    plt.ylabel('Natural gas price')
    plt.tight_layout()
    plt.show()

# ===== 主程序入口 =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 错误：缺少 Excel 文件路径参数")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"✅ 接收到文件路径参数：{file_path}")
    xgboost_model(file_path)
