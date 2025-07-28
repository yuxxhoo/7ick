"@A graduate student aiming to apply for a PhD"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def train_and_predict(df, target_column, title_suffix):
    X = df.drop(['Gas close', 'Next Day Gas Close', 'Next Week Gas Close', 'Next Month Gas Close'], axis=1)
    y = df[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape

    print(f'{title_suffix} - Test MSE: {mse}')
    print(f'{title_suffix} - Test RMSE: {rmse}')
    print(f'{title_suffix} - MAE: {mae}')
    print(f'{title_suffix} - MAPE: {mape}%')
    print(f'{title_suffix} - Accuracy: {accuracy}%')
    print(f'{title_suffix} - R-squared: {r2}')

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(y_test):], y_test, label='Actual', color='blue', marker='o')
    plt.plot(df.index[-len(y_test):], y_pred, label='Predicted', color='red', marker='x')
    plt.title(f'Actual vs Predicted（{title_suffix}）')
    plt.xlabel('日期')
    plt.ylabel('Natural gas price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_svm_model(file_path):
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 构造目标列
    df['Next Day Gas Close'] = df['Gas close'].shift(-1)
    df['Next Week Gas Close'] = df['Gas close'].shift(-7)
    df['Next Month Gas Close'] = df['Gas close'].shift(-30)
    df.dropna(inplace=True)

    # 分别预测三个目标
    train_and_predict(df, 'Next Day Gas Close', '次日预测')
    train_and_predict(df, 'Next Week Gas Close', '次周预测')
    train_and_predict(df, 'Next Month Gas Close', '次月预测')

# ===== 主程序入口 =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 错误：缺少 Excel 文件路径参数")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"✅ 接收到文件路径参数：{file_path}")
    run_svm_model(file_path)
