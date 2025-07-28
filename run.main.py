"@A graduate student aiming to apply for a PhD"
import subprocess
import os
import pandas as pd
import re

# 数据路径列表
DATA_FILES = [
    ("测试1.xlsx", r"C:\Users\7iCK\Desktop\最终测试1.xlsx"),
    ("测试2.xlsx", r"C:\Users\7iCK\Desktop\最终测试2.xlsx")
]

# 模型脚本路径
RF_SCRIPT = r"D:\资料\python\机器学习\paper1\All code\LASSO.py"
SVM_SCRIPT = r"D:\资料\python\机器学习\paper1\All code\svm.py"
XGB_SCRIPT = r"D:\资料\python\机器学习\paper1\All code\XGBOOST.py"
MLR_SCRIPT = r"D:\资料\python\机器学习\paper1\All code\MLR.py"
LASSO_SCRIPT = r"D:\资料\python\机器学习\paper1\All code\Lasso.py"

# 正则提取指标值
def extract_metrics(output):
    metrics = {}
    patterns = {
        "RMSE": r"RMSE:\s*([0-9.]+)",
        "MSE": r"MSE:\s*([0-9.]+)",
        "MAE": r"MAE:\s*([0-9.]+)",
        "R2": r"R2=|R-squared:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = match.group(1)
        else:
            metrics[key] = "N/A"
    return metrics

# 路径检查
def check_file_exists(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"路径不存在: {path}")
            return False
    return True

# 模型执行
def run_model(script_path, file_path, model_name):
    print(f"------- Running {model_name} -------")
    result = subprocess.run(
        ['python', script_path, file_path],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    if result.stdout:
        print(result.stdout)
    else:
        print(f"{model_name} 没有标准输出。")

    if result.stderr:
        print(f"{model_name} 错误信息：\n{result.stderr}")

    return extract_metrics(result.stdout)

# 主流程
def main():
    all_results = []

    print("======= Step 1: 检查路径 =======")
    if not check_file_exists([RF_SCRIPT, SVM_SCRIPT, XGB_SCRIPT, MLR_SCRIPT, LASSO_SCRIPT] + [f[1] for f in DATA_FILES]):
        print("请检查路径是否正确。")
        return

    for name, file_path in DATA_FILES:
        print(f"\n======= 正在处理数据文件：{name} =======")
        try:
            df = pd.read_excel(file_path)
            print(f"{name} 预览前5行：")
            print(df.head())
        except Exception as e:
            print(f"读取文件出错：{e}")
            continue

        # 运行随机森林模型
        results = {
            "数据集": name,
            "模型": "Random Forest",
            **run_model(RF_SCRIPT, file_path, "Random Forest")
        }
        all_results.append(results)

        # 运行支持向量机模型
        results = {
            "数据集": name,
            "模型": "SVM",
            **run_model(SVM_SCRIPT, file_path, "Support Vector Machine")
        }
        all_results.append(results)

        # 运行XGBoost模型
        results = {
            "数据集": name,
            "模型": "XGBoost",
            **run_model(XGB_SCRIPT, file_path, "XGBoost")
        }
        all_results.append(results)

        # 运行多元线性回归（MLR）模型
        results = {
            "数据集": name,
            "模型": "MLR",  # 运行MLR模型
            **run_model(MLR_SCRIPT, file_path, "MLR")
        }
        all_results.append(results)

        # 运行Lasso模型
        results = {
            "数据集": name,
            "模型": "Lasso",  # 运行Lasso模型
            **run_model(LASSO_SCRIPT, file_path, "Lasso")
        }
        all_results.append(results)

    print("\n======= 对比分析结果 =======")
    df_result = pd.DataFrame(all_results)
    print(df_result.to_string(index=False))

if __name__ == "__main__":
    main()
