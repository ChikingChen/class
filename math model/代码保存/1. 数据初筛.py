import os
from scipy.io import loadmat

def list_mat_variables(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                print(f"\n📂 File: {file_path}")

                try:
                    data = loadmat(file_path)
                    var_names = [k for k in data.keys() if not k.startswith("__")]
                    print("变量名 (scipy.io.loadmat):")
                    for v in var_names:
                        print("  -", v)
                except Exception as e2:
                    print(f"❌ 无法读取 {file_path}: {e2}")

# 使用示例（改成你的数据集路径）
list_mat_variables(r"C:\Users\root\Desktop\class\math model\dataset")
