# --- coding: utf-8 ---
# @Time    : 3/5/25 11:57 PM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import numpy as np
import pandas as pd
import glob
import os

# 获取 RT 目录下所有 .npy 文件
files = glob.glob('RT/*.npy')

# 按文件名中的数字排序（提取文件名不带扩展名部分转为整数进行排序）
files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

all_data = []
for file in files:
    data = np.load(file)
    all_data.append(data)

# 将所有数据转换为 DataFrame
df_all = pd.DataFrame(all_data)

# 将前3列 x, y, z 分别乘以1000
# df_all.iloc[:, :3] = df_all.iloc[:, :3] * 1000

# 保存为 Excel 文件，不写入 header 和 index
df_all.to_excel('all_data.xlsx', index=False, header=False)

print("所有数据已成功保存为 all_data.xlsx")
