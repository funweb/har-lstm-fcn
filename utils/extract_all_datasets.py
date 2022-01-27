import os
import glob
import numpy as np
import pandas as pd
import shutil
from joblib import Parallel, delayed

path = '_data'

if not os.path.exists(path):
    os.makedirs(path)


def process_file(fn):
    file_name = os.path.split(fn)[-1]
    file_name = file_name[:-4]
    new_path = os.path.join(path, file_name)

    # Load the Tab seperated values in the dataset
    # 在数据集中加载选项卡分隔的值
    df = pd.read_table(fn, header=None, encoding='latin-1')

    # Fill the empty timesteps with 0.0
    # 用0.0填充空的时间步
    df.fillna(0.0, inplace=True)

    # Save the prepared dataset as a CSV file that the dataset reader can use
    # 将准备好的数据集另存为CSV文件，以便数据集读取器使用
    df.to_csv(new_path, sep=',', index=False, header=None, encoding='latin-1')

    # shutil.copy(fn, new_path)
    print("Copied file from %s to %s" % (fn, new_path))


with Parallel(n_jobs=-1, backend='loky', verbose=1) as engine:
    engine([delayed(process_file)(fn) for fn in glob.glob("*/*.tsv")])

print()
print("Extracted all files. Transfer all these files to the `data` directory")
