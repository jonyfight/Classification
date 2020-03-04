"""
describe:
@project: 智能广告项目
@author: Jony
@create_time: 2019-07-09 12:21:10
@file: dd.py
"""

import re
from glob import glob
from itertools import chain
import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

def read_data_path(data_path,hidename):
    """
    内部功能函数：对文件夹路径下文件的进行搜索，并对其文件格式进行筛选
    :param data_path: 文件夹路径
    :param hidename: 文件筛选格式
    :return: 返回2个列表[filename][filepath]
    """
    #方式一
    #遍历文件夹内所有文件
    result = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            #进行扩展名筛选
            if os.path.splitext(filename)[1] == hidename :
                output = [dirpath , filename]
                result.append(output)
    return result

label_warp = {'others': 0,
              'points': 1,
              }

# train data
data_path = 'F:/data_images2/origin'  # data/origin
data = read_data_path(data_path,'.png')
img_path, label = [], []



# all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
# for i in tqdm(all_images):
for i in data:
    # image_folders = list(map(lambda x: i[0] + x, os.listdir(i[0])))
    i[0] = i[0].replace("\\","/")  # windows路径会有“\\”
    if i[0] == "F:/data_images2/origin/points":
        path = osp.join(i[0], i[1])  # 因为windows系统的路径问题，必须加一下三行
        path = path.replace("\\", "/")
        img_path.append((path))
    # if i[0] == 'data/origin/点点点':
    #     img_path.append(osp.join(i[0], i[1]))
        label.append('points')
    else:
        path = osp.join(i[0], i[1])
        path = path.replace("\\", "/")
        img_path.append((path))
        # img_path.append(osp.join(i[0], i[1]))
        label.append('others')


label_file = pd.DataFrame({'img_path': img_path, 'label': label})
label_file['label'] = label_file['label'].map(label_warp)

label_file.to_csv('./train_data.csv', index=False, encoding="utf8")  # encoding="GBK"才能编译中文路径名
# train_data = pd.read_csv('./train_data.csv').replace("\","/)