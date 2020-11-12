import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
import lightgbm as lgb

# 文件目录，相对路径
INPUT_PATH = '../data_yuanshi/'
# OUTPUT_PATH = 'data/result/'

# 文件读取行数
MAX_ROWS = None

# 训练数据及label
train_data = pd.read_csv(INPUT_PATH + 'train.csv', index_col='id')
target = train_data.pop('isDefault')

# 测试数据
test_data = pd.read_csv(INPUT_PATH + 'testA.csv', index_col='id')

# 查看数据信息
train_data.info()

# 查看数据数据
train_data.head()

from sklearn.model_selection import train_test_split


# n2 与 n3 重复性较高，删除 n2
train_data.drop(['n2', 'issueDate'], axis = 1, inplace = True)
test_data = test_data[train_data.columns]


# 获取非数值列
s = train_data.dtypes
tecols = s[s=='object'].index.tolist()

# 将非数值列直接利用TargetEncoder进行离散化编码
te = TargetEncoder(cols=tecols)
tf = te.fit_transform(train_data, target)
df = te.transform(test_data)

# 划分训练集和验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(tf, target, test_size=0.2)
train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
valid_matrix = lgb.Dataset(X_val, label=y_val)

"""使用优化后的参数初始化模型（参数通过网格搜索法进行优化，这里没有相关代码）"""
base_params_lgb = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'learning_rate': 0.01,
                    'num_leaves': 14,
                    'max_depth': 19,
                    'min_data_in_leaf': 37,
                    'min_child_weight':1.6,
                    'bagging_fraction': 0.98,
                    'feature_fraction': 0.69,
                    'bagging_freq': 96,
                    'reg_lambda': 9,
                    'reg_alpha': 7,
                    'min_split_gain': 0.4,
                    'nthread': 8,
                    'seed': 2020,
                    'silent': True,
}

"""使用训练集数据进行模型训练"""
final_model_lgb = lgb.train(base_params_lgb, train_set=train_matrix, valid_sets=valid_matrix, num_boost_round=13000, verbose_eval=1000, early_stopping_rounds=800)
# pre_lgb = final_model_lgb.predict(X_test)
# # print(pre_lgb)
# # print(len(pre_lgb))
# # print(type(pre_lgb))
# fpr_lgb, tpr_lgb, thresholds_lgb = roc_curve(y_test, pre_lgb)
# score_lgb = auc(fpr_lgb, tpr_lgb)
# print("LGB的的得分：",score_lgb)


# 预测test，输出结果
OUTPUT_PATH = '../submit/'

df = te.transform(test_data)
pre = final_model_lgb.predict(df)

import time

pd.Series(pre, name='isDefault', index=test_data.index).reset_index().to_csv(OUTPUT_PATH + str(int(time.time()))
                                                                             +'_submit.csv', index=False)
