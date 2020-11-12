import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
sns 相关设置
@return:
"""
# 声明使用 Seaborn 样式
sns.set()
# 有五种seaborn的绘图风格，它们分别是：darkgrid, whitegrid, dark, white, ticks。默认的主题是darkgrid。
sns.set_style("whitegrid")
# 有四个预置的环境，按大小从小到大排列分别为：paper, notebook, talk, poster。其中，notebook是默认的。
sns.set_context('talk')
# 中文字体设置-黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 解决Seaborn中文显示问题并调整字体大小
sns.set(font='SimHei')


# reduce_mem_usage 函数通过调整数据类型，帮助我们减少数据在内存中占用的空间
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

"""读取数据"""
df_data = pd.read_csv("../data_yuanshi/train.csv", encoding='gbk')
df_data = reduce_mem_usage(df_data)

"""建立模型：【模型参数：xgb-->鱼佬baseline，lgb --> 贝叶斯调参】"""
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb


def xgb_model(X_train, y_train, X_test, y_test=None):
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_matrix = xgb.DMatrix(X_train_split, label=y_train_split)
    valid_matrix = xgb.DMatrix(X_val, label=y_val)
    test_matrix = xgb.DMatrix(X_test)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'gamma': 1,
        'min_child_weight': 1.5,
        'max_depth': 5,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'eta': 0.04,
        'tree_method': 'exact',
        'seed': 2020,
        'n_jobs': -1,
        "silent": True,
    }
    watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

    model = xgb.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                      early_stopping_rounds=200)
    """计算在验证集上的得分"""
    val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
    fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('调参后xgboost单模型在验证集上的AUC：{}'.format(roc_auc))
    """对测试集进行预测"""
    test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)

    return test_pred


def lgb_model(X_train, y_train, X_test, y_test=None):
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    # 调参后的最优参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'min_child_weight': 0.32,
        'num_leaves': 14,
        'max_depth': 4,
        'feature_fraction': 0.81,
        'bagging_fraction': 0.61,
        'bagging_freq': 9,
        'min_data_in_leaf': 13,
        'min_split_gain': 0.27,
        'reg_alpha': 9.58,
        'reg_lambda': 4.62,
        'seed': 2020,
        'n_jobs': -1,
        'silent': True,
        'verbose': -1,
    }

    model = lgb.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,
                      early_stopping_rounds=500)
    """计算在验证集上的得分"""
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('调参后lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
    """对测试集进行预测"""
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return test_pred


"""对训练集数据进行划分，分成训练集和验证集，并进行相应的操作"""
from sklearn.model_selection import train_test_split

"""数据集设置"""
X_train = df_data.loc[df_data['sample']=='train', :].drop(['id','issueDate','isDefault', 'sample'], axis=1)
X_test = df_data.loc[df_data['sample']=='test', :].drop(['id','issueDate','isDefault', 'sample'], axis=1)

y_train = df_data.loc[df_data['sample']=='train', 'isDefault']
# 数据集划分
# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)


from heamy.dataset import Dataset
from heamy.estimator import Classifier

model_dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test)
model_xgb = Classifier(dataset=model_dataset, estimator=xgb_model, name='xgb', use_cache=False)
model_lgb = Classifier(dataset=model_dataset, estimator=lgb_model, name='lgb', use_cache=False)

from heamy.pipeline import ModelsPipeline

pipeline = ModelsPipeline(model_xgb, model_lgb)
pipeline

# 构建第一层新特征，其中k默认是5，表示5折交叉验证，full_test=True，对全部训练集进行训练得到基学习器，然后用基学习器对测试集预测得到新特征
stack_ds = pipeline.stack(k=5, seed=111, full_test=True)

from sklearn.linear_model import LogisticRegression
# 第二层使用逻辑回归进行stack
LogisticRegression(solver='lbfgs')
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression, parameters={'solver': 'lbfgs'})
# 测试集的预测结果
test_pred = stacker.predict()
test_pred

"""生成提交格式的DataFrame"""
df_result = pd.DataFrame({'id': df_data.loc[df_data['sample']=='test', 'id'].values, 'isDefault': test_pred})
df_result.sort_values(by='id').head(20)


"""保存数据用于预测建模"""
df_result.to_csv('dataset/submission_data_stacking_model_20200924_V1_5folds.csv', encoding='gbk', index=False)

# 构建第一层新特征，将训练集切分成8:2，其中80%用于训练基学习器，20%用于构建新特征
blend_ds = pipeline.blend(proportion=0.2,seed=111)
# 第二层使用逻辑回归进行blend
blender = Classifier(dataset=blend_ds, estimator=LogisticRegression, parameters={'solver': 'lbfgs'})
# 测试集的预测结果
test_pred = blender.predict()
test_pred

"""生成提交格式的DataFrame"""
df_result = pd.DataFrame({'id': df_data.loc[df_data['sample']=='test', 'id'].values, 'isDefault': test_pred})
df_result.sort_values(by='id').head()


"""保存数据用于预测建模"""
df_result.to_csv('dataset/submission_data_blending_model_20200924_V1.csv', encoding='gbk', index=False)