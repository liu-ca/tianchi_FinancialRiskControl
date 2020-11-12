#官方代码Task3，特征工程

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
warnings.filterwarnings('ignore')

data_test_a =pd.read_csv("../data_yuanshi/testA.csv")
data_train = pd.read_csv("../data_yuanshi/train.csv")

numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
label = 'isDefault'
if label in numerical_fea:
    numerical_fea.remove(label)

#按照平均数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].mean())
data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].mean())
#按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())

#转化成时间格式
for data in [data_train, data_test_a]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    #构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days

def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
for data in [data_train, data_test_a]:
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

for data in [data_train, data_test_a]:
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

# 部分类别特征
cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                 'applicationType', 'initialListStatus', 'title', 'policyCode']
for f in cate_features:
    print(f, '类型数：', data[f].nunique())

for data in [data_train, data_test_a]:
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})

# 类型数在2之上，又不是高维稀疏的,且纯分类特征
for data in [data_train, data_test_a]:
    data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'], drop_first=True)

#异常值检测：均方差
def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data


data_train = data_train.copy()
for fea in numerical_fea:
    data_train = find_outliers_by_3segama(data_train,fea)
    # print(data_train[fea+'_outliers'].value_counts())
    # print(data_train.groupby(fea+'_outliers')['isDefault'].sum())
    # print('*'*10)

#删除异常值
for fea in numerical_fea:
    data_train = data_train[data_train[fea+'_outliers']=='正常值']
    data_train = data_train.reset_index(drop=True)

#检测异常的方法二：箱型图，后期完善


#数据分桶（固定宽度分箱）
# 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
data['loanAmnt_bin1'] = np.floor_divide(data['loanAmnt'], 1000)
# 通过对数函数映射到指数宽度分箱
data['loanAmnt_bin2'] = np.floor(np.log10(data['loanAmnt']))

#数据分桶（分位数分桶）
data['loanAmnt_bin3'] = pd.qcut(data['loanAmnt'], 10, labels=False)

#数据分桶（卡方分箱及其他分箱方法的尝试）


#特征交互
for col in ['grade', 'subGrade']:
    temp_dict = data_train.groupby([col])['isDefault'].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col + '_target_mean'].to_dict()

    data_train[col + '_target_mean'] = data_train[col].map(temp_dict)
    data_test_a[col + '_target_mean'] = data_test_a[col].map(temp_dict)
# 其他衍生变量 mean 和 std
for df in [data_train, data_test_a]:
    for item in ['n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']:
    # for item in ['n0','n1','n2','n2.1','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14']:
        df['grade_to_mean_' + item] = df['grade'] / df.groupby([item])['grade'].transform('mean')
        df['grade_to_std_' + item] = df['grade'] / df.groupby([item])['grade'].transform('std')

#特征编码
#label-encode:subGrade,postCode,title
# 高维类别特征需要进行转换
for col in tqdm(['employmentTitle', 'postCode', 'title','subGrade']):
    le = LabelEncoder()
    le.fit(list(data_train[col].astype(str).values) + list(data_test_a[col].astype(str).values))
    data_train[col] = le.transform(list(data_train[col].astype(str).values))
    data_test_a[col] = le.transform(list(data_test_a[col].astype(str).values))
print('Label Encoding 完成')

# 举例归一化过程
#伪代码
# for fea in [要归一化的特征列表]：
#     data[fea] = ((data[fea] - np.min(data[fea])) / (np.max(data[fea]) - np.min(data[fea])))


#特征选择（方差选择）
# from sklearn.feature_selection import VarianceThreshold
# #其中参数threshold为方差的阈值
# VarianceThreshold(threshold=3).fit_transform(train,target_train)

#特征选择（相关系数法）
# from sklearn.feature_selection import SelectKBest
# from scipy.stats import pearsonr
# #选择K个最好的特征，返回选择特征后的数据
# #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
# #输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# #参数k为选择的特征个数
# SelectKBest(k=5).fit_transform(train,target_train)


#特征选择（卡方检验）
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# #参数k为选择的特征个数
# SelectKBest(chi2, k=5).fit_transform(train,target_train)

#特征选择（互信息法）
# from sklearn.feature_selection import SelectKBest
# from minepy import MINE
# #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，
# #返回一个二元组，二元组的第2项设置成固定的P值0.5
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
# #参数k为选择的特征个数
# SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(train,target_train)


#特征选择（Wrapper: Recursive feature elimination，RFE）
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# #递归特征消除法，返回特征选择后的数据
# #参数estimator为基模型
# #参数n_features_to_select为选择的特征个数
# RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(train,target_train)

#特征选择（Embedded: 基于惩罚项的特征选择法)
# from sklearn.feature_selection import SelectFromModel
# # from sklearn.linear_model import LogisticRegression
# # #带L1惩罚项的逻辑回归作为基模型的特征选择
# # SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(train,target_train)

#特征选择（Embedded: 基于树模型的特征选择)
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import GradientBoostingClassifier
# #GBDT作为基模型的特征选择
# SelectFromModel(GradientBoostingClassifier()).fit_transform(train,target_train)

# 删除不需要的数据
for data in [data_train, data_test_a]:
    data.drop(['issueDate','id'], axis=1,inplace=True)

# "纵向用缺失值上面的值替换缺失值"
data_train = data_train.fillna(axis=0,method='ffill')

# x_train = data_train.drop(['isDefault','id'], axis=1)
x_train = data_train
#计算协方差
# data_corr = x_train.corrwith(data_train.isDefault) #计算相关性
# result = pd.DataFrame(columns=['features', 'corr'])
# result['features'] = data_corr.index
# result['corr'] = data_corr.values

# 当然也可以直接看图
# data_numeric = data_train[numerical_fea]
# correlation = data_numeric.corr()
# f , ax = plt.subplots(figsize = (7, 7))
# plt.title('Correlation of Numeric Features with Price',y=1,size=16)
# sns.heatmap(correlation,square = True,  vmax=0.8)

features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']


def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
                                     train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs': 24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,
                              early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)

            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)

            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma':   0.2,#1,
                      'min_child_weight': 5,#1.5,
                      'max_depth': 4,
                      'lambda': 10,
                      'subsample': 0.8,#0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 0,#2020,
                      'nthread': 36,
                      "silent": True,

                      "reg_alpha": 3,
                      "reg_lambda": 3,
                      "random_state": 27,
                      }

            watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                              early_stopping_rounds=600)
            val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_x, ntree_limit=model.best_ntree_limit)

        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)

            val_pred = model.predict(val_x)
            test_pred = model.predict(test_x)

        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))

        print(cv_scores)

    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test


def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test

def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")


# lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)

testA_result = pd.read_csv('../testA_result.csv')

roc_auc_score(testA_result['isDefault'].values, xgb_test)

pd.DataFrame({'isDefault':xgb_test},index=data_test_a.index).reset_index().rename(columns={"index":"id"}).to_csv('../submit/submit.csv', index=0)
submit = pd.read_csv("../submit/submit.csv")
submit['id'] = [x+800000 for x in range(200000)]
print(submit['id'])
submit.to_csv("../submit/submit.csv", index=None)