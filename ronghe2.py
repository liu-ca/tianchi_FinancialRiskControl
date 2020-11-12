import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.base as base
from sklearn.model_selection import KFold
import warnings
import os
warnings.filterwarnings('ignore')
import sklearn

test = pd.read_csv("../data_yuanshi/testA.csv")
train = pd.read_csv("../data_yuanshi/train.csv")
train.drop("id", axis= 1,inplace = True)
train.head()
train.info(verbose = True)
train.describe()

numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))
label = 'isDefault'
if label in numerical_fea:
    numerical_fea.remove(label)

# #按照平均数填充数值型特征
# train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].mean())
# test[numerical_fea] = test[numerical_fea].fillna(train[numerical_fea].mean())
#按照众数填充类别型特征
train[category_fea] = train[category_fea].fillna(train[category_fea].mode())
test[category_fea] = test[category_fea].fillna(train[category_fea].mode())

def transform_issueDate(df):
    df['issueDate'] = pd.to_datetime(df['issueDate'],format='%Y-%m-%d')
    startdate = datetime.strptime('2007-06-01', '%Y-%m-%d')
    df['issueDateDT'] = df['issueDate'].apply(lambda x: x-startdate).dt.days
    return df
train = transform_issueDate(train)
test = transform_issueDate(test)
# plt.hist(train['issueDateDT'],label = "train")
# plt.hist(test['issueDateDT'], label = "test")

def transform_earliesCreditLine(df):
    df['earliesCreditLine_Year'] = df['earliesCreditLine'].apply(lambda x: 2020-int(x[-4:]))
    return df
train =  transform_earliesCreditLine(train)
test =  transform_earliesCreditLine(test)
# plt.hist(train['earliesCreditLine_Year'],label = "train")
# plt.hist(test['earliesCreditLine_Year'],label = "test")

train_positve = train[train['isDefault'] == 1]
train_negative = train[train['isDefault'] != 1]
# f, ax = plt.subplots(len(numerical_cols),2,figsize = (10,80))
# for i,col in enumerate(numerical_cols):
#     sns.distplot(train_positve[col],ax = ax[i,0],color = "blue")
#     ax[i,0].set_title("positive")
#     sns.distplot(train_negative[col],ax = ax[i,1],color = 'red')
#     ax[i,1].set_title("negative")
# plt.subplots_adjust(hspace = 1)


# is_null_index = train['n10'].isnull()
# for col in train.columns:
#     if train[col][is_null_index].notnull().sum() == 0:
        # print(col)

# is_null_index = train['n1'].isnull()
# for col in train.columns:
#     if train[col][is_null_index].notnull().sum() == 0:
        # print(col)

for colname in ['grade',"subGrade"]:
    unique_num = train.append(test)[colname].nunique()
    unuque_val = sorted(train.append(test)[colname].unique())
    for data in [train,test]:
        map_dict = {x:y for x,y in zip(unuque_val,range(unique_num))}
        data[colname] = data[colname].map(map_dict)

train['employmentLength'].unique()

def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
for data in [train, test]:
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

# for data in [train,test]:
#     data['employmentLength'].replace("< 1 year", "0 year", inplace=True)
#     data['employmentLength'].replace("10+ years", "10 years", inplace=True)
#     data['employmentLength'] = data['employmentLength'].apply(lambda x: int(str(x).split()[0]) if pd.notnull(x) else x)

# 部分类别特征
cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                 'applicationType', 'initialListStatus', 'title', 'policyCode']
# for f in cate_features:
#     print(f, '类型数：', data[f].nunique())

for data in [train, test]:
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})

# 类型数在2之上，又不是高维稀疏的,且纯分类特征
for data in [train, test]:
    data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'], drop_first=True)


# #固定宽度分箱
# # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
# data['loanAmnt_bin1'] = np.floor_divide(data['loanAmnt'], 1000)
# ## 通过对数函数映射到指数宽度分箱
# data['loanAmnt_bin2'] = np.floor(np.log10(data['loanAmnt']))
#
# #分位数分箱
# data['loanAmnt_bin3'] = pd.qcut(data['loanAmnt'], 10, labels=False)

#异常值检测：均方差


#eda得出的结果
discrete_fea = ['term','grade','subGrade','employmentLength','homeOwnership','verificationStatus',
               'purpose','regionCode','delinquency_2years','ficoRangeLow','ficoRangeHigh','pubRec',
                'initialListStatus','applicationType'
               ]
continuous_fea = ['loanAmnt','interestRate','installment','annualIncome','issueDate','dti','openAcc',
                 'revolBal','revolUtil','totalAcc','title']

# # 高维类别特征需要进行转换
# for data in [train, test]:
#     for f in ['employmentTitle', 'postCode', 'title']:
#         data[f+'_cnts'] = data.groupby([f])['id'].transform('count')
#         data[f+'_rank'] = data.groupby([f])['id'].rank(ascending=False).astype(int)
#         del data[f]

# Create Features based on anonymised prefix groups
for data in [train, test]:
    prefix = ['grade', 'n','interestRate']
    for i, p in enumerate(prefix):
        column_set = [x for x in data.columns.tolist() if x.startswith(prefix[i])]

        # Take NA count
        data[p + "group_nan_sum"] = data[column_set].isnull().sum(axis=1) / data[column_set].shape[1]

        # Take SUM/Mean if numeric
        numeric_cols = [x for x in column_set if data[x].dtype != object]
        if numeric_cols:
            data[p + "group_sum"] = data[column_set].sum(axis=1)
            data[p + "group_mean"] = data[column_set].mean(axis=1)
            # Zero Count
            data[p + "group_0_count"] = (data[column_set] == 0).astype(int).sum(axis=1) / (
                    data[column_set].shape[1] - data[p + "group_nan_sum"])


# New feature - log of transaction amount. ()
train['loanAmnt_Log'] = np.log(train['loanAmnt'])
test['loanAmnt_Log'] = np.log(test['loanAmnt'])

for col in ['grade', 'subGrade','pubRec']:  #分类数据
    temp_dict = train.groupby([col])['isDefault'].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col + '_target_mean'].to_dict()

    train[col + '_target_mean'] = train[col].map(temp_dict)
    test[col + '_target_mean'] = test[col].map(temp_dict)


for feature in ['ficoRangeLow__applicationType', 'ficoRangeLow__ficoRangeHigh']:
    f1, f2 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

for data in [train, test]:
    #贷款金额/分期付款金额 = 贷款期限
    data['loanTerm'] = data['loanAmnt'] / data['installment']
    #手动分箱
    data['pubRec'] = data['pubRec'].apply(lambda x: 7.0 if x >= 7.0 else x)
    data['pubRecBankruptcies'] = data['pubRecBankruptcies'].apply(lambda x: 7.0 if x >= 7.0 else x)

#feature interaction
for grad in ['grade_homeOwnership', 'grade_term', 'grade_verificationStatus',
             'grade_purpose', 'grade_regionCode']:
    card1 = grad.split('_')[0]
    card2 = grad.split('_')[1]
    train[grad] = train[card1].astype(str) + '_' + train[card2].astype(str)
    test[grad] = test[card1].astype(str) + '_' + test[card2].astype(str)

    train[grad + '_amt_mean'] = train[grad].map(
        (pd.concat([train[[grad, 'loanAmnt']], test[[grad, 'loanAmnt']]], ignore_index=True)).groupby(
            [grad])['loanAmnt'].mean())
    test[grad + '_amt_mean'] = test[grad].map(
        (pd.concat([train[[grad, 'loanAmnt']], test[[grad, 'loanAmnt']]], ignore_index=True)).groupby(
            [grad])['loanAmnt'].mean())

    train[grad + '_amt_std'] = train[grad].map(
        (pd.concat([train[[grad, 'loanAmnt']], test[[grad, 'loanAmnt']]], ignore_index=True)).groupby(
            [grad])['loanAmnt'].std())
    test[grad + '_amt_std'] = test[grad].map(
        (pd.concat([train[[grad, 'loanAmnt']], test[[grad, 'loanAmnt']]], ignore_index=True)).groupby(
            [grad])['loanAmnt'].std())



#决策树分箱
from sklearn.tree import DecisionTreeClassifier

def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()

    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    return boundary

for fea in continuous_fea:
    if fea == 'issueDate':
        train[fea] = train[fea].values.astype(int)
        test[fea] = test[fea].values.astype(int)
    boundary = optimal_binning_boundary(x=train[fea],y=train['isDefault'])
    train[fea + 'bins'] = pd.cut(train[fea], bins=boundary, labels=False)
    test[fea + 'bins'] = pd.cut(test[fea], bins=boundary, labels=False)

def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data

# print(numerical_fea)
data_train = train.copy()
for fea in numerical_fea:
    data_train = find_outliers_by_3segama(data_train,fea)
    # print(data_train[fea+'_outliers'].value_counts())
    # print(data_train.groupby(fea+'_outliers')['isDefault'].sum())
    # print('*'*10)

#删除异常值
for fea in numerical_fea:
    data_train = data_train[data_train[fea+'_outliers']=='正常值']
    data_train = data_train.reset_index(drop=True)


#特征交互
# for col in ['grade', 'subGrade']:
#     temp_dict = train.groupby([col])['isDefault'].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
#     temp_dict.index = temp_dict[col].values
#     temp_dict = temp_dict[col + '_target_mean'].to_dict()
#
#     train[col + '_target_mean'] = train[col].map(temp_dict)
#     test[col + '_target_mean'] = test[col].map(temp_dict)
# # 其他衍生变量 mean 和 std
# for df in [train, test]:
#     for item in ['n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']:
#     # for item in ['n0','n1','n2','n2.1','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14']:
#         df['grade_to_mean_' + item] = df['grade'] / df.groupby([item])['grade'].transform('mean')
#         df['grade_to_std_' + item] = df['grade'] / df.groupby([item])['grade'].transform('std')


import numpy as np


class ChiMerge():
    def __init__(self, df, col_name, target):
        self.num_bins = df[col_name].nunique()
        self.sorted_df = df.sort_values(by=col_name)[[target, col_name]]
        self.target = target
        self.unique_val = np.sort(df[col_name].unique())
        self.col_name = col_name
        self.reverse = 1
        self.shape = df.shape[0]

    def check_max_and_min_bin(self, to_merge_df):
        max_bin = to_merge_df[self.col_name].value_counts().values[0]

        min_bin = to_merge_df[self.col_name].value_counts().values[-1]

        return max_bin / self.shape, min_bin / self.shape

    def cal_Chi2(self, bin1, bin2, epsilon=1e-8):
        # 计算单个两个箱体的卡方值，加入epsilon为了防止除0错误

        bins = bin1.append(bin2)
        total = bins.shape[0]
        positive_rate = bins[self.target].sum() / total
        negative_rate = 1 - positive_rate

        chi2_val = (bin1[self.target].sum() - positive_rate * bin1.shape[0]) ** 2 / (
                    positive_rate * bin1.shape[0] + epsilon) + \
                   (bin2[self.target].sum() - positive_rate * bin2.shape[0]) ** 2 / (
                               positive_rate * bin2.shape[0] + epsilon) + \
                   (bin1.shape[0] - bin1[self.target].sum() - negative_rate * bin1.shape[0]) ** 2 / (
                               negative_rate * bin1.shape[0] + epsilon) + \
                   (bin2.shape[0] - bin2[self.target].sum() - negative_rate * bin2.shape[0]) ** 2 / (
                               negative_rate * bin2.shape[0] + epsilon)

        return chi2_val

    def calculate_every_Chi2(self):

        chi2_list = []
        if self.reverse == 1:
            # 如果数值较多的时候可能会出现很多卡方为0的箱，为了减少次数，两头循坏，避免全列表遍历
            # 水平较低，想暂时使用这个方法减少分箱时间
            for i in range(self.num_bins - 1):
                chi2 = self.cal_Chi2(self.sorted_df[self.sorted_df[self.col_name] == self.unique_val[i]],
                                     self.sorted_df[self.sorted_df[self.col_name] == self.unique_val[i + 1]])
                chi2_list.append(chi2)
                if chi2 == 0:
                    break
        else:
            for i in range(self.num_bins - 1, 0, self.reverse):
                chi2 = self.cal_Chi2(self.sorted_df[self.sorted_df[self.col_name] == self.unique_val[i]],
                                     self.sorted_df[self.sorted_df[self.col_name] == self.unique_val[i + 1]])
                chi2_list.append(chi2)
                if chi2 == 0:
                    break

        self.reverse = self.reverse * (-1)

        return chi2_list

    def chi2Merge(self, chi2_val):

        max_bin, min_bin = self.check_max_and_min_bin(self.sorted_df)

        if max_bin > 0.95:
            print("The max bin has more than 95% of samples")
            return self.sorted_df
        # 先初次判断，如果初始数据已经有箱体过大的情况，无法分箱

        chi2_list = [0]
        while self.num_bins > 5 and min(chi2_list) < chi2_val:

            remove_flag = True
            chi2_list = self.calculate_every_Chi2()
            unique_val = self.unique_val

            while remove_flag:
                to_merge = np.argmin(chi2_list)
                to_merge_df = self.sorted_df
                to_merge_df[self.col_name][to_merge_df[self.col_name] == unique_val[to_merge]] = unique_val[
                    to_merge + 1]
                max_bin, min_bin = self.check_max_and_min_bin(to_merge_df)
                if max_bin > 0.95:
                    chi2_list.pop(to_merge)
                    unique_val.pop(to_merge)
                else:
                    remove_flag = False
                    self.unique_val = unique_val

            self.sorted_df[self.col_name][self.sorted_df[self.col_name] == self.unique_val[to_merge]] = self.unique_val[
                to_merge + 1]
            self.unique_val = np.sort(self.sorted_df[self.col_name].unique())
            self.num_bins -= 1
            if self.num_bins % 1000 == 0:
                print(self.num_bins)
        _, min_bin = self.check_max_and_min_bin(self.sorted_df)
        if min_bin < 0.05:
            print("too small bin")


class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames, targetName, n_fold=5, verbosity=True, discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        assert (type(self.targetName) == str)
        assert (type(self.colnames) == str)
        assert (self.colnames in X.columns)
        assert (self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=False)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            #             print(tr_ind,val_ind)
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean())

        X[col_mean_name].fillna(mean_of_target, inplace=True)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                  self.targetName,
                                                                                  np.corrcoef(X[self.targetName].values,
                                                                                              encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.colnames, axis=1)

        return X


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, train, colNames, encodedName):
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index()

        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X


for colname in ['purpose',"verificationStatus", "regionCode","grade","subGrade"]:
    targetc = KFoldTargetEncoderTrain(colname,'isDefault',n_fold=5)
    train = targetc.fit_transform(train)
    test_targetc  = KFoldTargetEncoderTest(train,colname,colname + '_' + 'Kfold_Target_Enc')
    test = test_targetc.fit_transform(test)


for data in [train,test]:
    data['extra_col1'] = 3
    data['extra_col1'].loc[data['n10'].isnull()] = 1
    data['extra_col1'].loc[data['n1'].isnull() & data['n10'].notnull()] = 2
    data['extra_col1'].loc[data['n11'].isnull() & data['n1'].notnull()] = 4
for i in range(1,5):
    print(train[train['extra_col1']==i]['isDefault'].sum()/train[train['extra_col1']==i]['isDefault'].count())


def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data
data_train = train.copy()
# for fea in numerical_fea:
#     data_train = find_outliers_by_3segama(data_train,fea)
#     print(data_train[fea+'_outliers'].value_counts())
#     print(data_train.groupby(fea+'_outliers')['isDefault'].sum())
#     print('*'*10)

# #one-hot编码
# for data in [train, test]:
#     data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'], drop_first=True)



########################### Freq encoding
discrete_fea = ['term','grade','subGrade','employmentLength','homeOwnership','verificationStatus',
               'purpose','regionCode','delinquency_2years','ficoRangeLow','ficoRangeHigh','pubRec',
                'initialListStatus','applicationType'
               ]
for col in discrete_fea:
    temp_df = pd.concat([train[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
    train[col + '_fq_enc'] = train[col].map(fq_encode)
    test[col + '_fq_enc'] = test[col].map(fq_encode)
########################### Freq encoding


#########################label-encode   精度下降了
# from tqdm import tqdm_notebook
# from sklearn.preprocessing import LabelEncoder
# for col in tqdm_notebook(category_fea):
#     le = LabelEncoder()
#     le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
#     train[col] = le.transform(list(train[col].astype(str).values))
#     test[col] = le.transform(list(test[col].astype(str).values))
# print('Label Encoding 完成')

#######################WOE编码
# 获得某个变量各个分箱的WOE、IV值函数
def feature_woe_iv(x: pd.Series, y: pd.Series, nan: float = -999.) -> pd.DataFrame:
    '''
        计算变量各个分箱的WOE、IV值，返回一个DataFrame
    '''
    x = x.fillna(nan)
    boundary = optimal_binning_boundary(x, y, nan)  # 获得最优分箱边界值列表
    df = pd.concat([x, y], axis=1)  # 合并x、y为一个DataFrame，方便后续计算
    df.columns = ['x', 'y']  # 特征变量、目标变量字段的重命名
    df['bins'] = pd.cut(x=x, bins=boundary, right=False)  # 获得每个x值所在的分箱区间

    grouped = df.groupby('bins')['y']  # 统计各分箱区间的好、坏、总客户数量
    result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                             ('bad', lambda y: (y == 1).sum()),
                             ('total', 'count')])

    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()  # 好客户占比
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()  # 坏客户占比
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()  # 总客户占比

    result_df['bad_rate'] = result_df['bad'] / result_df['total']  # 坏比率

    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])  # WOE
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV

    # print(f"该变量IV = {result_df['iv'].sum()}")

    return result_df

# for fea in continuous_fea:
#     if fea == 'issueDate':
#         train[fea] = train[fea].values.astype(int)
#         test[fea] = test[fea].values.astype(int)
#     result_df = feature_woe_iv(x=train[fea],y=train['isDefault'])
    # train[fea + 'woe'] = pd.cut(train[fea], bins=boundary, labels=False)
    # test[fea + 'woe'] = pd.cut(test[fea], bins=boundary, labels=False)


target = train['isDefault']
train_X = train.drop("isDefault", axis=1)
##############特征选择
###相关系数法（pearsonr相关系数）
# from sklearn.feature_selection import SelectKBest
# from scipy.stats import pearsonr
#   #选择K个最好的特征，返回选择特征后的数据
#   #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
#   #输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#   #参数k为选择的特征个数
# SelectKBest(k=5).fit_transform(train,target)

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import auc, roc_curve
from xgboost import plot_importance
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn import preprocessing


target = train['isDefault']
train_X = train.drop("isDefault", axis=1)
#切分训练和检验集
X_train,X_test,y_train,y_test = train_test_split(train_X, target,test_size = 0.2, random_state = 0)


# def XGB():
#     model = XGBClassifier(learning_rate=0.1,
#                           n_estimators=600,
#                           max_depth=5,
#                           min_child_weight=5,
#                           gamma=1,
#                           subsample=0.8,
#                           random_state=27,
#                           verbosity=1,
#                           nthread=-1
#                           )
#     return model

# # %%time
# model = XGB()

lbl = preprocessing.LabelEncoder()
X_train['issueDate'] = lbl.fit_transform(X_train['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
# y_train['issueDate'] = lbl.fit_transform(y_train['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
X_test['issueDate'] = lbl.fit_transform(X_test['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
# y_test['issueDate'] = lbl.fit_transform(y_test['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
X_train['ficoRangeLow__applicationType'] = lbl.fit_transform(X_train['ficoRangeLow__applicationType'].astype(str))#将提示的包含错误数据类型这一列进行转换
X_test['ficoRangeLow__applicationType'] = lbl.fit_transform(X_test['ficoRangeLow__applicationType'].astype(str))#将提示的包含错误数据类型这一列进行转换

X_train['ficoRangeLow__ficoRangeHigh'] = lbl.fit_transform(X_train['ficoRangeLow__ficoRangeHigh'].astype(str))#将提示的包含错误数据类型这一列进行转换
X_test['ficoRangeLow__ficoRangeHigh'] = lbl.fit_transform(X_test['ficoRangeLow__ficoRangeHigh'].astype(str))#将提示的包含错误数据类型这一列进行转换

X_train['grade_homeOwnership'] = lbl.fit_transform(X_train['grade_homeOwnership'].astype(str))#将提示的包含错误数据类型这一列进行转换
X_test['grade_homeOwnership'] = lbl.fit_transform(X_test['grade_homeOwnership'].astype(str))#

X_train['grade_term'] = lbl.fit_transform(X_train['grade_term'].astype(str))
X_test['grade_term'] = lbl.fit_transform(X_test['grade_term'].astype(str))

X_train['grade_verificationStatus'] = lbl.fit_transform(X_train['grade_verificationStatus'].astype(str))
X_test['grade_verificationStatus'] = lbl.fit_transform(X_test['grade_verificationStatus'].astype(str))

X_train['grade_purpose'] = lbl.fit_transform(X_train['grade_purpose'].astype(str))
X_test['grade_purpose'] = lbl.fit_transform(X_test['grade_purpose'].astype(str))

X_train['grade_regionCode'] = lbl.fit_transform(X_train['grade_regionCode'].astype(str))
X_test['grade_regionCode'] = lbl.fit_transform(X_test['grade_regionCode'].astype(str))


test['ficoRangeLow__applicationType'] = lbl.fit_transform(test['ficoRangeLow__applicationType'].astype(str))#将提示的包含错误数据类型这一列进行转换
test['ficoRangeLow__ficoRangeHigh'] = lbl.fit_transform(test['ficoRangeLow__ficoRangeHigh'].astype(str))#将提示的包含错误数据类型这一列进行转换
test['grade_homeOwnership'] = lbl.fit_transform(test['grade_homeOwnership'].astype(str))#
test['grade_term'] = lbl.fit_transform(test['grade_term'].astype(str))
test['grade_verificationStatus'] = lbl.fit_transform(test['grade_verificationStatus'].astype(str))
test['grade_purpose'] = lbl.fit_transform(test['grade_purpose'].astype(str))
test['grade_regionCode'] = lbl.fit_transform(test['grade_regionCode'].astype(str))

X_train['earliesCreditLine'] = lbl.fit_transform(X_train['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换
# y_train['earliesCreditLine'] = lbl.fit_transform(y_train['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换
X_test['earliesCreditLine'] = lbl.fit_transform(X_test['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换
# y_test['earliesCreditLine'] = lbl.fit_transform(y_test['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换


# model.fit(X_train, y_train, eval_set = [(X_train,y_train),(X_test,y_test)],eval_metric="auc")
# result = model.evals_result()
#
#
# pre = model.predict_proba(X_train)[:,1]
# fpr, tpr, thresholds = roc_curve(y_train, pre)
# score = auc(fpr, tpr)
#
# f,[ax1,ax2] = plt.subplots(2,1,figsize = (7,15))
#
# ax1.plot([i for i in range(1,600+1)],result['validation_0']['auc'])
# ax1.plot([i for i in range(1,600+1)],result['validation_1']['auc'])
# ax2.set_xlim(0,1)
# ax2.set_ylim(0,1)
# ax2.plot(fpr,tpr,label = "AUC = {:.3f}".format(score))
# ax2.plot([0,1],[0,1],linestyle = "--")
# plt.legend()

# 在定义以下基准模型
# model = XGBClassifier(
#     learning_rate=0.1,
#     n_estimators=300,
#     max_depth=5,
#     min_child_weight=6,
#     gamma=1,
#     subsample=0.8,
#     scale_pos_weight=4,
#     random_state=27,
#     verbosity= 1,
#     nthread=-1
# )


# param_grid = {
# 	"max_depth":[i for i in range(3, 11)]
# }
# xgb_grid = GridSearchCV(
# 	model,
# 	param_grid = param_grid,
# 	scoring= "roc_auc", #"这次评价标准的auc"
# 	verbose=True, #"输出过程"
# 	cv=5, #"5折检验"
# 	n_jobs=-1, #"使用所有CPU"
# )
# xgb_grid.fit(X_train, y_train)
# xgb_grid.best_params_
# # 5
#
# param_grid = {
# 	"min_child_weight":[i for i in range(3, 11)]
# }
# xgb_grid = GridSearchCV(
# 	model,
# 	param_grid = param_grid,
# 	scoring= "roc_auc",#"roc_auc", "这次评价标准的auc"
# 	verbose=True, #"输出过程"
# 	cv=5, #"5折检验"
# 	n_jobs=-1 #"使用所有CPU"
# 	)
# xgb_grid.fit(X_train, y_train)
# xgb_grid.best_params_
# # 6
#
#
# param_grid = {
# 	"subsampele":[i*0.1 for i in range(3, 11)]
# }
# xgb_grid = GridSearchCV(
# 	model,
# 	param_grid = param_grid,
# 	scoring= "roc_auc", #"这次评价标准的auc"
# 	verbose=True, #"输出过程"
# 	cv=5, #"5折检验"
# 	n_jobs=-1 #"使用所有CPU"
# 	)
# xgb_grid.fit(X_train, y_train)
# xgb_grid.best_params_
# # # 0.6

import lightgbm as lgb
# params_test2 = {'max_bin': range(45, 75, 10), 'min_data_in_leaf': range(121, 172, 10)}
#
# gsearch2 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=508, max_depth=5, num_leaves=30, bagging_fraction=0.8,
#                                  feature_fraction=0.8),
#     param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch2.fit(X_train, y_train)
# print(gsearch2.cv_results_)
# print(gsearch2.best_params_)    #max_bin=25, min_data_in_leaf=101        45, 131, 0.7352214697873584    55, 151, 0.7353630299274441
# print(gsearch2.best_score_)

# params_test3 = {'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
#                 'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
#                 'bagging_freq': range(0, 81, 10)
#                 }

# gsearch3 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=508, max_depth=5, num_leaves=30, max_bin=55, min_data_in_leaf=151),
#     param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch3.fit(X_train, y_train)
# # gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
# print(gsearch3.cv_results_)     #{'bagging_fraction': 0.9, 'bagging_freq': 10, 'feature_fraction': 0.8} 0.7354771484428861
# print(gsearch3.best_params_)
# print(gsearch3.best_score_)

# params_test4 = {'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
#                 'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
#                 }
# 
# gsearch4 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=508, max_depth=5, num_leaves=30, max_bin=55, min_data_in_leaf=151,
#                                  bagging_fraction=0.9, bagging_freq=10, feature_fraction=0.8),
#     param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch4.fit(X_train, y_train)
# print(gsearch4.cv_results_)
# print(gsearch4.best_params_)    {'lambda_l1': 0.9, 'lambda_l2': 0.9}   # 0.7356366812496242
# print(gsearch4.best_score_)


# params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
#
# gsearch5 = GridSearchCV(
#     estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
#                                  n_estimators=508, max_depth=5, num_leaves=30, max_bin=55, min_data_in_leaf=151,
#                                  bagging_fraction=0.9, bagging_freq=10, feature_fraction=0.8,
#                                  lambda_l1=0.9, lambda_l2=0.9),
#     param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1)
# gsearch5.fit(X_train, y_train)
#
# print(gsearch5.cv_results_)
# print(gsearch5.best_params_)    #{'min_split_gain': 0.0} 0.7356366812496242
# print(gsearch5.best_score_)

# import sklearn
# model=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=5, num_leaves=30,max_bin=55,min_data_in_leaf=151,bagging_fraction=0.9,bagging_freq= 10, feature_fraction= 0.8,
# lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)
# model.fit(X_train,y_train)
# y_pre=model.predict(X_test)
# print("acc:",sklearn.metrics.accuracy_score(y_test,y_pre))
# print("auc:",sklearn.metrics.roc_auc_score(y_test,y_pre))



def XGB():
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=600,
                          max_depth=4,
                          min_child_weight=5,
                          seed = 0,
                          subsample=0.8,
                          colsample_bytree=0.7,
                          gamma=0.2,
                          reg_alpha = 3,
                          reg_lambda = 3,

                          random_state=27,
                          verbosity=1,
                          nthread=-1
                          )
    return model

# cv_params = {'max_depth': [3, 4, 5, 6, 7, 8], 'min_child_weight': [3, 4, 5, 6,7,8]}
# other_params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
# 'subsample': 0.8, 'colsample_bytree': 0.7, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 3}
# model = XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
# optimized_GBM.fit(X_train, y_train)
# evalute_result = optimized_GBM.cv_results_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))       #参数的最佳取值：{'max_depth': 4, 'min_child_weight': 5}, 最佳模型得分:-0.20761724622393474



from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
def LGB_reg():
    lgbr = LGBMRegressor(num_leaves=30
                        ,max_depth=4
                        ,learning_rate=.09
                        ,n_estimators=1000
                        ,subsample_for_bin=5000
                        ,min_child_samples=200
                        ,colsample_bytree=.2
                        ,reg_alpha=.05
                        ,reg_lambda=.1

                        ,max_bin=75
                        ,min_data_in_leaf=91
                        ,bagging_fraction=0.9
                        ,bagging_freq= 10
                        ,feature_fraction= 0.8
                        ,lambda_l1=0.9
                        ,lambda_l2=0.9
                        ,min_split_gain=0
                        )
    return lgbr
# LGB()._estimator_type = "classifier"

cv_params = {'learning_rate': [0.06,0.07, 0.08,0.09]} #'reg_lambda': [0.05, 0.1, 1, 2, 3]}
              #'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]}#

other_params = {
'num_leaves' : 20,
'max_depth' : 4,
'learning_rate' : .09,
'n_estimators' : 1000,
'subsample_for_bin' : 5000,
'min_child_samples' : 200,
'colsample_bytree' : .2,
'reg_alpha' : .05,
'reg_lambda' : .1,
'max_bin' : 75,
'min_data_in_leaf' : 91,
'bagging_fraction' : 0.8,
'bagging_freq' : 10,
'feature_fraction' : 0.6,
'lambda_l1' : 0.9,
'lambda_l2' : 0.9,
'min_split_gain' : 0
}
LGB_cla = lgb.LGBMClassifier(**other_params)
# optimized_GBM = GridSearchCV(estimator=LGB_cla, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
# optimized_GBM.fit(X_train, y_train)
# evalute_result = optimized_GBM.cv_results_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))       #参数的最佳取值：{'max_depth': 4, 'min_child_weight': 5}, 最佳模型得分:-0.20761724622393474

from catboost import CatBoostRegressor,CatBoostClassifier

def CAT():
    cat = CatBoostClassifier(learning_rate=0.05,
                            depth=7,
                            l2_leaf_reg=10,
                            bootstrap_type='Bernoulli',
                            od_type='Iter',
                            od_wait=50,
                            random_seed=11,
                            allow_writing_files=False,
                            iterations=1200,)

    return cat

cv_params = {'l2_leaf_reg': [4, 10, 13]} #'reg_lambda': [0.05, 0.1, 1, 2, 3]}
              #'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]}#
other_params = {
'learning_rate' : .05,
'depth' : 7,
'l2_leaf_reg': 10,
'od_type': 'Iter',
'od_wait': 50,
'random_seed': 11,

}
# Cat_cla = CatBoostClassifier(**other_params)
# optimized_GBM = GridSearchCV(estimator=Cat_cla, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
# optimized_GBM.fit(X_train, y_train)
# evalute_result = optimized_GBM.cv_results_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))



# lgb_model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=5, num_leaves=30,max_bin=55,min_data_in_leaf=151,bagging_fraction=0.9,bagging_freq= 10, feature_fraction= 0.8,
# lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)

# cat_model = CAT()
# cat_model.fit(X_train, y_train)
# pre_cat = cat_model.predict(X_test)
# # print(pre_lgb)
# # print(len(pre_lgb))
# # print(type(pre_lgb))
# fpr_cat, tpr_cat, thresholds_cat = roc_curve(y_test, pre_cat)
# score_cat = auc(fpr_cat, tpr_cat)
# print("CAT的的得分：",score_cat)
#
# lgb_model = LGB_reg()
# lgb_model.fit(X_train, y_train)
# pre_lgb = lgb_model.predict(X_test)
# # print(pre_lgb)
# # print(len(pre_lgb))
# # print(type(pre_lgb))
# fpr_lgb, tpr_lgb, thresholds_lgb = roc_curve(y_test, pre_lgb)
# score_lgb = auc(fpr_lgb, tpr_lgb)
# print("LGB的的得分：",score_lgb)
#
# model = XGB()
# model.fit(X_train, y_train)
# pre_xgb = model.predict_proba(X_test)[:, 1]   #XGB
# # pre_xgb = model.predict(X_test)                 #LGB
# fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, pre_xgb)
# score_xgb = auc(fpr_xgb, tpr_xgb)
# # print(len(pre_xgb))
# print("XGB的的得分：",score_xgb)
# #
# #加权融合
# pre_train = 0.68*pre_xgb + 0.32*pre_lgb #+ 0.1*pre_cat
# fpr, tpr, thresholds = roc_curve(y_test, pre_train)
# score = auc(fpr, tpr)
# print("加权融合:",score)
# # 0.7376016463273588


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


lgb_model = lgb.LGBMClassifier(learning_rate=0.09, n_estimators=1000, subsample_for_bin=5000,min_child_samples=200,colsample_bytree=0.2,reg_alpha=0.05,reg_lambda=0.1,max_depth=4, num_leaves=20,max_bin=75,min_data_in_leaf=91,bagging_fraction=0.8,bagging_freq= 10, feature_fraction= 0.6,
lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)
clf1 = lgb_model
# clf1._estimator_type = "classifier"
clf2 = XGB()

from mlxtend.classifier import StackingClassifier,StackingCVClassifier
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
xgb_model = XGB()
lgb_model = lgb.LGBMClassifier(learning_rate=0.09, n_estimators=1000, subsample_for_bin=5000,min_child_samples=200,colsample_bytree=0.2,reg_alpha=0.05,reg_lambda=0.1,max_depth=4, num_leaves=20,max_bin=75,min_data_in_leaf=91,bagging_fraction=0.8,bagging_freq= 10, feature_fraction= 0.6,
lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)

# clf3 = StackingClassifier(classifiers=[lgb_model,xgb_model],
#                           meta_classifier=LR,
#                           use_probas=True,
#                           verbose= 1)

# clf4 = LogisticRegression(random_state=1)
# clf4 = RandomForestClassifier(random_state=1)
clf4 = CAT()
vclf = VotingClassifier(estimators=[('lgb', clf1), ('xgb', clf2), ('clf4', clf4)], voting='soft', weights=[0.43, 0.57, 0.53])
vclf = vclf.fit(X_train, y_train)
pre_vot = vclf.predict_proba(X_test)[:,1]
fpr_vot, tpr_vot, thresholds_vot = roc_curve(y_test, pre_vot)
score_vot = auc(fpr_vot, tpr_vot)
print(score_vot)
test= test[train_X.columns]
test['issueDate'] = lbl.fit_transform(test['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
test['earliesCreditLine'] = lbl.fit_transform(test['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换
pre = vclf.predict_proba(test)[:,1]
pd.DataFrame({'isDefault':pre},index=test.index).reset_index().rename(columns={"index":"id"}).to_csv('../submit/submit.csv', index=0)
submit = pd.read_csv("../submit/submit.csv")
submit['id'] = [x+800000 for x in range(200000)]
# print(submit['id'])
submit.to_csv("../submit/submit.csv", index=None)


#blending模型融合
from sklearn.metrics import roc_auc_score
# 以python自带的鸢尾花数据集为例
# data_0 = iris.data
# data = data_0[:100, :]
#
# target_0 = iris.target
# target = target_0[:100]

# 模型融合中基学习器
# lgb_model = lgb.LGBMClassifier(learning_rate=0.09, n_estimators=1000, subsample_for_bin=5000,min_child_samples=200,colsample_bytree=0.2,reg_alpha=0.05,reg_lambda=0.1,max_depth=4, num_leaves=20,max_bin=75,min_data_in_leaf=91,bagging_fraction=0.8,bagging_freq= 10, feature_fraction= 0.6,
# lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)
# xgb_cla = XGB()
# clfs = [lgb_model,
#         xgb_cla]

# 切分一部分数据作为测试集
# X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.3, random_state=914)
# X = X_train
# X_predict = X_test
# y = y_train
# y_predict = y_test
#
# # 切分训练数据集为d1,d2两部分
# X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.2)
# dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
# dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))
#
# for j, clf in enumerate(clfs):
#     # 依次训练各个单模型
#     clf.fit(X_d1, y_d1)
#     y_submission = clf.predict_proba(X_d2)[:, 1]
#     dataset_d1[:, j] = y_submission
#     # 对于测试集，直接用这k个模型的预测值作为新的特征。
#     dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
#     print("val auc Score: %f"%roc_auc_score(y_predict, dataset_d2[:, j]))

# 融合使用的模型
# clf1 = xgb_cla
# clf1.fit(dataset_d1, y_d2)
# y_submission_1 = clf1.predict_proba(dataset_d2)[:, 1]
#
# clf2 = lgb_model
# clf2.fit(dataset_d1, y_d2)
# y_submission_2 = clf2.predict_proba(dataset_d2)[:, 1]
#
# y_submission  = 0.67*y_submission_1 + 0.33*y_submission_2

# clf = LogisticRegression()
# clf.fit(dataset_d1, y_d2)
# y_submission = clf.predict_proba(dataset_d2)[:, 1]
#
# print("Val auc Score of Blending: %f" % (roc_auc_score(y_predict, y_submission)))
#
# test= test[train_X.columns]
# test['issueDate'] = lbl.fit_transform(test['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
# test['earliesCreditLine'] = lbl.fit_transform(test['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换
# pre = clf.predict(test)
# # pre = 0.32*lgb_model.predict(test) + 0.68*model.predict_proba(test)[:, 1] #+ 0.1*cat_model.predict(test)
#
# pd.DataFrame({'isDefault':pre},index=test.index).reset_index().rename(columns={"index":"id"}).to_csv('../submit/submit.csv', index=0)
# submit = pd.read_csv("../submit/submit.csv")
# submit['id'] = [x+800000 for x in range(200000)]
# # print(submit['id'])
# submit.to_csv("../submit/submit.csv", index=None)





from mlxtend.classifier import StackingClassifier,StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier


# X_train.dropna(inplace = True)

# LR = LogisticRegression()
# xgb_model = XGB()
# # lgb_model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=5, num_leaves=30,max_bin=55,min_data_in_leaf=151,bagging_fraction=0.9,bagging_freq= 10, feature_fraction= 0.8,
# # lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)
# lgb_model = lgb.LGBMClassifier(learning_rate=0.09, n_estimators=1000, subsample_for_bin=5000,min_child_samples=200,colsample_bytree=0.2,reg_alpha=0.05,reg_lambda=0.1,max_depth=4, num_leaves=20,max_bin=75,min_data_in_leaf=91,bagging_fraction=0.8,bagging_freq= 10, feature_fraction= 0.6,
# lambda_l1=0.9,lambda_l2=0.9,min_split_gain=0)
# # lgb_model = LGB_cla()
# # cat_model = CAT()
# # clf1 = KNeighborsClassifier(n_neighbors=1)
# # clf1 = ExtraTreesClassifier()
# # clf2 = RandomForestClassifier(random_state=1)
# # clf3 = GaussianNB()
# # clf4 = GradientBoostingClassifier()
# sclf = StackingClassifier(classifiers=[lgb_model,xgb_model],
#                           meta_classifier=LR,
#                           use_probas=True,
#                           verbose= 1)
# sclf.fit(X_train,y_train)
# pre =sclf.predict_proba(X_test)[:,1]
# # print(X_test.size())[:,1]
# # pre =sclf.predict(X_test)
# fpr, tpr, thresholds = roc_curve(y_test, pre)
# score = auc(fpr, tpr)
# print(score) #0.7390504896093062
# #
# test= test[train_X.columns]
# test['issueDate'] = lbl.fit_transform(test['issueDate'].astype(str))#将提示的包含错误数据类型这一列进行转换
# test['earliesCreditLine'] = lbl.fit_transform(test['earliesCreditLine'].astype(str))#将提示的包含错误数据类型这一列进行转换
# pre = sclf.predict_proba(test)[:,1]
# pd.DataFrame({'isDefault':pre},index=test.index).reset_index().rename(columns={"index":"id"}).to_csv('../submit/submit.csv', index=0)
# submit = pd.read_csv("../submit/submit.csv")
# submit['id'] = [x+800000 for x in range(200000)]
# submit.to_csv("../submit/submit.csv", index=None)