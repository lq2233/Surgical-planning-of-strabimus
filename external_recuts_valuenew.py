import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # 分类树
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor # 分类随机森林
# from sklearn.svm import SVM
from sklearn.ensemble import GradientBoostingRegressor   # 集成算法
from sklearn.model_selection import cross_val_score    # 交叉验证
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
from sklearn import metrics # 用于分类任务评价
import math
# import seaborn as sns
from sklearn.utils import shuffle

# 数据导入
df = pd.read_excel('./data_右下斜肌.xlsx')
# df=shuffle(df)
# X_all=df.iloc[:, :25]
# YR_all=df.iloc[:, 27]
# YC_all=df.iloc[:, 26]
# df1=pd.read_excel('./测试集.xlsx')
X1=df.iloc[450:498, :25]  # 测试集变量
Y1=df.iloc[450:498, 27]  # 测试集目标  31!!!!!
YC=df.iloc[450:498, 26]  # 测试集目标 46!!!
# 自变量
X = df.iloc[:450, :25]
y = df.iloc[:450, 27]
# y = df.iloc[:,39]
yc = df.iloc[:450,26]
# 设置交叉验证次数
n_folds = 5
svc_model = SVC()
tree_model = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=False)
forest_model = RandomForestClassifier()

# 建立贝叶斯岭回归模型
br_model = BayesianRidge()

# 普通线性回归
lr_model = LinearRegression()

# 弹性网络回归模型8
svr_model = SVR()

# 梯度增强回归模型对
gbr_model = GradientBoostingRegressor(learning_rate=0.1599,random_state=True, n_estimators=150, max_depth=5)   # 右外直肌肉: 330,5 # 右内直肌 150 ,5  # 右下斜肌 # 左上直肌:150, 5 ,nodes:6
dtr_model = DecisionTreeRegressor(random_state=False)
rfr_model = RandomForestRegressor()
etr_model = ExtraTreesRegressor()
# 不同模型的名称列表
# model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR','Decision Tree','Forest']
model_names = ['SVR',  'Decision Tree', 'Random Forest', 'Extra Tree','GBR']
classify_model_names = ['SVM', 'DecisionTreeClassifier', 'RandomForestClassifier']
# 不同回归模型
# model_dic = [br_model, lr_model, etc_model, svr_model, gbr_model,dtr_model,rfr_model]
model_dic = [svr_model, dtr_model, rfr_model, etr_model, gbr_model]
model_classify_dic = [svc_model, tree_model, forest_model]

# resultr11 = cross_val_score(gbr_model, X_all, YR_all, cv=10, scoring='neg_mean_squared_error')  # pre
# resultr12 = cross_val_score(gbr_model, X_all, YR_all, cv=10, scoring='neg_mean_absolute_error')  # rec
# resultr13 = cross_val_score(gbr_model, X_all, YR_all, cv=10, scoring='r2')  # f1
#
# resultr21 = cross_val_score(rfr_model, X_all, YR_all, cv=10, scoring='neg_mean_squared_error')  # pre
# resultr22 = cross_val_score(rfr_model, X_all, YR_all, cv=10, scoring='neg_mean_absolute_error')  # rec
# resultr23 = cross_val_score(rfr_model, X_all, YR_all, cv=10, scoring='r2')  # f1
#
# resultr31 = cross_val_score(etr_model, X_all, YR_all, cv=10, scoring='neg_mean_squared_error')  # pre
# resultr32 = cross_val_score(etr_model, X_all, YR_all, cv=10, scoring='neg_mean_absolute_error')  # rec
# resultr33 = cross_val_score(etr_model, X_all, YR_all, cv=10, scoring='r2')  # f1
#
# resultr41 = cross_val_score(svr_model, X_all, YR_all, cv=10, scoring='neg_mean_squared_error')  # pre
# resultr42 = cross_val_score(svr_model, X_all, YR_all, cv=10, scoring='neg_mean_absolute_error')  # rec
# resultr43 = cross_val_score(svr_model, X_all, YR_all, cv=10, scoring='r2')  # f1
#
#
# resultr51 = cross_val_score(dtr_model, X_all, YR_all, cv=10, scoring='neg_mean_squared_error')  # pre
# resultr52 = cross_val_score(dtr_model, X_all, YR_all, cv=10, scoring='neg_mean_absolute_error')  # rec
# resultr53 = cross_val_score(dtr_model, X_all, YR_all, cv=10, scoring='r2')  # f1
# 计算平均
# print(resultr11*(-1), (resultr12)*(-1), np.mean(resultr13))   # GBR

# RFC 交叉验证
# resultc11 = cross_val_score(forest_model, X_all, YC_all, cv=10, scoring='precision_weighted')  # pre
# resultc12 = cross_val_score(forest_model, X_all, YC_all, cv=10, scoring='recall_weighted')  # rec
# resultc13 = cross_val_score(forest_model, X_all, YC_all, cv=10, scoring='f1_weighted')  # f1
# # print(resultc11, (resultc12), np.mean(resultc13))   # RFC
# # DTC 交叉验证
# resultc21 = cross_val_score(tree_model, X_all, YC_all, cv=10, scoring='precision_weighted')  # pre
# resultc22 = cross_val_score(tree_model, X_all, YC_all, cv=10, scoring='recall_weighted')  # rec
# resultc23 = cross_val_score(tree_model, X_all, YC_all, cv=10, scoring='f1_weighted')  # f1


# print('GBR')
# print(resultr11, (resultr12), (resultr13), np.mean(resultr11), np.mean(resultr12), np.mean(resultr13))   # GBR
# print('RFR')
# print(resultr21, (resultr22), (resultr23), np.mean(resultr21), np.mean(resultr22), np.mean(resultr23))   # GBR
# print('ETR')
# print(resultr31, (resultr32), (resultr33), np.mean(resultr31), np.mean(resultr32), np.mean(resultr33))   # GBR
# print('SVR')
# print(resultr41, (resultr42), (resultr43), np.mean(resultr41), np.mean(resultr42), np.mean(resultr43))   # GBR
# print('DTR')
# print(resultr51, (resultr52), (resultr53), np.mean(resultr51), np.mean(resultr52), np.mean(resultr53))   # GBR
# print('RFC')
# print(resultc11, (resultc12), (resultc13), np.mean(resultc11), np.mean(resultc12), np.mean(resultc13))   # RFC
# print('DTC')
# print(resultc21, (resultc22), (resultc23), np.mean(resultc21), np.mean(resultc22), np.mean(resultc23))   # DTC
# 交叉验证结果
cv_score_list = []
# 各个回归模型预测的y值列表
pre_y_list = []
pre_y1_list = []
pre_yc_list = []
pre_y2_list = []
pre_yc1_list = []
# 读出每个回归模型对象
for model in model_dic:
    # 将每个回归模型导入交叉检验
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='r2')
    # 将交叉检验结果存入结果列表
    cv_score_list.append(scores)
    # 将回归训练中得到的预测y存入列表
    pre_y_list.append(model.fit(X, y).predict(X))
    temp = model.fit(X, y).predict(X1)
    pre_y1_list.append(temp)


pre_yc1_list.append(svc_model.fit(X,yc).predict(X))
pre_yc1_list.append(tree_model.fit(X,yc).predict(X))
pre_yc1_list.append(forest_model.fit(X,yc).predict(X))
pre_yc_list.append(svc_model.fit(X,yc).predict(X1))
pre_yc_list.append(tree_model.fit(X,yc).predict(X1))
pre_yc_list.append(forest_model.fit(X,yc).predict(X1))
### 模型效果指标评估 ###
# 获取样本量，特征数
n_sample, n_feature = X.shape
# 回归评估指标对象列表
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
# 回归评估指标列表
model_metrics_list = []
# 循环每个模型的预测结果
for pre_y in pre_y_list:
    # 临时结果列表
    tmp_list = []
    # 循环每个指标对象
    for mdl in model_metrics_name:
        # 计算每个回归指标结果
        tmp_score = mdl(y, pre_y)
        # 将结果存入临时列表
        tmp_list.append(tmp_score)
    # 将结果存入回归评估列表
    model_metrics_list.append(tmp_list)
df_score = pd.DataFrame(cv_score_list, index=model_names)
df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])

# 各个交叉验证的结果

print (pre_y1_list)  # 测试集回归结果
for i in range(len(pre_y1_list)):
    # 处理单个元组。
    for j in range(len(pre_y1_list[i])):
        temp=pre_y1_list[i][j]
        ceil1=math.ceil(pre_y1_list[i][j])
        floor1=math.floor(pre_y1_list[i][j])
        if (math.fabs(temp-floor1)>0.25 or math.fabs(ceil1-temp)>0.25):
            pre_y1_list[i][j]=floor1+0.5
        if (math.fabs(temp-floor1)<=0.25):
            pre_y1_list[i][j]=floor1
        if (math.fabs(ceil1-temp)<=0.25):
            pre_y1_list[i][j] = ceil1
    print(model_names[i], '均方误差', math.sqrt(mean_squared_error(pre_y1_list[i], Y1)), '绝对误差', mean_absolute_error(pre_y1_list[i], Y1),'R方',r2_score(pre_y1_list[i],Y1))
print(pre_y1_list)

# 计算每个绝对误差
# print(list(Y1))

# for i in range(len(pre_y1_list)):
#     for j in range(len(pre_y1_list[i])):
#         temp = pre_y1_list[i][j]
#         pre_y1_list[i][j] = math.fabs(temp-list(Y1)[j])
print(pre_y1_list)
print(pre_yc_list) # 测试集分类结果
for i in range(len(pre_yc_list)):
    precision, recall, fscore, support = score(YC, pre_yc_list[i])
    # print('pre',precision,'recall',recall,'fscore',fscore,'support',support)
    print(classify_model_names[i])
    print('pre',metrics.precision_score(YC,pre_yc_list[i], average='weighted'))
    print('rec', metrics.recall_score(YC, pre_yc_list[i], average='weighted'))
    print('f1', metrics.f1_score(YC, pre_yc_list[i], average='weighted'))
    # print(classify_model_names[i],'PRE', precision_score(pre_yc_list[i], YC, average='macro'), 'REC', recall_score(pre_yc_list[i], YC, average='macro'),'F1',f1_score(pre_yc_list[i],YC, average='macro'))
print(df_score)
print(df_met)



# print(len(SVR_result),len(Y1))
# plt.figure(figsize=(10, 3.3))
# plt.subplot(1,3,1)
# plt.scatter(np.array(SVR_result), np.array(Y1), c='red')
# plt.subplot(1,3,2)
# plt.scatter(np.array(GBR_result), np.array(Y1), c='red')
# plt.subplot(1,3,3)
# plt.scatter(np.array(DTR_result), np.array(Y1), c='red')
# plt.xticks(range(0,9,1))
# plt.yticks(range(0,9,1))

SVR_result = pre_y1_list[0]
DTR_result = pre_y1_list[1]
RFR_result = pre_y1_list[2]
ETR_result = pre_y1_list[3]
GBR_result = pre_y1_list[4]
plt.plot(np.arange(X1.shape[0]), GBR_result, 'k')
# print(SVR_result,GBR_result,DTR_result)
### 可视化 ###
# 创建画布
plt.figure(figsize=(6, 6))
# 颜色列表
color_list = ['k', 'tan', 'darkgoldenrod', 'darkseagreen', 'b']
# 循环结果画图

for i, pre_y in enumerate(pre_y1_list):
    # 子网络
    # plt.subplot(2, 4, i+1)
    # 画出原始值的曲线
    # plt.plot(np.arange(X1.shape[0]), Y1, color='k', label='y')
    # 画出各个模型的预测线
    plt.plot(np.arange(X1.shape[0]), pre_y, color_list[i], label=model_names[i])
    # plt.title(model_names[i])
plt.plot(np.arange(X1.shape[0]), Y1, color='r', label='label')

plt.xlabel("Test samples")
plt.ylabel("Surgical dosahe")
plt.legend()
plt.grid(True)
plt.savefig('xxx.png')
plt.show()



### 可视化 ### 分类
# 创建画布
plt.figure(figsize=(9, 6))
# 颜色列表
color_list = ['k', 'g', 'b']
# 循环结果画图
for i, pre_y in enumerate(pre_yc1_list):
    # 子网络
    plt.subplot(1, 3, i+1)
    # 画出原始值的曲线
    plt.plot(np.arange(X.shape[0]), yc, color='r', label='y')
    # 画出各个模型的预测线
    plt.plot(np.arange(X.shape[0]), pre_y, color_list[i], label=classify_model_names[i])
    print(classify_model_names[i])  # 模型名称
    TP = np.sum(np.logical_and(np.equal(yc, 1), np.equal(pre_y, 1)))
    # print(TP)

    # false positive
    FP = np.sum(np.logical_and(np.equal(yc, 0), np.equal(pre_y, 1)))
    # print(FP)

    # true negative
    TN = np.sum(np.logical_and(np.equal(yc, 1), np.equal(pre_y, 0)))
    # print(TN)

    # false negative
    FN = np.sum(np.logical_and(np.equal(yc, 0), np.equal(pre_y, 0)))
    # print(FN)
    Pre=TP/(TP+FP)
    Rec=TP/(TP+FN)
    print('Pre',TP/(TP+FP))
    print('Rec',TP/(TP+FN))
    print('ACC',(TP+TN)/(TP+TN+FP+FN))
    print('F1',2*Pre*Rec/(Pre+Rec))
    plt.title(classify_model_names[i])
    plt.legend(loc='lower left')
plt.savefig('xxx_c.png')
plt.show()


