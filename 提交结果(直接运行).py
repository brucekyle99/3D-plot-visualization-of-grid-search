# coding: utf-8
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt1
import matplotlib.pylab as plt2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

if __name__=='__main__':
    # 读取数据
    train=pd.read_csv('./Test1_features.csv',sep=',', header=None)
    label=pd.read_csv('./Test1_labels.csv', header=None)
    # print(label[0].value_counts())

    train=pd.DataFrame(train)
    label=pd.DataFrame(label)

    # 预处理label数据, 以防cv报错
    c, r = label.shape
    label = label.values.reshape(c,)

    '''
    # 用lgbm训练，并做5折cv
    '''
    clf = lgbm.LGBMClassifier(
                boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=5,
                max_depth=-1, n_estimators=100, objective='binary',
                subsample=0.6, colsample_bytree=0.7, subsample_freq=1,
                learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=-1
            )
    scores = cross_val_score(clf, train, label, cv=5, scoring='roc_auc')

    '''
    # 用普通的lr
    '''
    clf2 = LogisticRegression()

    scores2 = cross_val_score(clf2, train, label, cv = 5, scoring = 'roc_auc')

    '''
    # 用随机森林
    '''
    clf3 = RandomForestClassifier()

    scores3 = cross_val_score(clf3, train, label, cv = 5, scoring = 'roc_auc')

   # grid search
    param_test1 = {
     'n_estimators':list(range(80,140,10)),
      'num_leaves':list(range(15,60,5))
    }

    gsearch1 = GridSearchCV(clf,
    param_grid = param_test1, scoring='roc_auc',n_jobs=4,cv=5)

    gsearch1.fit(train,label)

    res=gsearch1.cv_results_
    res=pd.DataFrame(res)

    print('lgbm cv5 test result is: ')
    print(scores.sum() / 5)
    print('lgbm feature importance is: ')
    # plt2.figure(figsize=(12, 6))
    # lgbm.plot_importance(clf, max_num_features=10)
    # plt2.title("Featurertances")
    # plt2.show()

    print('lr cv5 test result is: ')
    print(scores2.sum() / 5)

    print('rf cv5 test result is: ')
    print(scores3.sum() / 5)

    print('------------lgbm gridsearch is done-------------')

    # 根据坐标绘制三维图
    X=[]
    Y=[]
    Z=[]
    iter=res.shape[0]
    for i in range(iter):
        X.append(res.loc[i]['params']['n_estimators'])
        Y.append(res.loc[i]['params']['num_leaves'])
        Z.append(res.loc[i]['mean_test_score'])

    fig = plt1.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z)

    print('---------plotting gridsearch result----------')
    plt1.show()

























