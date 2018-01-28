# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:27:07 2018

@author: Leo Mao
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LassoLars,LinearRegression
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier as cbc
from catboost import CatBoostRegressor as cbr


def catboost_kfold(Train,Pred,predictors,n_splits=5,early_stop = 10,ins_rmse = 0,imbalance = None,seed=615):
    dfTrain = Train.copy()
    dfPred = Pred.copy()
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    round=0
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']
        if dfTrain.Y.nunique()==2:
            print('it is a classifier problem')
            model = cbc(iterations=2000, learning_rate=0.01,depth=5, l2_leaf_reg=3,verbose=0,od_type = "Iter",od_wait = early_stop)
            model.fit(train_X,train_Y,eval_set=(test_X,test_Y))
            bst_tree = model.tree_count_ - early_stop -2
            print(bst_tree)
            pred_test = model.predict_proba(test_X,ntree_end=bst_tree)
            pred_score = model.predict_proba(dfPred[predictors].values,ntree_end=bst_tree)
            
            if round==1:
                test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test.T[1],'target':test_Y})
                result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score.T[1]})
            else:
                test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test.T[1],'target':test_Y})],axis=0)
                result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score.T[1]}),'inner','ID')
            
        else:
            print('it is a classifier problem')
            model = cbr(iterations=2000, learning_rate=0.01,depth=5, l2_leaf_reg=3,verbose=0,od_type = "Iter",od_wait = early_stop)
            model.fit(train_X,train_Y,eval_set=(test_X,test_Y))
            pred_test = model.predict(test_X)
   
            pred_score = model.predict(dfPred[predictors].values)
            if round==1:
                test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
                result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
            else:
                test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
                result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
            
    if dfTrain.Y.nunique()==2:
        print("Test Logloss:",metrics.log_loss(test_result['target'], test_result['score']))
        return test_result,result,model
    else:
        print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))
        return test_result,result

def xgb_kfold(Train,Pred,predictors,n_splits=5,early_stop = 100,ins_rmse = 0,imbalance = None,params = {'max_depth':3, 'eta':0.01, 'silent':0,'objective':'reg:linear','lambda':1,'subsample':0.8,
                         'colsample_bytree':0.8},seed=615):
    dfTrain = Train.copy()
    dfPred = Pred.copy()
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    dpred = xgb.DMatrix(dfPred[predictors].values,label=[0]*len(dfPred),missing=np.nan,feature_names=predictors)
    imp = pd.DataFrame({'variable':predictors,'lk':['f'+str(i) for i in range(len(predictors))]})
    round=0
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        trainTmp = dfTrain.loc[train_index,:]
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']
        ###case in imbalance model
        if imbalance =='smote':
            sm = SMOTE(random_state=202)
            train_X,train_Y = sm.fit_sample(trainTmp[predictors],trainTmp['Y'])

        

        dtrain = xgb.DMatrix(train_X, label=train_Y, missing = np.nan,feature_names=predictors)
        dtest = xgb.DMatrix(test_X.values, label=test_Y.values, missing = np.nan,feature_names=predictors)
        param = params 
        evallist  = [(dtrain,'train'),(dtest,'eval')]  
        num_round = 5000
        evals_dict = {}
        if 'eval_metric' in params:
            metric = params['eval_metric']
        else:
            print('No metric defined, will use rmse in the model')
            metric ='rmse'
        model = xgb.train(param,dtrain,num_round, evallist,early_stopping_rounds=early_stop,evals_result=evals_dict,verbose_eval =100)
        performance_df = pd.DataFrame({'train':evals_dict['train'][metric],'eval':evals_dict['eval'][metric]})
        performance_df =performance_df.loc[performance_df['train']>=ins_rmse]
        if metric!='auc':
            bst_tree = performance_df.loc[performance_df['eval']==performance_df['eval'].min()].index.tolist()[0] + 1
        else:
            bst_tree = performance_df.loc[performance_df['eval']==performance_df['eval'].max()].index.tolist()[0] + 1
        print('Best tree is %d, performance is %f, %f'%(bst_tree,performance_df.loc[bst_tree-1,'train'],performance_df.loc[bst_tree-1,'eval']))
        pred_test = model.predict(dtest,ntree_limit =bst_tree)
        
        ###plot the importance
        '''fig, ax = plt.subplots(figsize=(12,18))
        xgb.plot_importance(model,max_num_features=50,height=0.8, ax=ax)
        plt.show()'''
        
        
        tmp_imp = pd.DataFrame(model.get_score(),index=['imp_fold%d'%round]).T
        tmp_imp['variable'] = tmp_imp.index
        imp = imp.merge(tmp_imp,'left','variable').fillna(0)


        pred_score = model.predict(dpred,ntree_limit =bst_tree)
        if round==1:
            test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
            result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
        else:
            test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
            result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
    if metric == 'logloss':
        print("Test LogLoss:",metrics.log_loss(test_result['target'], test_result['score']))
    else:
        print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))
    return test_result,result,imp



def rf_kfold(dfTrain,dfPred,predictors,n_splits=5,num_round = 5000,seed=202):  
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    imp = pd.DataFrame({'variable':predictors,'lk':['f'+str(i) for i in range(len(predictors))]})
    round=0
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']

        if dfTrain['Y'].nunique() == 2:
            model = RandomForestClassifier(n_estimators=num_round, max_features='sqrt',  max_depth=5, random_state=seed)
            model.fit(train_X,train_Y)        
            pred_test = model.predict_proba(test_X)
            imp['imp_fold%d'%round] = model.feature_importances_
            pred_score = model.predict_proba(dfPred[predictors].values)
            if round==1:
                test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test.T[1],'target':test_Y})
                result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score.T[1]})
            else:
                test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test.T[1],'target':test_Y})],axis=0)
                result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score.T[1]}),'inner','ID')
            print("Test Logloss:",metrics.log_loss(test_result['target'], test_result['score']))
        else:
            model = RandomForestRegressor(n_estimators=num_round, max_features='sqrt',  max_depth=5, random_state=seed)
            model.fit(train_X,train_Y)        
            pred_test = model.predict(test_X)
            imp['imp_fold%d'%round] = model.feature_importances_
            pred_score = model.predict(dfPred[predictors].values)
            if round==1:
                test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
                result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
            else:
                test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
                result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
            print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))       
    return test_result,result,imp


def linear_kfold(dfTrain,dfPred,predictors,n_splits=5):  
    kf = KFold(n_splits=n_splits,shuffle=True)
    
    round=0
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']

        
        model = LinearRegression()
        model.fit(train_X,train_Y)
        
        pred_test = model.predict(test_X)


        pred_score = model.predict(dfPred[predictors].values)
        if round==1:
            test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
            result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
        else:
            test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
            result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
    print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))
    return test_result,result

def simple_avg(dfTrain,dfPred,predictors,weight=False):
    test_result = dfTrain.copy()
    result = dfPred.copy()
    if weight:
        weightList = []
        for var in predictors:
            weightList.append(1/metrics.mean_squared_error(test_result['Y'], test_result[var]))
        weightList =np.array(weightList)
        weightList = len(predictors)*weightList/weightList.sum()
        for i in range(len(predictors)):
            test_result[predictors[i]] = test_result[predictors[i]]*weightList[i]
            result[predictors[i]] = result[predictors[i]]*weightList[i]
    
    test_result['score'] = test_result[predictors].mean(axis=1)
    result['score'] = result[predictors].mean(axis=1)
    print("Test MSE:",metrics.mean_squared_error(test_result['Y'], test_result['score']))
    return test_result,result[['ID','score']]
    


























































































































