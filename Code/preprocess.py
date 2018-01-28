# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:49:36 2017

@author: maoli
"""

import sys
#import cPickle
import numpy as np
from sklearn import metrics
import pandas as pd
#from nlp_utils import clean_text, pos_tag_text
sys.path.append("../")
from param_config import config
import datetime
import model_ml
import feat_selection as fs



###############
## Load Data ##
###############
print("Load data...")

dfTrain = pd.read_csv(config.original_train_data_path)
dfPred = pd.read_csv(config.original_test_data_path)
# number of train/test samples
num_train, num_test = dfTrain.shape[0], dfPred.shape[0]

print("Done.")


###############
## Load Features ##
###############

print("Load Features")
Feat_file = ['Feat_cnt_col_nonmissing','Feat_cnt_tool_nonmissing','Feat_pcent_col_nonmissing','Feat_pcent_tool_nonmissing','All_minus','group_var','All_minus_single_col','All_minus_byother_col']
ex_feat = []
for filename in Feat_file:
    if filename == 'Feat_minus_non_missing':
        tmp = pd.read_csv('../../'+filename+'.csv')
    else:
        tmp = pd.read_csv('../Cache/'+filename+'.csv')
    tmp.set_index(tmp['ID'],inplace=True,drop=True)
    tmp_feat_list = tmp.columns.tolist()
    if 'ID' in tmp_feat_list:
        tmp_feat_list.remove('ID')
    ex_feat += tmp_feat_list
    dfTrain = pd.merge(dfTrain,tmp,'left','ID')
    dfPred = pd.merge(dfPred,tmp,'left','ID')

print("Done")

###############
## Get dummies ##
###############

print("Get dummies")
predictors = dfTrain.columns.tolist()
predictors.remove('ID')
predictors.remove('Y')

dummies_list = ['TOOL_210','TOOL_220','TOOL_300','TOOL_310','TOOL_311','TOOL_312','TOOL_330','TOOL_340','TOOL_344','TOOL_360','TOOL_440','TOOL_520','TOOL_750']

for var in dummies_list:
    #if 'TOOL' in var:
        #continue
    #if dfTrain[var].dtypes=='object':
    predictors.remove(var)
    tmpTrain = pd.get_dummies(dfTrain[var],prefix=var,dummy_na=True)
    tmpPred = pd.get_dummies(dfPred[var],prefix=var,dummy_na=True)
    predictors = predictors + tmpTrain.columns.tolist()
    dfTrain = pd.concat([dfTrain,tmpTrain],axis=1)
    dfPred = pd.concat([dfPred,tmpPred],axis=1)     
print("Delete the dummy variable missing in dfPred")
for var in predictors:
    if var not in dfPred.columns:
        print(var)
        del dfTrain[var]
        predictors.remove(var)
    
print("Done")


###############
## Feature Selection ##
###############
print("Feature selection")
predictors_rf = fs.RF_selection(dfTrain,predictors)
print('RF Done')
predictors_pear = fs.Pearson_selection(dfTrain,predictors)
print('Pearson Done')
#predictors_mic = fs.MIC_selection(dfTrain,predictors)
print('MIC Done')






#featSelected = fs.feat_combination([predictors_rf],350)
featSelected = fs.feat_combination([predictors_pear],800)
#featSelected_list = fs.FeatList(fs.feat_combination([predictors_rf],1500))
#featSelected_list = featSelected_list + fs.FeatList(fs.feat_combination([predictors_pear],1500))

#featSelected_list = featSelected_list + FeatList(fs.feat_combination([predictors_pear],1500),random_pick=True)
#featSelected_list = featSelected_list + FeatList(fs.feat_combination([predictors_pear],1500),random_pick=True)
#featSelected_list = featSelected_list + FeatList(fs.feat_combination([predictors_mic],1500))


print("Done")


###############
## model ##
###############
print("run model")

n_splits = 5
early_stop =50
ins_rmse = 0.01
params = {'max_depth':3, 'eta':0.01, 'silent':0,'objective':'reg:linear','lambda':1,'subsample':0.8,
                         'colsample_bytree':0.8,'eval_metric':'rmse'}
test_result,result,imp = model_ml.xgb_kfold(dfTrain,dfPred,featSelected,n_splits=n_splits,early_stop=early_stop,ins_rmse = ins_rmse)
result['score']=result[['Score_%d'%i for i in range(1,n_splits+1)]].mean(axis=1)
print(result.min())
print(result.iloc[61,:])



'''model_round = 0
for featSelected in featSelected_list:
    model_round+=1
    test_result,result,imp = model_ml.xgb_kfold(dfTrain,dfPred,featSelected,params=params,n_splits=n_splits,early_stop=early_stop,ins_rmse = ins_rmse)
    test_result = test_result.rename(columns={'score':'score_%d'%model_round,'target':'Y'})
    result['score_%d'%model_round]=result[['Score_%d'%i for i in range(1,n_splits+1)]].mean(axis=1)
    if model_round==1:
        ensembleTrain = test_result
        ensemblePred = result[['ID','score_%d'%model_round]]
    else:
        ensembleTrain = ensembleTrain.merge(test_result[['ID','score_%d'%model_round]],'inner','ID')
        ensemblePred = ensemblePred.merge(result[['ID','score_%d'%model_round]],'inner','ID')



for i in range(1,ensembleTrain.shape[1]-1):
    print(metrics.mean_squared_error(ensembleTrain['Y'], ensembleTrain['score_%d'%i]))
    
    
test_result,result =  model_ml.linear_kfold(ensembleTrain,ensemblePred,[i for i in ensembleTrain.columns if 'score_' in i],n_splits=n_splits)


test_result,result =  model_ml.simple_avg(ensembleTrain,ensemblePred,[i for i in ensembleTrain.columns if 'score_' in i])
test_result,result =  model_ml.simple_avg(ensembleTrain,ensemblePred,[i for i in ensembleTrain.columns if 'score_' in i],True)
#'''
result['score']=result[['Score_%d'%i for i in range(1,n_splits+1)]].mean(axis=1)
print(result.min())
print(result.iloc[61,:])
'''
print("Done")



###############
## update result ##
###############
print("get submission")

other_note ='_400_rf'
result['score']=result[['Score_%d'%i for i in range(1,n_splits+1)]].mean(axis=1)
submit = result[['ID','score']]
today = datetime.date.today().strftime('%Y-%m-%d')
result.to_csv('../Submission/result/result_%s'%today+other_note+'.csv',index=False)
submit.to_csv('../Submission/submit_%s'%today+other_note+'.csv',header=False,index=False)
test_result.to_csv('../Submission/test/test_result_%s'%today+other_note+'.csv',index=False)
#imp.to_csv('../Submission/imp/importance_%s'%today+other_note+'.csv',index=False)

#'''

print("Done")
result['score']=result[['Score_%d'%i for i in range(1,n_splits+1)]].mean(axis=1)
import matplotlib.pyplot as plt
plt.hist(result['score'], facecolor='red')


