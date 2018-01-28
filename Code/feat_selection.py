# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:49:46 2018

@author: Leo Mao
"""

import pandas as pd
import numpy as np
#from minepy import MINE
import random
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 


def RF_selection(Train,predictors,seed = 202):
    dfTrain = Train.copy()
    dfTrain.fillna(dfTrain.median(),inplace = True)
    if dfTrain['Y'].nunique() == 2:
        rf = RandomForestClassifier(n_estimators=5000, max_features='sqrt',  max_depth=5, random_state=seed)
    else:
        rf = RandomForestRegressor(n_estimators=5000, max_features='sqrt',  max_depth=5, random_state=seed)
    rf.fit(dfTrain[predictors], dfTrain['Y'])     
    rf_imp_df = pd.DataFrame({'var':predictors,'imp':rf.feature_importances_})
    rf_imp_df.sort_values('imp',ascending=False,inplace =True)
    predictors_rf = rf_imp_df['var'].values.tolist()
    return predictors_rf


def Pearson_selection(dfTrain,predictors):
    corr = []
    for var in predictors:
        corr.append(abs(dfTrain['Y'].corr(dfTrain[var])))
    dfcorr = pd.DataFrame({'var':predictors,'corr':corr})
    dfcorr = dfcorr.sort_values('corr',ascending=False)
    predictors_pear = dfcorr['var'].values.tolist()
    return predictors_pear

def MIC_selection(dfTrain,predictors):
    m = MINE()
    predictors_mic = []
    for var in predictors:
        m.compute_score(dfTrain[var].values.tolist(),dfTrain['Y'].values.tolist())
        predictors_mic.append(m.mic())
    mic_df=pd.DataFrame({'var':predictors,'mic':predictors_mic})
    mic_df = mic_df.sort_values('mic',ascending=False)
    predictors_mic = mic_df['var'].values.tolist()
    return predictors_mic


def feat_combination(predistor_lists = [],top_n = 1000):
    if len(predistor_lists)>1:
        dfCombination = pd.DataFrame(predistor_lists).T
        output = []
        cnt = 0
        while len(output)<top_n:
            output+= dfCombination.iloc[cnt,:].values.tolist()
            output = set(output)
            cnt+=1
        return output
    elif len(predistor_lists)==1:
        return predistor_lists[0][0:top_n]
    else:
        return None
            


def FeatList(featSlected,bins=5,random_pick=False):
    output = [[] for i in range(bins)]
    featTmp = featSlected.copy()
    i = 0
    if random_pick:
        while len(featTmp)>=1:
            for j in range(bins):
                try:
                    pick = random.sample(featTmp,1)[0]
                except:
                    break
                output[j].append(pick)
                featTmp.remove(pick)

    else:    
        while i<=len(featSlected)-1:
            for j in range(bins):
                try:
                    output[j].append(featTmp[i+j])
                except:
                    break
            i += bins
    return output






