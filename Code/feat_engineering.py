# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:35:17 2017

@author: maoli
"""

import pandas as pd
import numpy as np
import random
import datetime
import scipy.stats as stats
from scipy import special
from functools import reduce
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization



def fillna_RF(df,predictors,recurrent = False,seed = 202):
    dfTrain = df.copy()
    dfTrain.fillna(dfTrain.median(),inplace = True)
    baseVar = []
    fillVar = []
    for var in predictors:
        if dfTrain[var].isnull().sum()==0:
            baseVar.append(var)
        else:
            fillVar.append(var)
    fillVar = dfTrain[fillVar].isnull().sum().reset_index().sort_values(0,ascending=True)['index'].values.tolist()
    while len(fillVar)>0:
        missVar = fillVar[0]
        trainX = dfTrain.loc[~dfTrain[missVar].isnull(),baseVar]
        trainY = dfTrain.loc[~dfTrain[missVar].isnull(),missVar]
        testX = dfTrain.loc[dfTrain[missVar].isnull(),baseVar]
        rf = RandomForestRegressor(n_estimators=5000, max_features='sqrt',max_depth=5, random_state=seed)
        rf.fit(trainX,trainY)
        result = rf.predict(testX)
        if dfTrain[missVar].nunique()<=20:
            print("%s is a category variable, will fill by the nearest cate"%missVar)
            unique = dfTrain[missVar].unique()
            unique.sort()
            for i in range(len(result)):
                t = -999999
                pred = result[i]
                for j in range(len(unique)):
                    value = unique[j]
                    gap = value-pred
                    if t*gap<=0:
                        if gap<=-1*t:
                            result[i] = value
                        else:
                            result[i] = unique[j-1]
                    else:
                        t = gap      
        dfTrain.loc[dfTrain[missVar].isnull(),baseVar] = result
        fillVar.remove(missVar)
        if recurrent:
            baseVar.append(missVar)
    return dfTrain
        
        
        

def count_single_col(dataset,var_list,keep_list = []):
    for var in var_list:
        if not var in dataset.columns:
            continue
        dataset['cnt_'+var] = 0
        keep_list.append('cnt_'+var)
        groups = dataset.groupby([var])
        for name,group in groups:
            count = group[var].count()
            dataset['cnt_'+var].ix[group.index] = count
    return dataset[keep_list]


def count_by_other_col(dataset,var_dict,keep_list = []):
    for tool,pro in var_dict.items():
        try:
            groups = dataset.groupby(['TOOL_'+tool[-3:]])
        except:
            continue
        for var in pro:
            if not var in dataset.columns:
                continue
            dataset['cnt_'+tool+'_'+var] = 0
            keep_list.append('cnt_'+tool+'_'+var)
            for name,group in groups:
                grps = group.groupby([var])
                for name2,grp in grps:
                    dataset['cnt_'+tool+'_'+var].ix[grp.index] = float(len(grp))/float(len(group))
    return dataset[keep_list]



def pcent_single_col(dataset,var_list,keep_list = []):
    for var in var_list:
        if not var in dataset.columns:
            continue
        if 'TOOL' not in var:
            dataset['pcent_'+var] = dataset[var].rank(method='max')/float(len(dataset))
            keep_list.append('pcent_'+var)
    return dataset[keep_list]


def pcent_by_other_col(data,var_dict,keep_list = []):
    dataset = data.copy()
    for tool,pro in var_dict.items():
        try:
            groups = dataset.groupby([tool])
        except:
            continue
        for var in pro:
            if not var in dataset.columns:
                continue
            dataset['pcent_'+tool+'_'+var] = 0
            keep_list.append('pcent_'+tool+'_'+var)
            for name,group in groups:
                dataset['pcent_'+tool+'_'+var].ix[group.index] = group[var].rank(method='max')/float(len(group))
    return dataset[keep_list]



def var_minus(dataset,var_list,keep_list = [],criteria=0.8):
    for i in range(len(var_list)-1):
        var1_unique = dataset[var_list[i]].unique().tolist()
        for j in range(i+1,len(var_list)):
            var2_unique = dataset[var_list[j]].unique().tolist()
            combine_list = set(var1_unique+var2_unique)
            if float(len(var1_unique))/len(combine_list)>=criteria and float(len(var2_unique))/len(combine_list)>=criteria:
                tmp_var = var_list[i]+'-'+var_list[j]
                dataset[tmp_var] = dataset[var_list[i]]-dataset[var_list[j]]
                keep_list.append(tmp_var)
    return dataset[keep_list]


'''def sim_group(dataset,var_list):
    base_list = var_list.copy()
    range_df = pd.DataFrame({'min':dataset[var_list].min(),'max':dataset[var_list].max()},index=var_list)
    range_df.sort_values(['max','min'],inplace=True)
    range_df['range'] = 1
    tmp = range_df.groupby(['max','min'])['range'].sum()
    tmp.reset_index(inplace=True) 
    range_dict ={}
    for i in tmp.index:
        tmp_list = range_df.loc[(range_df['min']==tmp.loc[i,'min'])&(range_df['max']==tmp.loc[i,'max'])].index.tolist()
        range_dict[tmp_list] = []
        for var in tmp_list:
            
    for var_list in range_list:
        for var in var_list:
            base_list.remove(var)'''
            
def NN_DAE(dataset,predictors):
    data = dataset.copy()
    data = data.reset_index(drop=True)
    scale = StandardScaler()
    data[predictors] = scale.fit_transform(data[predictors])
    featNum = len(predictors)
    model = Sequential()
    model.add(Dense(int(featNum/2),activation='relu',input_shape=(featNum,),kernel_initializer='glorot_normal'))
    model.add(Dense(int(featNum/3),activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dense(int(featNum/4),activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dense(int(featNum/3),activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dense(int(featNum/2),activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dense(featNum,activation='relu'))
    model.compile(optimizer='adam',loss='mse')
    model.fit(data[predictors].values,data[predictors].values,epochs=5,batch_size=5,verbose=2)
    
    layer_feature = Model(inputs=model.input, outputs=model.get_layer(index=int(len(model.layers)/2)).output)
    features_array = layer_feature.predict(data[predictors].values, batch_size=1)
    column_name = ['Feat_' + str(i + 1) for i in range(features_array.shape[1])]
    dfFeat = pd.DataFrame(features_array,index=data['ID'],columns=column_name).reset_index()
    return dfFeat
    
###contingency
__all__ = ['margins', 'expected_freq', 'chi2_contingency']


def margins(a):
    margsums = []
    ranged = list(range(a.ndim))
    for k in ranged:
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums


def expected_freq(observed):
    observed = np.asarray(observed, dtype=np.float64)
    margsums = margins(observed)
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected

def chi2_contingency(observed, correction=True):
    observed = np.asarray(observed) + 0.0001
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = ((observed - expected) ** 2 / expected).sum()
        p = special.chdtrc(dof, chi2)    
    return chi2, p, dof, expected           


###numeric woe
def woe_calc_base(bad, good):
    try:
        woe = np.log(float(bad / float(bad + good)) / float(good / float(bad + good)))
    except:
        woe = -999999
    return woe


# locate bins for numeric variables
def bin_loc(value, uvbucket):
    if pd.isnull(value):
        # NA bins
        return (-np.inf, -np.inf)

    bins = np.empty(0)
    for i in range(len(uvbucket) - 1):
        if value >= uvbucket[i] and value < uvbucket[i + 1]:
            bins = (uvbucket[i], uvbucket[i + 1])
            break

    return bins


# main function: get the reference table for numeric variables
def calc_numinal_woe(df, var, tgt, max_bins, verbose=True):
    start_time = datetime.datetime.now()

    '''
    ------- Initialize: create the numeric bins -------#
    '''
    if len(df[var].unique()) < max_bins:
        uvalue = np.sort(df[var].unique())
        uvdiff = np.append(np.diff(uvalue).astype(float) / 2, 0)
        uvbucket = np.append(np.nanmin(uvalue), uvalue + uvdiff)
    else:
        print(1. * np.arange(max_bins + 1) / max_bins)
        q = df.ix[~df[var].isnull(), [var]] \
            .quantile(1. * np.arange(max_bins + 1) / max_bins) \
            .drop_duplicates()

        uvalue = list(q[var])
        uvdiff = np.append(np.diff(uvalue).astype(float) / 2, 0)
        uvbucket = np.append(np.nanmin(uvalue), uvalue + uvdiff)
        b_bc = np.bincount(np.digitize(df[var].values, uvbucket))
        b_idx = np.where(b_bc == 0)
        uvbucket = np.delete(uvbucket, b_idx[0][1:])

        if df[var].isnull().sum() > 0:
            uvbucket = np.append(uvbucket, np.nan)

    uvbucket[0] = -np.inf
    if np.isnan(uvbucket[-1]):
        uvbucket[-2] = np.inf
    else:
        uvbucket[-1] = np.inf

    df[var + '_bin'] = df[var].apply(lambda x: bin_loc(x, uvbucket))

    col_t = [c for c in df.columns if c != var and c != tgt][0]
    ds = df[[var + '_bin', tgt, col_t]].groupby([var + '_bin', tgt]).count().unstack().fillna(value=0)
    ds['bin'] = [[str(i[0]), str(i[1])] for i in list(ds.index)]
    ds = ds.reset_index(drop=True)
    chisq = []
    for i in range(ds.shape[0] - 1):
        chisq.append(chi2_contingency(ds.iloc[i: i + 2,0:2])[0])
    chisq.append(9999999.0)
    ds['chisq'] = chisq
    ds.columns = ds.columns.swaplevel(0, 1).droplevel()
    ds.columns = [0, 1, 'bin', 'chisq']

    '''
    #------- chimerge: merge the adjacent bins, except bin for NA (bin as ['-inf', '-inf']) -------#
    '''
    start_time = datetime.datetime.now()
    inds_na = ds['bin'].apply(lambda b: b == ['-inf', '-inf'])
    while (ds.shape[0] > 5) or (ds.shape[0] > 2 and ds.ix[~inds_na, 'chisq'].min() <= stats.chi2.ppf(0.95, 1)):
        start_time = datetime.datetime.now()
        # calculate chisquare statistic
        chisq = []
        for i in range(ds.shape[0] - 1):
            chisq.append(chi2_contingency(ds.iloc[i: i + 2,0: 2])[0])
        chisq.append(9999999.0)
        ds['chisq'] = chisq
        ds_idx_list = list(ds.index)
        k = ds_idx_list.index(ds[ds['chisq'] == ds.ix[~inds_na, 'chisq'].min()].index[0])
        ds.ix[ds_idx_list[k], 0:2] = ds.ix[ds_idx_list[k], 0:2] + ds.ix[ds_idx_list[k + 1], 0:2]
        ds['bin'].ix[ds_idx_list[k]] = [ds['bin'].ix[ds_idx_list[k]][0], ds['bin'].ix[ds_idx_list[k + 1]][1]]
        ds = ds.drop(ds_idx_list[k + 1], axis=0)
        ds = ds.reset_index(drop=True)
        ds_idx_list = list(ds.index)
        if k != 0:
            ds['chisq'].ix[ds_idx_list[k - 1]] = chi2_contingency(ds.ix[ds_idx_list[k - 1:k + 1], 0:2])[0]
        if k < ds.shape[0] - 1:
            ds['chisq'].ix[ds_idx_list[k]] = chi2_contingency(ds.ix[ds_idx_list[k:k + 2], 0:2])[0]
        else:
            ds['chisq'].ix[ds_idx_list[k]] = 9999999.0

        inds_na = ds['bin'].apply(lambda b: b == ['-inf', '-inf'])
        end_time = datetime.datetime.now()
    end_time = datetime.datetime.now()
    if verbose:
        print("\n#--------------- 3. Merge bins by chisq rules Done --------------#")
        print(('  Duration of merge bins by chisq rules: {}'.format(end_time - start_time)))
        print(("  shape of the reduced table: {}".format(ds.shape)))

    '''
    #-------- chimerge: control bin size, except bin for NA (bin as ['-inf', '-inf']) -------#
    '''
    inds_na = ds['bin'].apply(lambda b: b == ['-inf', '-inf'])
    ds_na = ds[inds_na]

    ds = ds[~inds_na]
    pop_cut = df.shape[0] / 20
    ds['pop'] = ds[0] + ds[1]

    while ds['pop'].min() < pop_cut and ds.shape[0] > 2:
        # calculate chisquare statistic
        chisq = []
        for i in range(ds.shape[0] - 1):
            chisq.append(chi2_contingency(ds.iloc[i: i + 2,0:2])[0])
        chisq.append(9999999.0)
        ds['chisq'] = chisq

        # locate the smallest size by index
        ds_idx_list = list(ds.index)
        k = ds_idx_list.index(ds[ds['pop'] == ds['pop'].min()].index[0])
        if k == len(ds_idx_list) - 1:
            k -= 1
        elif ds['chisq'].ix[ds_idx_list[k]] > ds['chisq'].ix[ds_idx_list[k - 1]]:
            k -= 1

        # merge the adjacent binsssss, drop the second bin
        ds.ix[ds_idx_list[k], 0:2] = ds.ix[ds_idx_list[k], 0:2] + ds.ix[ds_idx_list[k + 1], 0:2]
        ds['bin'].ix[ds_idx_list[k]] = [ds['bin'].ix[ds_idx_list[k]][0], ds['bin'].ix[ds_idx_list[k + 1]][1]]
        ds = ds.drop(ds_idx_list[k + 1], axis=0)
        ds['pop'] = ds[0] + ds[1]

    # Add NaN bin back
    ds = pd.concat([ds, ds_na])
    ds['pop'] = ds[0] + ds[1]
    ds['dft_rto'] = 1.0 * ds[1] / ds['pop']

    if verbose:
    	print("\n  #--------------- 4. Done: merge bins by bin size --------------#")
    	print(("  shape of the reduced table: {}".format(ds.shape)))

    '''
    #------- get the reference table -------#
    '''
    ds = ds.reset_index(drop=True)

    ds['ref_table'] = None
    goodfreq = ds[0].sum()
    badfreq = ds[1].sum()
    ds[var + '_nwoe'] = ds.apply(lambda x: __woe_calc(x[1], x[0], goodfreq, badfreq), axis=1)
    ds['ref_table'] = ds['bin'].apply(lambda x: x[0] + '_' + x[1])

    # IVs
    ds[var + '_IVs'] = ds.apply(lambda x: x[var + '_nwoe'] * (float(x[1]) / badfreq - float(x[0]) / goodfreq), axis=1)

    # Set order of columns
    ds = ds.reindex_axis([0, 1, 'bin', 'dft_rto', 'chisq', 'pop', 'ref_table', var + '_nwoe', var + '_IVs'], axis=1)

    ref_table = {}
    ref_table = dict(list(zip(ds['ref_table'], ds[var + '_nwoe'])))
    ref_table['base'] = woe_calc_base(ds[1].sum(), ds[0].sum())
    end_time = datetime.datetime.now()

    if verbose:
    	print("\n  #--------------- get the reference table --------------#")
    	print(('  Duration of getting the reference table: {}'.format(end_time - start_time)))

    # IV
    # ref_table['IVs'] = ds[var + '_IVs']
    ref_table['IV'] = sum(ds[var + '_IVs'])

    dict_result = {'var_name': pd.Series([]), 'var_value': pd.Series([]), 'woe': pd.Series([])}
    df_result = pd.DataFrame(dict_result, index=list(range(0, len(list(ref_table.keys())))))
    df_result['var_name'] = var
    for index, item in enumerate(ref_table):
        df_result.ix[index, 'var_value'] = item
        df_result.ix[index, 'woe'] = ref_table.get(item)

    IV = df_result[df_result['var_value'] == 'IV']['woe'].values[0]
    return df_result, ds[var + '_IVs'], ds, IV


def iv_num(list_fields, data, verbose=True):
    df_iv = pd.DataFrame(columns=['field', 'coltype', 'caltype', 'iv'])
    for var in list_fields:
        print('#============ Calculating woe of {} ... ============#'.format(var))
        ref_table, b_iv, b_stat, IV = calc_numinal_woe(data, var, 'target', 5,verbose)
        df_iv.loc[len(df_iv)] = [var, data[var].dtype,'num', IV]
    return df_iv


def __woe_calc(bad, good, goodfreq, badfreq):
    target_rt = float(bad) / float(badfreq)
    non_target_rt = float(good) / float(goodfreq)
    if float(bad) != 0.0 and float(bad) / (float(bad) + float(good)) != 1.0:
        woe = np.log(float(target_rt / non_target_rt))
    elif target_rt == 0.0:
        woe = -99999999.0
    elif float(bad) / (float(bad) + float(good)) == 1.0:
        woe = 99999999.0
    return woe


def save_num_ref_table(list_num_ref_table, csvfile):
    num_ref_table = pd.concat(list_num_ref_table)
    num_ref_table.to_csv(csvfile, index=False, sep=',')


# nvlookup function
def __nvlookup(table, value):

    ref = None
    for key in list(table.keys()):
        if key.lower() == 'base' or key.lower() == 'iv':
            continue

        krange = list(map(np.float, key.split('_')))
        if pd.isnull(value) and krange[1] == -np.inf:
            ref = table[key]
            break

        if value >= krange[0] and value < krange[1]:
            if table[key] == '-99999999.0':
                ref = table['base']
            else:
                ref = table[key]
            break

    return ref


def apply_num_woe(datain, ref_table):
    datain_copy = datain.copy(deep=True)
    for vn in ref_table['var_name'].unique():
        dict_ref_table = dict(list(zip(ref_table.loc[ref_table['var_name'] == vn, 'var_value'].tolist(),
                                       ref_table.loc[ref_table['var_name'] == vn, 'woe'].tolist())))
        datain_copy[vn + '_nwoe'] = datain_copy[vn].apply(lambda x: __nvlookup(dict_ref_table, x))
    return datain_copy

















