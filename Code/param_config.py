# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:33:32 2017

@author: maoli
"""

import os
import numpy as np

### Config ###

class ParamConfig:
    def __init__(self,
                 feat_folder,
                 ):
        self.n_class = 4
        
        ## CV params
        self.n_runs = 3
        self.n_folds =3
        self.stratified_label = 'query'
        
        ## path
        self.data_folder ='../../Data'
        self.feat_fold = feat_folder
        self.original_train_data_path = "%s/d_train_20180102.csv"% self.data_folder
        self.original_pred_data_path = "%s/d_test_B_20180128.csv"% self.data_folder
        self.original_test_data2_path = "%s/test_b_non_missing_v2.csv"% self.data_folder






## initialize a param config					
config = ParamConfig(feat_folder="../../Feat/solution",
                     )