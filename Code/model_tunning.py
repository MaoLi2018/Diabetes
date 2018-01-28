# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:48:24 2018

@author: maoli
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
import os as os 



from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import xgboost as xgb
from sklearn.metrics import roc_curve as roc_curve
from sklearn.metrics import auc as auc
import matplotlib.pyplot as plt
import pylab as plb
from datetime import datetime as datetime


def evaluate_performance(all_target, predicted, toplot=True):
    fpr, tpr, thresholds = roc_curve(all_target, predicted)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    maxind = plb.find(tpr - fpr == ks)

    event_rate = sum(all_target) / 1.0 / all_target.shape[0]
    cum_total = tpr * event_rate + fpr * (1 - event_rate)
    minind = plb.find(abs(cum_total - event_rate) == min(abs(cum_total - event_rate)))
    if minind.shape[0] > 0:
        minind = minind[0]

    print('KS=' + str(round(ks, 3)) + ', AUC=' + str(round(roc_auc, 2)) + ', N=' + str(predicted.shape[0]))
    print('At threshold=' + str(round(event_rate, 3)) + ', TPR=' + str(round(tpr[minind], 2)) + ', ' + str(
        int(round(tpr[minind] * event_rate * all_target.shape[0]))) + ' out of ' + str(
        int(round(event_rate * all_target.shape[0]))))
    print('At threshold=' + str(round(event_rate, 3)) + ', FPR=' + str(round(fpr[minind], 2)) + ', ' + str(
        int(round(fpr[minind] * (1.0 - event_rate) * all_target.shape[0]))) + ' out of ' + str(
        int(round((1.0 - event_rate) * all_target.shape[0]))))

    # Score average by percentile
    binnum = 10
    ave_predict = np.zeros((binnum))
    ave_target = np.zeros((binnum))
    indices = np.argsort(predicted)
    binsize = int(round(predicted.shape[0] / 1.0 / binnum))
    for i in range(binnum):
        startind = i * binsize
        endind = min(predicted.shape[0], (i + 1) * binsize)
        ave_predict[i] = np.mean(predicted[indices[startind:endind]])
        ave_target[i] = np.mean(all_target[indices[startind:endind]])
    print('Ave_target: ' + str(ave_target))
    print('Ave_predicted: ' + str(ave_predict))

    if toplot:
        print('plot')
        # KS plot
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr)
        plt.hold
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        plt.title('KS=' + str(round(ks, 2)) + ' AUC=' + str(round(roc_auc, 2)), fontsize=20)
        plt.plot([fpr[maxind], fpr[maxind]], [fpr[maxind], tpr[maxind]], linewidth=4, color='r')
        plt.plot([fpr[minind]], [tpr[minind]], 'k.', markersize=10)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False positive', fontsize=20);
        plt.ylabel('True positive', fontsize=20);

        # print 'At threshold=' + str(round(event_rate, 3))
        # print str(round(fpr[minind],2))
        # print str(int(round(fpr[minind]*(1.0-event_rate)*all_target.shape[0])))
        # print str(int(round((1.0-event_rate)*all_target.shape[0])))


        # Score distribution score
        plt.subplot(1, 3, 2)
        # print predicted.columns
        plt.hist(predicted, bins=20)
        plt.hold
        plt.axvline(x=np.mean(predicted), linestyle='--')
        plt.axvline(x=np.mean(all_target), linestyle='--', color='g')
        plt.title('N=' + str(all_target.shape[0]) + ' Tru=' + str(round(np.mean(all_target), 3)) + ' Pred=' + str(
            round(np.mean(predicted), 3)), fontsize=20)
        plt.xlabel('Target rate', fontsize=20)
        plt.ylabel('Count', fontsize=20)

        plt.subplot(1, 3, 3)
        plt.plot(ave_predict, 'b.-', label='Prediction', markersize=5)
        plt.hold
        plt.plot(ave_target, 'r.-', label='Truth', markersize=5)
        plt.legend(loc='lower right')
        plt.xlabel('Percentile', fontsize=20)
        plt.ylabel('Target rate', fontsize=20)
        plt.show()

    return ks
