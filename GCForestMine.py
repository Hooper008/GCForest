
# coding: utf-8

# In[188]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn import model_selection
import itertools
import numpy as np


class gcForestMine(object):
    
    ##-----------函数1:按窗口大小切割数据-------------

    def window_sliced(window,X,y):
        len_seq = X.shape[1]
        X_sliced = [ ]
        for i in range(len_seq-1):
            X_sliced.append(X[:,i:i+window])
    
        y_sliced = y
    
        return X_sliced,y_sliced
#---------------------------------------------------
        

##----------函数2:random forest训练而得的类别概率----------------------------------------
    def prf_prob(X,y):
    
    #建立随机森林模型
        prf = RandomForestClassifier(n_estimators=30,oob_score=True, max_features='sqrt')
    
    
    #训练模型
        prf.fit(X,y)
   

    #使用K折交叉验证
        kfold = model_selection.KFold(n_splits=10, random_state=2)
        prf_results = model_selection.cross_val_score(prf, X, y, cv=kfold)
    
    #类别概率
        prf_prob = prf.oob_decision_function_
    
    
        return prf_prob
#-----------------------------------------------------------------------------------------------

##----------函数3:r完全andom forest训练而得的类别概率----------------------------------------
    def crf_prob(X,y):
    
    #建立完全随机森林模型
        crf = RandomForestClassifier(n_estimators=30,oob_score=True, max_features=1)
    
    #训练模型
        crf.fit(X,y)

    #使用K折交叉验证
        kfold = model_selection.KFold(n_splits=10, random_state=2)
        crf_results = model_selection.cross_val_score(crf, X, y, cv=kfold)
    
    #类别概率
        crf_prob = crf.oob_decision_function_
    
        return crf_prob
#-----------------------------------------------------------------------------------------------

#------------函数4:连接prf_prob
    def con_prf_prob(window,X,y):
        X_sliced , y =  window_sliced(window,X,y)
        n_iter = len_seq - window + 1
    
        con_prf_prob = [ ]
        for i in range(n_iter):
            con_prf_prob.append(prf_prob(X_sliced[i],y))

        return  np.concatenate(con_prf_prob,axis=1)
    
#---------------------------------------        
    

    #------------函数5:连接crf_prob
    def con_crf_prob(window,X,y):
        X_sliced , y =  window_sliced(window,X,y)
        n_iter = len_seq - window + 1
    
        con_crf_prob = [ ]
        for i in range(n_iter):
            con_crf_prob.append(crf_prob(X_sliced[i],y))

        return np.concatenate(con_crf_prob,axis=1)
    
#---------------------------------------


#-------------函数6:拼接两个随机森林的预测分类概率分布
    def mgs(window,X,y):
    
        con_prob = np.c_[con_prf_prob(window,X,y),con_crf_prob(window,X,y)]
    
        return con_prob

    


#-------------------------第一部分结束-----------------


#---------------------第二部分:cascade forest------------------

#----
##----------函数7:4个不同的random forest训练而得的类别概率的连接---------------------------------------
    def cas_rf_prob(X,y):
    
    
    
    #建立4个不同的随机森林模型
        crf1 = RandomForestClassifier(n_estimators=50,oob_score=True, max_features=1)
        crf2 = RandomForestClassifier(n_estimators=80,oob_score=True, max_features=1)
        prf1 = RandomForestClassifier(n_estimators=50,oob_score=True, max_features='sqrt')
        prf2 = RandomForestClassifier(n_estimators=80,oob_score=True, max_features='sqrt')
    
    #训练模型
        crf1.fit(X,y)
        crf2.fit(X,y)
        prf1.fit(X,y)
        prf2.fit(X,y)
     

    #使用K折交叉验证
        kfold = model_selection.KFold(n_splits=10, random_state=2)
        crf1_results = model_selection.cross_val_score(crf1, X, y, cv=kfold)
        crf2_results = model_selection.cross_val_score(crf2, X, y, cv=kfold)
        prf1_results = model_selection.cross_val_score(prf1, X, y, cv=kfold)
        prf2_results = model_selection.cross_val_score(prf2,X, y, cv=kfold)
    #类别概率
        crf1_prob = crf1.oob_decision_function_
        crf2_prob = crf2.oob_decision_function_
        prf1_prob = prf1.oob_decision_function_
        prf2_prob = prf2.oob_decision_function_
    
    #将四个随即森林的预测分类概率分布进行拼接
        prob = np.concatenate((crf1_prob,crf2_prob,prf1_prob,prf2_prob),axis=1)
    
    #每一层的预测结果
        ave_pred_prob = (crf1_prob+crf2_prob+prf1_prob+prf2_prob)/4
        max_pred_prob = np.argmax(ave_pred_prob,axis=1)
    
    #每一层的预测精度
        layer_accuracy = accuracy_score(y_true=y,y_pred=max_pred_prob)
   
        return prob,layer_accuracy,max_pred_prob

#---------------------


#-------函数8:增加层数训练------------------
    def cas(window,X,y):
    
        before = cas_rf_prob(mgs(window,X,y),y)
        accuracy_before = before[1]
    
        con_inform = np.concatenate((before[0] , mgs(window,X,y)) , axis=1)
        now = cas_rf_prob(con_inform,y)
        accuracy_now = now[1]
    
        while accuracy_before > accuracy_now+0.01:
            accuracy_before = accuracy_now
        
            before = now
            accuracy_before = before[1]
    
            con_inform = np.concatenate((before[0] , mgs(window,X,y)) , axis=1)
            now = cas_rf_prob(con_inform,y)
            accuracy_now = now[1]
            max_now = now[2]
        
        return max_now,accuracy_now
        
    
    
  
    
#--------------------------------------------------

