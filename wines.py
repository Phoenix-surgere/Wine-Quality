# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:09:03 2019

@author: black

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
seed = 43
random.seed(seed)
np.random.seed(seed)

red_wine = pd.read_csv('winedata\winequality_red.csv')
white_wine = pd.read_csv('winedata\winequality_white.csv')

white_wine['color'] = 0; red_wine['color'] = 1

wines = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
wines=wines.reindex(np.random.permutation(wines.index)).reset_index(drop=True)

plt.style.use('dark_background')

#Initial EDA - NA values and very basic info
print(wines.isna().sum())
print(wines.info())
print(wines.quality.describe()) #mean of quality around 5.8
print(wines.quality.unique())  #Unique qualities
print(wines.quality.value_counts())

wines.quality.value_counts().plot(kind='bar', title='Quality Distribution'); plt.show()
wines.quality.value_counts().plot.pie(autopct='%1.1f%%', figsize=(8,6)); plt.show()

#sns.jointplot(y='quality', x='alcohol', data=wines)
corrmatrix = wines.corr()
mask = np.triu(np.ones_like(corrmatrix, dtype=bool))
plt.figure(figsize=(12,8))
sns.heatmap(corrmatrix, annot=True, mask=mask); plt.show()

corrmatrix.quality.sort_values()[:-1].plot(kind='bar', title='Quality  Correlations'); plt.show()
sns.violinplot(x='quality', y='alcohol', data=wines); plt.show()
sns.boxplot(x='quality', y='pH', data=wines); plt.show()
sns.boxplot(x='quality', y='citric acid', data=wines); plt.show()
sns.boxplot(x='quality', y='density', data=wines); plt.show()

#Need to create benchmark - perhaps naively only predict most frequent class? -SOLVED: Use sklearn's dummy
#Confusion matrix not convenient, too many classes, will stick to single number metrics - Need to do that for dim reduced as well?
from sklearn.dummy import DummyClassifier as DC
dc = DC(random_state=seed)
dc.fit(X_train, y_train)
print(f1(y_test, dc.predict(X_test), average='weighted'))
print(acc(y_test, dc.predict(X_test)))


#REMINDER: Take into account large class imbalance - eg accuracy NOT good a metric! SOLVED - Resampling, F1 score, recall_score
#Tactics to be used (compared): group outlier groups into one and upsample them collecticely
X = pd.concat([X_train, y_train], axis=1)
print(X.quality.value_counts())

def imp(x):
    if x <= 4:
        return 3
    elif x >= 8:
        return 9
    else: 
        return x

X.quality = X.quality.apply(imp)  
print(X.quality.value_counts())

terrible = X[X.quality==3]
good = X[X.quality == 9]
good_up = resample(good, replace=True, n_samples=550, random_state=seed)


#Dimension reduction vs w/out Dimension reduction
from helper_funcs import model_reduce, plot_roc_auc, plot_precision_recall,fit_metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import RandomizedSearchCV as RSCV
qualities = wines['quality']
wines.drop(columns=['quality'],inplace=True)
X_train, X_test, y_train, y_test = tts(wines, qualities, test_size=0.2, 
                                    random_state=seed, stratify=qualities)
scaler = SS()

X_train.loc[:,X_train.columns] = scaler.fit_transform(X_train.loc[:,X_train.columns])
X_test.loc[:, X_test.columns] = scaler.transform(X_test.loc[:,X_test.columns])

from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
import xgboost as xgb
from sklearn.metrics import r2_score

logit = LR(random_state=seed,solver='lbfgs',max_iter=300, multi_class='auto')
rf = RFC(n_estimators=250, random_state=seed)
gb = GBC(n_estimators=250, random_state=seed)
xgb = xgb.XGBClassifier(objective='reg:logistic', n_estimators=250, seed=42)
models = [logit, rf, gb, xgb]
masks = []

rf_distributions = {'n_estimators': [100, 200, 300, 500],
                 'max_features': ['auto','sqrt', 'log2'],
                 'min_samples_leaf':  [0.10, 0.12, 0.16],
                 'max_depth': [10,20,30, 40] }

rfclf = RSCV(rf, rf_distributions, n_iter=80, cv=3, random_state=seed, 
             verbose=1, scoring='f1_micro')
rfclf.fit(X_train, y_train)
rf_preds = rfclf.predict(X_test)
print(r2_score(y_test, rf_preds))

for model in models:
    masks.append(model_reduce(model, 5, X_train, y_train))

votes = np.sum(masks, axis=0)
print(dict(zip(X_train.columns,votes)))
meta_mask = votes >= 3

wines_reduced = wines.loc[:, meta_mask]
print(wines_reduced.columns)

X_train, X_test, y_train, y_test = tts(wines_reduced, qualities, test_size=0.2,
                                     random_state=seed, stratify=qualities)

                 
rfclf = RSCV(rf, rf_distributions, n_iter=80, cv=3, random_state=seed, 
             verbose=1, scoring='f1_micro')
rfclf.fit(X_train, y_train)
rf_preds_red = rfclf.predict(X_test)
print(r2_score(y_test, rf_preds_red))
