# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 00:04:15 2020

@author: 12583
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import os
from collections import defaultdict
warnings.filterwarnings('ignore')
y_Q3_3 = pd.read_csv('y_train_3/y_Q3_3.csv')
y_Q4_3 = pd.read_csv('y_train_3/y_Q4_3.csv')
aum_fils = os.listdir('x_train/aum_train/')+os.listdir('x_test/aum_test/')
aum = []
for f in aum_fils:
    print(f)
    mon = int((f.split('.')[0]).split('_')[-1].replace('m', ''))
    if mon>=7:
        tmp = pd.read_csv('x_train/aum_train/'+f)
        tmp['mon'] = mon
    else:
        tmp = pd.read_csv('x_test/aum_test/'+f)
        tmp['mon'] = mon+12
    aum.append(tmp)
aum = pd.concat(aum, axis=0, ignore_index=True)

behavior_fils = os.listdir('x_train/behavior_train/')+os.listdir('x_test/behavior_test/')
behavior = []
for f in behavior_fils:
    print(f)
    mon = int((f.split('.')[0]).split('_')[-1].replace('m', ''))
    if mon>=7:
        tmp = pd.read_csv('x_train/behavior_train/'+f)
        tmp['mon'] = mon
    else:
        tmp = pd.read_csv('x_test/behavior_test/'+f)
        tmp['mon'] = mon+12
    behavior.append(tmp)
behavior = pd.concat(behavior, axis=0, ignore_index=True)



event_fils = os.listdir('x_train/big_event_train/')+os.listdir('x_test/big_event_test/')
event = []
for f in event_fils:
    print(f)
    season = int((f.split('.')[0]).split('_')[-1].replace('Q', ''))
    if season>=3:
        tmp = pd.read_csv('x_train/big_event_train/'+f)
    else:
        tmp = pd.read_csv('x_test/big_event_test/'+f)
    tmp['season'] = season
    event.append(tmp)
event = pd.concat(event, axis=0, ignore_index=True)


cunkuan_fils = os.listdir('x_train/cunkuan_train/')+os.listdir('x_test/cunkuan_test/')
cunkuan = []
for f in cunkuan_fils:
    print(f)
    mon = int((f.split('.')[0]).split('_')[-1].replace('m', ''))
    if mon>=7:
        tmp = pd.read_csv('x_train/cunkuan_train/'+f)
        tmp['mon'] = mon
    else:
        tmp = pd.read_csv('x_test/cunkuan_test/'+f)
        tmp['mon'] = mon+12
    cunkuan.append(tmp)
cunkuan = pd.concat(cunkuan, axis=0, ignore_index=True)




cust_avli_Q3 = pd.read_csv('x_train/cust_avli_Q3.csv')
cust_avli_Q4 = pd.read_csv('x_train/cust_avli_Q4.csv')
cust_info_Q3 = pd.read_csv('x_train/cust_info_Q3.csv')
cust_info_Q4 = pd.read_csv('x_train/cust_info_Q4.csv')

cust_avli_Q1 = pd.read_csv('x_test/cust_avli_Q1.csv')
cust_info_Q1 = pd.read_csv('x_test/cust_info_Q1.csv')
###################################################################
bef4 = pd.read_csv('./bef4.csv').drop_duplicates()
bef3 = pd.read_csv('./bef3.csv').drop_duplicates()
train = y_Q4_3.copy()
train1 = y_Q3_3.copy()
test = cust_avli_Q1.copy()
train.shape, test.shape

y_Q3_3 = y_Q3_3.rename(columns={'label': 'bef_label'})
train = train.merge(y_Q3_3, on=['cust_no'], how='left')
train1['bef_label']=np.nan
y_Q4_3 = y_Q4_3.rename(columns={'label': 'bef_label'})
test = test.merge(y_Q4_3, on=['cust_no'], how='left')
#train['bef_label']=train['bef_label'].fillna(1)
'''bef3=bef3[['cust_no','label']].rename(columns={'label': 'bef_label'})

y_Q3_3 = y_Q3_3.rename(columns={'label': 'bef_label'})
y_Q3_3=pd.concat([y_Q3_3,bef3],ignore_index=True).drop_duplicates('cust_no')
train = train.merge(y_Q3_3, on=['cust_no'], how='left')
train1['bef_label']=np.nan
bef4=bef4[['cust_no','label']].rename(columns={'label': 'bef_label'})

y_Q4_3 = y_Q4_3.rename(columns={'label': 'bef_label'})
y_Q4_3=pd.concat([y_Q4_3,bef4],ignore_index=True).drop_duplicates('cust_no')
test = test.merge(y_Q4_3, on=['cust_no'], how='left')'''
#cohen_kappa_score((train['label']+1), (train['bef_label'].fillna(1)+1))
train = train.merge(cust_info_Q4, on=['cust_no'], how='left')
train1 = train1.merge(cust_info_Q3, on=['cust_no'], how='left')
test = test.merge(cust_info_Q1, on=['cust_no'], how='left')
for col in [f for f in train.select_dtypes('object').columns if f not in ['label', 'cust_no']]:
    train[col].fillna('-1', inplace=True)
    train1[col].fillna('-1', inplace=True)
    test[col].fillna('-1', inplace=True)
    le = LabelEncoder()
    le.fit(pd.concat([train[[col]],train1[[col]], test[[col]]], axis=0, ignore_index=True))
    train[col] = le.transform(train[col])
    train1[col] = le.transform(train1[col])
    test[col] = le.transform(test[col])

#############################################################################
cunkuan['C3'] = cunkuan['C1'] / cunkuan['C2']
cunkuan = cunkuan.sort_values(by=['cust_no', 'mon']).reset_index(drop=True)

agg_stat = {'C1': ['mean',  'sum', 'last'],
            'C2': ['mean', 'sum',  'last'],
            'C3': ['mean',  'sum', 'last']}
group_df = cunkuan[(cunkuan['mon']<=12)&(cunkuan['mon']>=10)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['c1'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
group_df.columns = [group_df.columns[0]]+['c0'+f[2:] for f in group_df.columns[1:]]
group_df1 = cunkuan[(cunkuan['mon']<=9)&(cunkuan['mon']>=7)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['c0'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)

train = train.merge(group_df1, on=['cust_no'], how='left')
group_df2 = cunkuan[(cunkuan['mon']<=15)&(cunkuan['mon']>=13)].groupby(['cust_no']).agg(agg_stat)
group_df2.columns = ['c1'+f[0]+'_'+f[1] for f in group_df2.columns]
group_df2.reset_index(inplace=True)
test = test.merge(group_df2, on=['cust_no'], how='left')
test = test.merge(group_df, on=['cust_no'], how='left')
group_df1 = cunkuan[(cunkuan['mon']<=9)&(cunkuan['mon']>=7)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['c1'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
for i in train.keys().copy():
    if i[:2]=='c1':
        train['c2'+i[2:]]=train[i]-train['c0'+i[2:]]
        train1['c0'+i[2:]]=0
        train1['c2'+i[2:]]=train[i]-train['c0'+i[2:]]
        test['c2'+i[2:]]=test[i]-test['c0'+i[2:]]

X_cols = [f for f in cunkuan.columns if f.startswith('C')]

X_cols = ['C1', 'C2', 'C3']

tmp = cunkuan[cunkuan['mon']==12].copy()
del tmp['mon']
train = train.merge(tmp, on=['cust_no'], how='left')

tmp = cunkuan[cunkuan['mon']==15].copy()
del tmp['mon']
test = test.merge(tmp, on=['cust_no'], how='left')
tmp = cunkuan[cunkuan['mon']==9].copy()
del tmp['mon']
train1=train1.merge(tmp, on=['cust_no'], how='left')


for i in range(1,6):
    
    tmp = cunkuan[cunkuan['mon']==12-i].copy()
    del tmp['mon']
    
    tmp.columns = [tmp.columns[0]]+['pc'+str(i)+f for f in tmp.columns[1:]]
    train = train.merge(tmp, on=['cust_no'], how='left')
    
    tmp = cunkuan[cunkuan['mon']==15-i].copy()
    del tmp['mon']
    
    tmp.columns = [tmp.columns[0]]+['pc'+str(i)+f for f in tmp.columns[1:]]
    test = test.merge(tmp, on=['cust_no'], how='left')
    if 9-i>=7:
        tmp = cunkuan[cunkuan['mon']==9-i].copy()
        del tmp['mon']
        
        tmp.columns = [tmp.columns[0]]+['pc'+str(i)+f for f in tmp.columns[1:]]
        train1 = train1.merge(tmp, on=['cust_no'], how='left')
    for j in train.keys().copy():
        if j[:3]=='pc'+str(i):
            train['change'+str(i)+j[3:]]=-train[j]+train[j[3:]]
            if 9-i>=7:
                train1['change'+str(i)+j[3:]]=-train1[j]+train1[j[3:]]
            else:
                train1[j]=0
                train1['change'+str(i)+j[3:]]=+train1[j[3:]]
            test['change'+str(i)+j[3:]]=-test[j]+test[j[3:]]




agg_stat = {           
            }
for i in ['C1', 'C2', 'C3']:
    agg_stat[i]=['mean', 'last']

group_df = cunkuan[(cunkuan['mon']==12)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['c3'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
#group_df.columns = [group_df.columns[0]]+['x0'+f[2:] for f in group_df.columns[1:]]
#group_df.reset_index(inplace=True)
group_df1 = cunkuan[(cunkuan['mon']<12)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['c4'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)

train = train.merge(group_df1, on=['cust_no'], how='left')
group_df2 = cunkuan[(cunkuan['mon']==15)].groupby(['cust_no']).agg(agg_stat)
group_df2.columns = ['c3'+f[0]+'_'+f[1] for f in group_df2.columns]
group_df2.reset_index(inplace=True)
test = test.merge(group_df2, on=['cust_no'], how='left')
group_df= cunkuan[(cunkuan['mon']<15)&(cunkuan['mon']>=10)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['c4'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
test = test.merge(group_df, on=['cust_no'], how='left')
group_df1 = cunkuan[(cunkuan['mon']<9)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['c4'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
group_df1 = cunkuan[(cunkuan['mon']==9)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['c3'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
for i in train.keys().copy():
    if i[:2]=='c3':
        train['c5'+i[2:]]=train[i]-train['c4'+i[2:]]
        train1['c5'+i[2:]]=train1[i]-train1['c4'+i[2:]]
        test['c5'+i[2:]]=test[i]-test['c4'+i[2:]]

##################################################################
X_cols = [f for f in aum.columns if f.startswith('X')]

aum['X_sum1'] = aum[X_cols].sum(axis=1)   
X_cols.remove('X7')

aum['X_sum'] = aum[X_cols].sum(axis=1)
aum['X_sum3'] = aum['X_sum']-aum['X3']
aum['X_sum4'] = aum['X_sum']-aum['X3']-aum['X8']
aum['X_sum5'] = aum['X8']-aum['X7']
aum['X_sum6'] = aum['X8']+aum['X3']-aum['X7']


aum['X_sum7'] = aum['X4']+aum['X5']+aum['X6']
aum['X_sum2'] = aum['X_sum']-aum['X7']
aum['X_num3'] = (aum['X_sum']>aum['X3']).astype(int)
aum['X_num2'] = (aum['X_sum']>aum['X7']).astype(int)


aum['X_num'] = (aum[X_cols]>0).sum(axis=1)

aum['X_num'] = (aum[X_cols]>0).sum(axis=1)
X_cols = [f for f in aum.columns if f.startswith('X')]
'''X_cols = [f for f in aum.columns if (f.startswith('X'))]
for i in X_cols:
    if '_' not in i:
        g=aum.groupby('cust_no')[i].mean().reset_index(name=i+'cust_no')
        aum=aum.merge(g,on='cust_no',how='left')
        aum[i+'culv']=aum[i]/aum[i+'cust_no']
      
X_cols = [f for f in aum.columns if (f.startswith('X'))]'''
am= aum.sort_values(by=['cust_no', 'mon']).reset_index(drop=True)
agg_stat = {           
            }
for i in X_cols:
    agg_stat[i]=['mean', 'sum', 'last']

group_df = am[(am['mon']<=12)&(am['mon']>=10)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['x1'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
group_df.columns = [group_df.columns[0]]+['x0'+f[2:] for f in group_df.columns[1:]]
#group_df.reset_index(inplace=True)
group_df1 = am[(am['mon']<=9)&(am['mon']>=7)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['x0'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)

train = train.merge(group_df1, on=['cust_no'], how='left')
group_df2 = am[(am['mon']<=15)&(am['mon']>=13)].groupby(['cust_no']).agg(agg_stat)
group_df2.columns = ['x1'+f[0]+'_'+f[1] for f in group_df2.columns]
group_df2.reset_index(inplace=True)
group_df1 = am[(am['mon']<=9)&(am['mon']>=7)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['x1'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
test = test.merge(group_df2, on=['cust_no'], how='left')
test = test.merge(group_df, on=['cust_no'], how='left')
for i in train.keys().copy():
    if i[:2]=='x1':
        train['x2'+i[2:]]=train[i]-train['x0'+i[2:]]
        train1['x2'+i[2:]]=train[i]-0
        train1['x0'+i[2:]]=0
        test['x2'+i[2:]]=test[i]-test['x0'+i[2:]]
agg_stat = {           
            }
for i in ['X_sum','X_num','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
    agg_stat[i]=['mean', 'last']
group_df = am[(am['mon']==12)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['x3'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
group_df.columns = [group_df.columns[0]]+['x0'+f[2:] for f in group_df.columns[1:]]
#group_df.reset_index(inplace=True)
group_df1 = am[(am['mon']<12)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['x4'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)

train = train.merge(group_df1, on=['cust_no'], how='left')
group_df2 = am[(am['mon']==15)].groupby(['cust_no']).agg(agg_stat)
group_df2.columns = ['x3'+f[0]+'_'+f[1] for f in group_df2.columns]
group_df2.reset_index(inplace=True)
test = test.merge(group_df2, on=['cust_no'], how='left')
group_df= am[(am['mon']<15)&(am['mon']>=10)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['x4'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
test = test.merge(group_df, on=['cust_no'], how='left')
group_df1 = am[(am['mon']<9)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['x4'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
group_df1 = am[(am['mon']==9)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['x3'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
for i in train.keys().copy():
    if i[:2]=='x3':
        train['x5'+i[2:]]=train[i]-train['x4'+i[2:]]
        train1['x5'+i[2:]]=train1[i]-train1['x4'+i[2:]]
        test['x5'+i[2:]]=test[i]-test['x4'+i[2:]]
######################################################
'''am= aum.sort_values(by=['cust_no', 'mon']).reset_index(drop=True)

agg_stat = {           
            }
for i in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
    agg_stat[i]=['mean']

group_df = am.groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['xn1'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
test = test.merge(group_df, on=['cust_no'], how='left')'''
###################################################################

#aum['X_bi'] = aum['X3']/aum['X_sum']
X_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
tmp = aum[aum['mon']==12].copy()
del tmp['mon']
train = train.merge(tmp, on=['cust_no'], how='left')

tmp = aum[aum['mon']==15].copy()
del tmp['mon']
test = test.merge(tmp, on=['cust_no'], how='left')
tmp = aum[aum['mon']==9].copy()
del tmp['mon']
train1 = train1.merge(tmp, on=['cust_no'], how='left')

for i in range(1,6):
    
    tmp = aum[aum['mon']==12-i].copy()
    del tmp['mon']
    
    tmp.columns = [tmp.columns[0]]+['px'+str(i)+f for f in tmp.columns[1:]]
    train = train.merge(tmp, on=['cust_no'], how='left')
    
    tmp = aum[aum['mon']==15-i].copy()
    del tmp['mon']
    
    tmp.columns = [tmp.columns[0]]+['px'+str(i)+f for f in tmp.columns[1:]]
    test = test.merge(tmp, on=['cust_no'], how='left')
    if 9-i>=7:
        tmp = aum[aum['mon']==9-i].copy()
        del tmp['mon']
        
        tmp.columns = [tmp.columns[0]]+['px'+str(i)+f for f in tmp.columns[1:]]
        train1 = train1.merge(tmp, on=['cust_no'], how='left')
    for j in train.keys().copy():
        if j[:3]=='px'+str(i):
            train['change'+str(i)+j[3:]]=-train[j]+train[j[3:]]
            if 9-i>=7:
                train1['change'+str(i)+j[3:]]=-train1[j]+train1[j[3:]]
            else:
                train1[j]=0
                train1['change'+str(i)+j[3:]]=+train1[j[3:]]
            test['change'+str(i)+j[3:]]=-test[j]+test[j[3:]]
ch_cols = [f for f in aum.columns if f.startswith('change')]
train['change']=train[ch_cols ].sum(axis=1) 
test['change']=test[ch_cols ].sum(axis=1) 
train1['change']=train1[ch_cols ].sum(axis=1)   
ch_cols = [f for f in aum.columns if f.startswith('change')]
for i in ch_cols:
    group_df = train.groupby(['cust_no'])[i].mean().reset_index('CH'+i)
    train = train.merge(group_df, on=['cust_no'], how='left')
    group_df = train1.groupby(['cust_no'])[i].mean().reset_index('CH'+i)
    train1 = train1.merge(group_df, on=['cust_no'], how='left')
    group_df = test.groupby(['cust_no'])[i].mean().reset_index('CH'+i)
    test = test.merge(group_df, on=['cust_no'], how='left')
##################################################################

behavior['B5-B3'] = behavior['B5'] - behavior['B3']
behavior['B2-B4'] = behavior['B2'] - behavior['B4']
behavior=behavior[['cust_no','B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',  'mon', 'B5-B3','B2-B4']]
tmp = behavior[behavior['mon']==12].copy()
del tmp['mon']
train = train.merge(tmp, on=['cust_no'], how='left')

tmp = behavior[behavior['mon']==9].copy()
del tmp['mon']
train1 = train1.merge(tmp, on=['cust_no'], how='left')


tmp = behavior[behavior['mon']==15].copy()
del tmp['mon']
test = test.merge(tmp, on=['cust_no'], how='left')



X_cols = [f for f in behavior.columns if f.startswith('B')]
X_cols.remove('B6')
agg_stat = {           
            }
for i in X_cols:
    agg_stat[i]=['mean', 'sum', 'last']

group_df = behavior[(behavior['mon']<=12)&(behavior['mon']>=10)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['b1'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
group_df.columns = [group_df.columns[0]]+['b0'+f[2:] for f in group_df.columns[1:]]
#group_df.reset_index(inplace=True)
group_df1 = behavior[(behavior['mon']<=9)&(behavior['mon']>=7)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['b0'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)

train = train.merge(group_df1, on=['cust_no'], how='left')
group_df2 = behavior[(behavior['mon']<=15)&(behavior['mon']>=13)].groupby(['cust_no']).agg(agg_stat)
group_df2.columns = ['b1'+f[0]+'_'+f[1] for f in group_df2.columns]
group_df2.reset_index(inplace=True)
test = test.merge(group_df2, on=['cust_no'], how='left')
test = test.merge(group_df, on=['cust_no'], how='left')
group_df1 = behavior[(behavior['mon']<=9)&(behavior['mon']>=7)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['b1'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')

for i in train.keys().copy():
    if i[:2]=='b1':
        train['b2'+i[2:]]=train[i]-train['b0'+i[2:]]
        train1['b2'+i[2:]]=train[i]-0
        train['b0'+i[2:]]=0
        test['b2'+i[2:]]=test[i]-test['b0'+i[2:]]

agg_stat = {           
            }
for i in X_cols:
    agg_stat[i]=['mean',  'last']
group_df = behavior[(behavior['mon']==12)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['b3'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
train = train.merge(group_df, on=['cust_no'], how='left')
#group_df.columns = [group_df.columns[0]]+['x0'+f[2:] for f in group_df.columns[1:]]
#group_df.reset_index(inplace=True)
group_df1 = behavior[(behavior['mon']<12)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['b4'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)

train = train.merge(group_df1, on=['cust_no'], how='left')
group_df2 = behavior[(behavior['mon']==15)].groupby(['cust_no']).agg(agg_stat)
group_df2.columns = ['b3'+f[0]+'_'+f[1] for f in group_df2.columns]
group_df2.reset_index(inplace=True)
test = test.merge(group_df2, on=['cust_no'], how='left')
group_df= behavior[(behavior['mon']<15)&(behavior['mon']>=10)].groupby(['cust_no']).agg(agg_stat)
group_df.columns = ['b4'+f[0]+'_'+f[1] for f in group_df.columns]
group_df.reset_index(inplace=True)
test = test.merge(group_df, on=['cust_no'], how='left')
group_df1 = behavior[(behavior['mon']<9)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['b4'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
group_df1 = behavior[(behavior['mon']==9)].groupby(['cust_no']).agg(agg_stat)
group_df1.columns = ['b3'+f[0]+'_'+f[1] for f in group_df1.columns]
group_df1.reset_index(inplace=True)
train1 = train1.merge(group_df1, on=['cust_no'], how='left')
for i in train.keys().copy():
    if i[:2]=='b3':
        train['b5'+i[2:]]=train[i]-train['b4'+i[2:]]
        train1['b5'+i[2:]]=train1[i]-train1['b4'+i[2:]]
        test['b5'+i[2:]]=test[i]-test['b4'+i[2:]]

###################################################################
train['B6_gap'] = (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(train['B6'])).dt.total_seconds()#//(24*60*60)
#train1['B6_gap'] = (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(train1['B6'])).dt.total_seconds()#//(24*60*60)

test['B6_gap'] = (pd.to_datetime('2020-04-01 00:00:00') - pd.to_datetime(test['B6'])).dt.total_seconds()#//(24*60*60)
train['B6_hour'] = pd.to_datetime(train['B6']).dt.hour
#train1['B6_hour'] = pd.to_datetime(train1['B6']).dt.hour
test['B6_hour'] = pd.to_datetime(test['B6']).dt.hour
#train['B6_day'] =31- pd.to_datetime(train['B6']).dt.day
#train1['B6_day'] = pd.to_datetime(train1['B6']).dt.day
#test['B6_day'] = 31-pd.to_datetime(test['B6']).dt.day
event['E15-17']=event['E15']-event['E17']
E_cols = [f for f in event.columns if f.startswith('E')]
event['event_num'] = len(E_cols) - event[E_cols].isnull().sum(axis=1)

tmp = event[event['season']==4].copy()
del tmp['season']
tmp.columns =[tmp.columns[0]]+['e1'+f for f in tmp.columns[1:]]
train = train.merge(tmp, on=['cust_no'], how='left')
tmp = event[event['season']==3].copy()
del tmp['season']
tmp.columns =[tmp.columns[0]]+['e0'+f for f in tmp.columns[1:]]

train = train.merge(tmp, on=['cust_no'], how='left')
tmp = event[event['season']==1].copy()
del tmp['season']
tmp.columns =[tmp.columns[0]]+['e1'+f for f in tmp.columns[1:]]
test = test.merge(tmp, on=['cust_no'], how='left')
tmp = event[event['season']==4].copy()
del tmp['season']
tmp.columns =[tmp.columns[0]]+['e0'+f for f in tmp.columns[1:]]
test = test.merge(tmp, on=['cust_no'], how='left')
tmp = event[event['season']==3].copy()
del tmp['season']
tmp.columns =[tmp.columns[0]]+['e1'+f for f in tmp.columns[1:]]
train1 = train1.merge(tmp, on=['cust_no'], how='left')
for col in E_cols:
    if col not in ['E15', 'E17']:
        train['e0'+col] = ((pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(train['e0'+col])).dt.days)//30
        test['e0'+col] =( (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(test['e0'+col])).dt.days)//30
        train['e1'+col] =( (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(train['e1'+col])).dt.days)//30

        test['e1'+col] = ((pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(test['e1'+col])).dt.days)//30
        train1['e1'+col] =( (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(train['e1'+col])).dt.days)//30
    train1['e0'+col] =0
    #train1[col] = (pd.to_datetime('2020-01-01 00:00:00') - pd.to_datetime(train1[col])).dt.days
    #train['e2'+col] = train['e1'+col]-train['e0'+col]
    #test['e2'+col] = test['e1'+col]-test['e0'+col]
    train['e2'+col] = train['e1'+col]-train['e0'+col]
    train1['e2'+col] = train1['e1'+col]-train1['e0'+col]
    test['e2'+col] = test['e1'+col]-test['e0'+col]
############################################################################


###########################################################################################################
def kappa(preds, train_data):
    y_true = train_data.label
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    score = cohen_kappa_score(y_true, preds)
    return 'kappa', score, True

def LGB_classfication_model(train, target, test, k,fx=[]):
    
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
    oof_preds = np.zeros(train.shape[0])
    oof_probs = np.zeros((train.shape[0], 3))
    output_preds = []
    feature_importance_df = pd.DataFrame()
    offline_score = []
    train['weight']=train.label.map({0:1.03,1:0.58,-1:1})
    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train.iloc[train_index, :], train.iloc[test_index, :]
        test0=test.copy()
        #train_X, test_X,test0=rate(train_X, test_X,test0)
        
        feats = [f for f in train_X.columns if f[0]!='ex' and 'max' not in f and 'weight' not in f and 'min' not in f and f[0]!='b2' and 'code' not in f and f not in fx and f not in ['cust_no','B6_gap', 'label','B6_hour', 'I7', 'I9', 'B6']]
        print('Current num of features:', len(feats))
        dtrain = lgb.Dataset(train_X[feats],
                             label=train_y,weight=train_X['weight'].values.flatten(order='F'))
        dval = lgb.Dataset(test_X[feats],
                           label=test_y)
        parameters = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'None',
            'num_leaves': 64,
            'num_class': 3,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 128,
            'verbose': -1,
            'nthread': 12
        }
        lgb_model = lgb.train(
            parameters,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=kappa,
        )
        oof_probs[test_index] = lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration)
        oof_preds[test_index] = np.argmax(lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration), axis=1)
        offline_score.append(lgb_model.best_score['valid_0']['kappa'])
        output_preds.append(lgb_model.predict(test0[feats], num_iteration=lgb_model.best_iteration))
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-KAPPA score:%.6f, OOF-STD:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(30))
    print('confusion matrix:')
    print(confusion_matrix(target, oof_preds))
    print('classfication report:')
    print(classification_report(target, oof_preds))

    return output_preds, oof_probs, np.mean(offline_score),feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False)
###############################################################################
ln=train.shape[0]
#train1=train
train1=pd.concat([train,train1],ignore_index=True)
target = train1['label'] + 1
lgb_preds, lgb_oof, lgb_score,f = LGB_classfication_model(train1, target, test, 5)
lgb_preds, lgb_oof, lgb_score,f = LGB_classfication_model(train1, target, test, 5,fx=list(f[f<100].keys()))

target1=target[:ln]
lgb_oof=lgb_oof[:ln]
cohen_kappa_score(target1,  np.argmax(lgb_oof, axis=1))
sub_df = test[['cust_no']].copy()
sub_df['label'] = np.argmax(np.mean(lgb_preds, axis=0), axis=1) - 1
sub_df['label'].value_counts(normalize=True)
sub_df.to_csv('baseline_sub1.csv', index=False)


sub_df = test[['cust_no']].copy()
sub_df['label0'] = np.mean(lgb_preds, axis=0)[:,0]
sub_df['label1'] = np.mean(lgb_preds, axis=0)[:,1]
sub_df['label2'] = np.mean(lgb_preds, axis=0)[:,2]

sub_df.to_csv('p1baseline_sub1.csv', index=False)




















