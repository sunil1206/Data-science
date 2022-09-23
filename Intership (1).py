#!/usr/bin/env python
# coding: utf-8

# In[175]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize']= (20,10)
from datetime import date
import seaborn as sns
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[176]:


df=pd.read_csv('HRDataset_v14.csv')
df


# # Basic information

# In[177]:


df1=df.copy()


# In[178]:


df.isna().sum()


# fill date of termination column with 2020. consider nan value as still working.
# 
# for calculating experience

# In[179]:


df.groupby(['State']).mean().sort_values("Salary", ascending = False).plot.bar();


# In[180]:


df[['DateofTermination']]


# In[181]:


import datetime


# In[182]:


year=['DOB','DateofHire','LastPerformanceReview_Date','DateofTermination']
for i in year:
    df1[i]=pd.to_datetime(df1[i])
    df1[i]=pd.DatetimeIndex(df1[i]).year


# In[ ]:





# In[183]:


df1['DateofTermination'].fillna(2022, inplace = True)


# In[184]:


df1[['DOB','DateofHire','LastPerformanceReview_Date','DateofTermination']]


# In[185]:


df1[['DOB','DateofHire','LastPerformanceReview_Date','DateofTermination']]


# In[186]:


def fix_time(f):
    if f>2020:
        f=f-100
    else:
        f=f
    return f


# In[187]:


df1['DOB']<=0


# In[188]:



df1['DOB']= df1['DOB'].apply(fix_time).astype('int')


# # Feature enginering

# Find Age. consider 2022 as base year for calculating age 

# In[189]:


df1['Age']= (date.today().year-df1.DOB).astype('int')


# In[190]:


df1.loc[df1['Age']<=40]


# In[191]:


df1['Experience']=(df1['DateofTermination']-df1['DateofHire']).astype('int')


# In[192]:


for i in range(len(df1['DateofTermination'])):
    if i<2019:
        df1['carrierbreak']=2022-df1['DateofTermination']
    else:
        df1['carrierbreak']=0  


# In[193]:


[df1['carrierbreak']]


# In[194]:


df1.carrierbreak.value_counts()


# In[195]:


for i in df1:
    print('Unique value of ' + i,len(df1[i].unique()))


# In[196]:


table = pd.pivot_table(df1, values='EmpID', index=['Position','Department'],aggfunc=lambda x: len(x.unique()))
table.rename({"EmpID":"Count of Employees"},axis=1,inplace=True)
table


# In[197]:


df1[['Position','Department','LastPerformanceReview_Date','DateofTermination']]


# In[198]:


df1['Age'].unique()


# age

# In[199]:


# X_train_data = df1['Age']

# bins= [25,30,35,40,45,50,55,60,65,70]
# labels = ["25-30",'30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70']
# df1['AgeGroup'] = pd.cut(df1['Age'], bins=bins, labels=labels, right=False)
# df1['AgeGroup']

bins= [20,30,40,50,60,70,80]
labels = ["20-30",'30-40','40-50','50-60','60-70','80-90']
df1['AgeGroup'] = pd.cut(df1['Age'], bins=bins, labels=labels, right=False)
df1['AgeGroup']


# In[ ]:





# In[200]:


df.Salary.min()


# In[201]:


df.Salary.max()


# In[202]:


# bins2= [40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000]
# labels2 = ['40000-60000','60000-80000','80000-100000','100000-120000','120000-140000','140000-160000','160000-180000','180000-200000','200000-220000','220000-240000','240000-260000']
# df1['SalaryGroup'] = pd.cut(df1['Salary'], bins=bins2, labels=labels2, right=False)
# df1['SalaryGroup']

bins2= [40000,70000,100000,130000,160000,190000,220000,250000,280000]
labels2 = ['40000-70000','70000-100000','100000-130000','130000-160000','160000-190000','190000-220000','220000-250000','250000-2800000']
df1['SalaryGroup'] = pd.cut(df1['Salary'], bins=bins2, labels=labels2, right=False)
df1['SalaryGroup']


# In[203]:


table = pd.pivot_table(df1, values='EmpID', index=['Position','SalaryGroup'],aggfunc=lambda x: len(x.unique()))
table.rename({"EmpID":"Count of Employees"},axis=1,inplace=True)
table


# In[204]:


df1.loc[df1['Salary']>=100000]


# In[205]:


df1.Experience.unique()


# In[206]:


# bins3= [0,3,8,12,18]
# labels3 = ['Fresher','Experienced','Highly experienced','efficient']
# df1['ExperiencedGroup'] = pd.cut(df1['Experience'], bins=bins3, labels=labels3, right=False)
# df1['ExperiencedGroup']


# In[207]:


df1.corr()


# In[208]:


# df2[df2.columns].corr()['SalaryGroup'][:]


# In[209]:


position= df1[['Position','EmpID']].groupby(by=['Position'],as_index=False).count()
position.rename(columns={'Position':'Position','EmpID':'counts'},inplace=True)


# In[210]:


position['%']=(position['counts']/position['counts'].sum())*100


# In[211]:


position


# In[212]:


position['counts'].describe()


# In[213]:


min_count= position.loc[(position['counts']==1)]


# In[214]:


min_count['Position'].unique()


# In[215]:


len(position.loc[(position['counts']==1)])


# In[216]:


plt.figure(figsize=(16,8))
sns.countplot(x='Position', data=df1, palette='viridis')


# In[217]:


position_stats = df1['Position'].value_counts(ascending=False)
position_stats


# In[ ]:





# In[218]:


position_stats_less_than_1 = position[position['counts']<=1]
position_stats_less_than_1


# In[219]:


df1.Position = df1.Position.apply(lambda x: 'other' if x in position_stats_less_than_1['Position'].value_counts() else x)


# In[220]:


len(df1.Position.value_counts())


# In[221]:


df1.loc[df1['Position']=='other']


# In[222]:


age= df1[['AgeGroup','EmpID']].groupby(by=['AgeGroup'],as_index=False).count()
age.rename(columns={'AgeGroup':'Age','EmpID':'counts'},inplace=True)
age


# In[223]:


salary= df1[['SalaryGroup','EmpID']].groupby(by=['SalaryGroup'],as_index=False).count()
salary.rename(columns={'SalaryGroup':'SalaryGroup','EmpID':'counts'},inplace=True)
salary


# In[224]:


# # y=merged['Salary']

# y=df2['SalaryGroup']

# x=df2.drop(['Salary','SalaryGroup'],axis=1)


# # Drop columns

# In[225]:


df1.columns


# In[226]:


df2=df1.drop(['Employee_Name', 'EmpID', 
      'Termd', 'PositionID',  'Zip', 'DOB',
       'Sex', 'MaritalDesc', 'HispanicLatino', 'RaceDesc',
       'DateofHire', 'DateofTermination', 'TermReason', 'EmploymentStatus',
      'ManagerName', 'ManagerID', 'RecruitmentSource',
        'EmpSatisfaction','Department',
      'LastPerformanceReview_Date', 'DaysLateLast30',
       'Absences','Age'],axis=1)


# In[227]:


df5=df1.drop(['MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID', 'FromDiversityJobFairID', 'Termd', 'State', 'Zip', 'CitizenDesc', 'HispanicLatino', 'TermReason','RecruitmentSource','EngagementSurvey','EmpSatisfaction','DaysLateLast30','Absences'],axis=1)


# In[228]:


df3=df2.copy()


# In[229]:


df4=df1[['Department','Position','RaceDesc','SpecialProjectsCount','Age','PerfScoreID','Experience','SalaryGroup','Salary','AgeGroup]]


# In[230]:


df2['AgeGroup'][41]


# In[231]:


df3['AgeGroup'][41]


# # Label encoding

# In[ ]:





# In[ ]:





# In[232]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df2['Position']= le.fit_transform(df2['Position'])
df2['AgeGroup']= le.fit_transform(df2['AgeGroup'])
df2['SalaryGroup']= le.fit_transform(df2['SalaryGroup'])
df2['State']= le.fit_transform(df2['State'])
df2['CitizenDesc']= le.fit_transform(df2['CitizenDesc'])
df2['PerformanceScore']= le.fit_transform(df2['PerformanceScore'])                                     
                                     
# df1['ExperiencedGroup']= le.fit_transform(df1['ExperiencedGroup'])


# In[ ]:





# In[233]:


# df2=le.fit_transform(df2)


# In[234]:


for i in df2:
    print('Unique value of ' + i,len(df2[i].unique()))


# In[ ]:





# In[235]:


df1.corr()


# In[236]:


df2


# In[237]:


# dummies = pd.get_dummies(df[['Position','CitizenDesc','Department','PerformanceScore']],drop_first=True)
# df2.drop(['Position','CitizenDesc','PerformanceScore','Department'],axis=1,inplace=True)
# merged=pd.concat([df,dummies],axis=1)


# In[ ]:





# # Scaling

# In[238]:


# y=merged['Salary']

y=df2['SalaryGroup']

X=df2.drop(['Salary','SalaryGroup'],axis=1)


# In[239]:


df2.columns


# In[240]:


df2


# In[ ]:





# In[241]:


from sklearn.preprocessing import StandardScaler


# In[242]:


# sc=StandardScaler()
# x=sc.fit_transform(x)


# In[243]:


from sklearn.preprocessing import MinMaxScaler

min_=MinMaxScaler()
X=min_.fit_transform(x)


# In[244]:


from sklearn.decomposition import PCA


# In[245]:


# pca = PCA(n_components=5)
# X=pca.fit_transform(x)


# # Model

# In[246]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)


# In[247]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred3 = svc.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred3)


# In[248]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X=pca.fit_transform(x)


# In[249]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=42)


# In[250]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

models = {
    " K-Nearest Neighbors": KNeighborsClassifier(),
    " Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "         Decision Tree": DecisionTreeClassifier(),
    "        Neural Network": MLPClassifier()
}


for name, model in models.items():
    model.fit(X_train, y_train)

for name, model in models.items():
    print(name + " Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100)) 


# In[251]:


df3


# In[252]:


df3['State']= le.fit_transform(df3['State'])


# In[253]:


df3['SalaryGroup']= le.fit_transform(df3['SalaryGroup'])


# In[254]:


dummies = pd.get_dummies(df3[['Position','CitizenDesc','PerformanceScore','AgeGroup']],drop_first=True)
df3.drop(['Position','CitizenDesc','PerformanceScore','PerformanceScore','AgeGroup'],axis=1,inplace=True)
merged=pd.concat([df3,dummies],axis=1)


# In[255]:


# df3['State']= le.fit_transform(df3['State'])


# In[256]:



y=merged['SalaryGroup']

x=merged.drop(['Salary','SalaryGroup'],axis=1)


# In[257]:


df3


# In[258]:


y


# In[259]:


from sklearn.preprocessing import MinMaxScaler

min_=MinMaxScaler()
X=min_.fit_transform(x)


# In[260]:


pca = PCA(n_components=5)
X=pca.fit_transform(X)


# In[261]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)


# In[262]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train) #model fitting


# In[263]:


Y0_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[264]:


from sklearn.metrics import confusion_matrix
#confusion matrix
confusion_matrix = confusion_matrix(y_test, Y0_pred)
print(confusion_matrix) #correction prediction(6989+992), incorrect prediction(561+1227)


# In[265]:


print(classification_report(y_test, Y0_pred))


# In[266]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

models = {
    " K-Nearest Neighbors": KNeighborsClassifier(),
    " Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "         Decision Tree": DecisionTreeClassifier(),
    "        Neural Network": MLPClassifier()
}


for name, model in models.items():
    model.fit(X_train, y_train)

for name, model in models.items():
    print(name + " Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100)) 


# In[267]:


df2['AgeGroup'][41]


# In[268]:


y


# In[269]:


plt.rc("font", size=14)


# In[270]:


# logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# # plt.plot([0, 1], [0, 1],'r--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# # plt.savefig('Log_ROC')
# plt.show() #ROC curve stays above average line. the model is good.


# In[271]:


from sklearn.model_selection import GridSearchCV


# In[272]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}


# In[273]:


grid = GridSearchCV(SVC(),param_grid)
grid.fit(X_train,y_train)


# In[274]:


print(grid.best_estimator_)


# In[275]:


grid_predictions = grid.predict(X_test)
# print(confusion_matrix(y_test,grid_predictions))
print(" Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100)) 
print(classification_report(y_test,grid_predictions))


# In[276]:


df1.columns


# In[277]:


def predict_salary(AgeGroup,SpecialProjectsCount,Experience):
    loc_index= np.where(X.columns==AgeGroup)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=SpecialProjectsCount
    x[1]=Experience
    if loc_index >=0:
        x[loc_index] = 1
        
    return grid_predictions.predict([x])[0]    


# In[278]:


X.columns


# In[279]:


# def predict_price(location,sqft,bath,bhk):    
#     loc_index = np.where(X.columns==location)[0][0]

#     x = np.zeros(len(X.columns))
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk
#     if loc_index >= 0:
#         x[loc_index] = 1

#     return lr_clf.predict([x])[0]


# In[ ]:





# In[280]:


predict_salary('45-50',3, 5)


# In[281]:


df2.columns


# In[282]:


# y=merged['Salary']

Y=df2['Salary']

x=df2[['DeptID','Position','SpecialProjectsCount']]


# In[283]:


from sklearn.preprocessing import MinMaxScaler

min_=MinMaxScaler()
Xw=min_.fit_transform(x)


# In[284]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(Xw,Y, test_size=0.2, random_state=42)


# In[285]:


from sklearn.linear_model import LinearRegression
lin_re = LinearRegression()
lin_re.fit(X_train,y_train)
y1_pred = lin_re.predict(X_test)


# In[286]:


from sklearn.metrics import mean_squared_error, r2_score
print('r^2 value of the linear model:',r2_score(y_test,y1_pred))


# In[287]:


diffence= y1_pred-y_test
diffence=pd.DataFrame(diffence)
diffence


# In[288]:


df2


# In[289]:


DeptID=3
Position=6
SpecialProjectsCount=5

lin_re.predict([[DeptID, Position, SpecialProjectsCount]])


# In[290]:


x.columns


# In[291]:


df['DeptID']


# In[292]:


def predict_salary(DeptID,SpecialProjectsCount):
    loc_index= np.where(x.columns==DeptID)[0][0]
    x=np.zeros(len(x.columns))
    x[0]=SpecialProjectsCount
    
    if loc_index >=0:
        x[loc_index] = 1
        
    return grid_predictions.predict([x])[0]  


# In[293]:


predict_salary(3, 5)


# # New one

# In[294]:


df4=df1[['Department','Position','RaceDesc','SpecialProjectsCount','Age','PerfScoreID','Experience','SalaryGroup','Salary','AgeGroup']]


# In[ ]:





# In[ ]:





# In[295]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df4['Position']= le.fit_transform(df4['Position'])
# df2['AgeGroup']= le.fit_transform(df2['AgeGroup'])
# df2['SalaryGroup']= le.fit_transform(df2['SalaryGroup'])
df4['Department']= le.fit_transform(df4['Position'])
df4['RaceDesc']= le.fit_transform(df4['RaceDesc'])


# In[296]:


df4


# In[297]:



y=df4['Salary']

X=df4.drop(['Salary','AgeGroup','SalaryGroup'],axis=1)


# In[298]:


from sklearn.preprocessing import MinMaxScaler

min_=MinMaxScaler()
X=min_.fit_transform(X)


# In[299]:


df4.Department.unique()


# In[300]:


from sklearn.decomposition import PCA
# pca = PCA(n_components=5)
# X=pca.fit_transform(x)


# In[301]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)


# In[302]:


from sklearn.linear_model import LinearRegression
lin_re = LinearRegression()
lin_re.fit(X_train,y_train)
y1_pred = lin_re.predict(X_test)


# In[303]:


from sklearn.metrics import mean_squared_error, r2_score
print('r^2 value of the linear model:',r2_score(y_test,y1_pred))


# In[304]:


diffence= y1_pred-y_test
diffence=pd.DataFrame(diffence)
diffence


# In[305]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[306]:


from sklearn.model_selection import GridSearchCV


# In[307]:


from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()
regr_trans = TransformedTargetRegressor(regressor=gbr, transformer=MinMaxScaler())
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)

print('r^2 value of the Gredient booster model:',r2_score(y_test,y_pred))


# In[308]:


best_score = 0.0
best_params = {'max_depth': None, 'max_features': 'auto','n_estimators': 10}
for max_depth in [None, 2,3,5]:
    for max_features in ['auto','sqrt', 'log2']:
        for n_estimators in [10,100,200]:
            score = cross_val_score(GradientBoostingRegressor(n_estimators=n_estimators,
                                                          max_features=max_features,
                                                          max_depth=max_depth,
                                                          random_state=43
                                                          ),
                                    X_train,
                                    y_train,
                                    cv=5,
                                    n_jobs=-1).mean()
            if score > best_score:
                best_score= score
                best_params['max_depth'],best_params['max_features'], best_params['n_estimators'] = max_depth, max_features, n_estimators

            print('max_depth : %s, max_features : %s, n_estimators : %s , Average R^2 Score : %.4f'%(str(max_depth), max_features, str(n_estimators), score))

print('\nBest Score : %.4f, Best Params : %s'%(best_score, str(best_params)))


# In[309]:


from sklearn.ensemble import RandomForestRegressor


# In[310]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X_train,y_train):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        },

        'Gredient_booster':{
            'model':GradientBoostingRegressor(random_state=43),
            'params':{ 
                      'max_features': ['auto','sqrt','log2'],
                      'n_estimators': [1,5,10,50,100,200]
                
            }
        },
        'Random_forest':{
            'model':RandomForestRegressor(random_state=43),
            'params':{'n_estimators': [1,5,10,50,100,200],
#                       'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)],
                      'max_features': ['auto', 'sqrt'],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
#                        'min_samples_split': [2, 5, 10],
#                        'min_samples_leaf': [1, 2, 4],
#                        'bootstrap': [True, False]
                     }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        # gs.fit(X,y)
        gs.fit(X_train,y_train)
        y_pred=gs.predict(X_test)
        r2=r2_score(y_test,y_pred)

        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_,
            'r2_score': r2
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[311]:


y


# In[312]:


best_score = 0.0
best_params = {'max_depth': None, 'max_features': 'auto','n_estimators': 10}
for max_depth in [None, 2,3,5]:
    for max_features in ['auto','sqrt', 'log2']:
        for n_estimators in [10,100,200]:
            score = cross_val_score(GradientBoostingRegressor(n_estimators=n_estimators,
                                                          max_features=max_features,
                                                          max_depth=max_depth,
                                                          random_state=43
                                                          ),
                                    X_train,
                                    y_train,
                                    cv=5,
                                    n_jobs=-1).mean()
            if score > best_score:
                best_score= score
                best_params['max_depth'],best_params['max_features'], best_params['n_estimators'] = max_depth, max_features, n_estimators

            print('max_depth : %s, max_features : %s, n_estimators : %s , Average R^2 Score : %.4f'%(str(max_depth), max_features, str(n_estimators), score))

print('\nBest Score : %.4f, Best Params : %s'%(best_score, str(best_params)))


# # Last model

# In[313]:


df1


# In[314]:


# bins2= [40000,70000,100000,130000,160000,190000,220000,250000,280000]
# labels2 = ['level1','level2','','130000-160000','160000-190000','190000-220000','220000-250000','250000-2800000']
# df1['SalaryGroup'] = pd.cut(df1['Salary'], bins=bins2, labels=labels2, right=False)
# df1['SalaryGroup']


# In[315]:


x= df5.drop(['Salary'],axis=1)
y =df5['Salary']


# In[326]:


from sklearn.preprocessing import MinMaxScaler

min_=MinMaxScaler()
X=min_.fit_transform(x)


# In[316]:


x.columns


# In[317]:


df5.loc[df5['carrierbreak']==1]


# In[318]:


corr = df5.corr() # We already examined SalePrice correlations
# .drop('Salary', axis=1)
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# In[319]:


for i in range(len(df1['DateofTermination'])):
    if i<2019:
        df5['carrierbreak2']=1
    else:
        df5['carrierbreak2']=0  


# In[320]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
# df5['Position']= le.fit_transform(df2['Position'])
# df5['AgeGroup']= le.fit_transform(df2['AgeGroup'])
# df5['SalaryGroup']= le.fit_transform(df2['SalaryGroup'])
# df5['State']= le.fit_transform(df2['State'])
# df5['CitizenDesc']= le.fit_transform(df2['CitizenDesc'])
# df5['PerformanceScore']= le.fit_transform(df2['PerformanceScore']) 
df5['RaceDesc']= le.fit_transform(df1['RaceDesc'])   
                                     
# df1['ExperiencedGroup']= le.fit_transform(df1['ExperiencedGroup'])


# In[353]:


x= df5[['DeptID','ManagerID','SpecialProjectsCount','Age', 'Experience']]
y =df5['Salary']


# In[351]:


df5.ManagerID.fillna(df5.ManagerID.median(), inplace = True)


# In[352]:


df5.isna().sum()


# In[322]:


df5.columns


# In[323]:


x


# In[354]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y, test_size=0.3, random_state=42)


# In[358]:


from sklearn.linear_model import LinearRegression
lin_re = LinearRegression()
lin_re.fit(X_train,y_train)
y1_pred = lin_re.predict(X_test)


# In[329]:


X


# In[ ]:





# In[ ]:





# In[ ]:





# In[356]:


from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
lin_re = LinearRegression()
regr_trans = TransformedTargetRegressor(regressor=lin_re, transformer=MinMaxScaler())
regr_trans.fit(X_train,Y_train)
Y1_pred = regr_trans.predict(X_test)


# In[359]:


from sklearn.ensemble import RandomForestRegressor

rand_re = RandomForestRegressor()
regr_trans = TransformedTargetRegressor(regressor=rand_re, transformer=MinMaxScaler())
regr_trans.fit(X_train,Y_train)
Y2_pred = regr_trans.predict(X_test)


# In[360]:


from sklearn.ensemble import GradientBoostingRegressor
gbr_re = GradientBoostingRegressor()
regr_trans = TransformedTargetRegressor(regressor=gbr_re, transformer=MinMaxScaler())
regr_trans.fit(X_train,Y_train)
Y3_pred = regr_trans.predict(X_test)


# In[361]:


import xgboost as xg
xgb_re =xg.XGBRegressor()
regr_trans = TransformedTargetRegressor(regressor=xgb_re, transformer=MinMaxScaler())
regr_trans.fit(X_train,Y_train)
Y4_pred = regr_trans.predict(X_test)


# In[362]:


from sklearn.linear_model import Lasso
las = Lasso()
las.fit(X_train,Y_train)
Y5_pred = las.predict(X_test)


# In[363]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 80, 60, 55, 51, 45],  
              'max_depth': [7, 8],
              #'reg_lambda' :[0.26, 0.25, 0.2]
             }
                
grid = GridSearchCV(RandomForestRegressor(), param_grid, refit = True, verbose = 3, n_jobs=-1) #
regr_trans = TransformedTargetRegressor(regressor=grid, transformer=MinMaxScaler(feature_range=(0, 1)))
# fitting the model for grid search 
grid_result=regr_trans.fit(X_train,Y_train)
best_params=grid_result.regressor_.best_params_
print(best_params)


# In[364]:


best_model = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"])
regr_trans = TransformedTargetRegressor(regressor=best_model, transformer=MinMaxScaler(feature_range=(0, 1)))
regr_trans.fit(X_train, Y_train)
Y6_pred = regr_trans.predict(X_test)


# In[365]:


print('r^2 value of the linear model:',r2_score(Y_test,Y1_pred))


# In[369]:


from sklearn.metrics import mean_squared_error, r2_score
print('r^2 value of the linear model:',r2_score(Y_test,Y1_pred))
print('r^2 value of the Random forest model:',r2_score(Y_test,Y2_pred))
print('r^2 value of the Gradient booster model:',r2_score(Y_test,Y3_pred))
# print('r^2 value of the xgboost model:',r2_score(Y_test,Y4_pred))
print('r^2 value of the Random forest after finetunning model:',r2_score(Y_test,Y6_pred))
# print('r^2 value of the Gredient boosting after finetunning model:',r2_score(Y_test,Y7_pred))
# print('r^2 value of the Xg boosting after finetunning model:',r2_score(Y_test,Y8_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


table = pd.pivot_table(df1, values='EmpID', index=['Position','SalaryGroup'],aggfunc=lambda x: len(x.unique()))
table.rename({"EmpID":"Count of Employees"},axis=1,inplace=True)
table


# In[ ]:


plt.hist(df['Salary'], bins=4)


# In[ ]:


from sklearn.compose import TransformedTargetRegressor

