import os
import pandas as pd
import requests
PATH=r'F:\\'
r=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
r
with open(PATH+'iris.data','w') as f:
    f.write(r.text)

os.chdir(PATH)
df=pd.read_csv(PATH+'iris.data',names=['sepal length',
               'sepal width','petal width','petal length','class'])
df.head()
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
#%matplotlib inline
import numpy as np

fig,ax=plt.subplots(figsize=(6,4))
ax.hist(df['petal width'],color='black')
ax.set_ylabel('Count',fontsize=12)
ax.set_xlabel('Width',fontsize=12)
plt.title('Iris Petal Width',fontsize=14,y=1.01)
plt.show()

fig,ax=plt.subplots(2,2,figsize=(6,4))
ax[0][0].hist(df['petal width'],color='black')
ax[0][0].set_ylabel('Count',fontsize=12)
ax[0][0].set_xlabel('Width',fontsize=12)
ax[0][0].set_title('Iris Petel Width',fontsize=14,y=1.01)

ax[0][1].hist(df['petal width'],color='black')
ax[0][1].set_ylabel('Count',fontsize=12)
ax[0][1].set_xlabel('Width',fontsize=12)
ax[0][1].set_title('Iris Petel Width',fontsize=14,y=1.01)

ax[1][0].hist(df['sepal width'],color='black')
ax[1][0].set_ylabel('Count',fontsize=12)
ax[1][0].set_xlabel('Width',fontsize=12)
ax[1][0].set_title('Iris Sepal Width',fontsize=14,y=1.01)

ax[1][1].hist(df['sepal width'],color='black')
ax[1][1].set_ylabel('Count',fontsize=12)
ax[1][1].set_xlabel('Width',fontsize=12)
ax[1][1].set_title('Iris Sepal Width',fontsize=14,y=1.01)

plt.tight_layout();

plt.show()

fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(df['petal width'],df['petal length'],color='green')
ax.set_xlabel('Petal Width')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Scatterplot')
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
ax.plot(df['petal length'],color='blue')
ax.set_xlabel('Specimen Number')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Scatterplot')
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
bar_width=.8
labels=[x for x in df.columns 
        if 'length' in x or 'width' in x]
ver_y=[df[df['class']=='Iris-versicolor'][x].mean()
       for x in labels]
vir_y=[df[df['class']=='Iris-virginica'][x].mean()
       for x in labels]
set_y=[df[df['class']=='Iris-setosa'][x].mean()
       for x in labels]
x=np.arange(len(labels))
ax.bar(x,vir_y,bar_width,bottom=set_y,color='darkgrey')
ax.bar(x,set_y,bar_width,bottom=ver_y,color='white')
ax.bar(x,ver_y,bar_width,color='black')
ax.set_xticks(x+(bar_width/2))
ax.set_xticklabels(labels,rotation=-70,fontsize=12)
ax.set_title('Mean Feature Measurement By Class',y=1.01)
ax.legend(['Virginica','Setosa','Versicolor'])
plt.show()

import seaborn as sns
sns.pairplot(df,hue="class")
plt.show()
fig,ax=plt.subplots(2,2,figsize=(7,7))
sns.set(style='white',palette='muted')
sns.violinplot(x=df['class'],y=df['sepal length'],ax=ax[0,0])
sns.violinplot(x=df['class'],y=df['sepal width'],ax=ax[0,1])
sns.violinplot(x=df['class'],y=df['petal length'],ax=ax[1,0])
sns.violinplot(x=df['class'],y=df['petal width'],ax=ax[1,1])
fig.suptitle('Violin Plots',fontsize=16,y=1.03)
for i in ax.flat:
    plt.setp(i.get_xticklabels(),rotation=-90)
fig.tight_layout()
plt.show()

df['class']=df['class'].map({'Iris-setosa':'SET','Iris-virginica':                             'VIR','Irix-versicolor':'VER'})
df['wide petal']=df['petal width'].apply(lambda v:1 if v>=1.3 else 0)
df['petal area']=df.apply(lambda r:r['petal length']*r['petal width'],axis=1)
df.applymap(lambda v:np.log(v) if isinstance(v,float) else v)
df.groupby('class').mean()
df.groupby('class').mean()
df.groupby('class')['petal width'].agg(
    {'delta':lambda x:x.max()-x.min(),'max':np.max,'min':np.min})

fig,ax=plt.subplots(figsize=(7,7))
ax.scatter(df['sepal width'][:50],df['sepal length'][:50])
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs.  Sepal Length',fontsize=14,y=1.02)
plt.show()

import statsmodels.api as sm
y=df['sepal length'][:50]
x=df['sepal width'][:50]

X=sm.add_constant(x)
results=sm.OLS(y,X).fit()
print(results.summary())

ax.plot(x, results.fittedvalues, label='regression line')
ax.scatter(x, y, label='data point', color='r')
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs. Sepal Length', fontsize=14,y=1.02)
ax.legend(loc=2)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
clf=RandomForestClassifier(max_depth=5,n_estimators=10)
x=df.ix[:,:4]
y=df.ix[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
rf=pd.DataFrame(list(zip(y_pred,y_test)),columns=['predicted',
                                                  'actual'])
rf['correct']=rf.apply(lambda r:1 if r['predicted']==r['actual'] else 0,axis=1)
rf['correct'].sum()/rf['correct'].count()

f_importances=clf.feature_importances_
f_names=df.columns[:4]
f_std=np.std([tree.feature_importances_ for tree in
              clf.estimators_],axis=0)
zz=zip(f_importances,f_names,f_std)
zzs=sorted(zz,key=lambda x:x[0],reverse=True)
imps=[x[0]for x in zzs]
labels=[x[1] for x in zzs]
errs=[x[2]for x in zzs]
plt.bar(range(len(f_importances)),imps,color='r',yerr=errs
        ,align='center')
plt.xticks(range(len(f_importances)),labels)
plt.show()

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
clf=OneVsRestClassifier(SVC(kernel='linear'))
x=df.ix[:,:4]
y=np.array(df.ix[:,4]).astype(str)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
rf=pd.DataFrame(list(zip(y_pred,y_test)),
                columns=['predicted','actual'])
rf['correct']=rf.apply(lambda r:1 if r['predicted']==r['actual'] else 0,axis=1)
