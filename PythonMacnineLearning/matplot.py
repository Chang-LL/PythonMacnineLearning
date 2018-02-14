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

