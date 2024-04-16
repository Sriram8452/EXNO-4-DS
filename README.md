# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/fbb1b7d7-c7db-46cd-ba49-4a20261f49e5)
```
data.isnull().sum()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/a5db7246-53eb-43a3-8cd8-e8327a1ca968)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/b99a2a4e-7e36-4b2c-8735-fe0a77a87f25)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/e1cea5a6-83c5-442c-be6b-3f62a06ec55f)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/d2ce0d6a-ebdd-40f0-87be-068a64367488)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/bd59b814-4eaa-4f6f-9393-8beaa5bcf5cf)
```
data2
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/74751cdd-090e-42f2-a7a0-ecd01b039a11)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/7629d94b-5096-4913-be8b-10059e864049)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/6c699569-3af2-4e5f-9504-98a8abcfa904)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/30e4406b-3f11-4e51-8da7-28b4d1cc027d)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/2bb82a1f-5f8d-40c3-86d0-03d9da50f9fb)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/3eaaacfb-5fcb-48eb-8227-f79987d4d247)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/fb74640e-d4ae-4dfe-9591-6edeb9fa0eb3)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/c7f855e0-2a43-48a8-9152-d59faae74edf)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/b6985774-62cb-4999-8150-105c4ee04b60)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/ac660e31-0c5c-4905-aee5-672af6010f3b)
```
data.shape
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/7ff45d15-3ab3-4d7e-b08a-077bd89f19ea)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/a0f6f2de-c4f8-4595-9564-ab49beaf05c3)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/49d40277-a6da-48d2-af5d-a739414ccdb1)
```
tips.time.unique()
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/6f9c57d1-963b-404b-a871-3b4348d664c5)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/b87a6218-043e-4bb3-9488-d5c0da1b1864)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/DHINESH-SEC/EXNO-4-DS/assets/118708032/5a678abd-2715-44ff-855a-8b1604c9cda4)

# RESULT:
       # INCLUDE YOUR RESULT HERE
