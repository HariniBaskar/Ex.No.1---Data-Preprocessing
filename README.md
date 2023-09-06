# Ex.No.1 Data-Preprocessing
## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
1. Importing the libraries
2. Importing the dataset
3. Taking care of missing data
4. Encoding categorical data
5. Normalizing the data
6. Splitting the data into test and train

## PROGRAM:
```
Developed by: Harini.B
Register Number: 212221230035
```

```
# Importing Libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset
df=pd.read_csv('Churn_Modelling.csv')
df

#Checking for null values
df.isnull().sum()

#Checking for dulpicated values
df.duplicated()

#Dropping unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df

#Normalising using MinMaxScaler
ms=MinMaxScaler()
df2=pd.DataFrame(ms.fit_transform(df))
df2

#Splitting the dataset - x
X=df2.iloc[:,:-1].values
X

#Splitting the dataset - y
y=df2.iloc[:,-1].values
y

# Training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
```

## OUTPUT:
### Reading dataset:
![img1](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/afb06d33-65e1-44a7-9184-8fee734e94d3)

### Checking for null values:
![img2](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/1b6e388e-e63f-4acd-9706-930669a81af7)

### Duplicated values:
![img3](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/f4d4e344-57b8-4e5f-b9a6-68a4d0fb94d4)

### Dropping off irrelevant values:
![img4](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/40236e2f-a19f-4fac-8293-01932c996b96)

### Normalization using MinMaxScaler:
![img5](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/3943e13b-d7dc-4d16-865a-439935c8b903)

### Array of X:
![img6](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/3f69863a-c4c5-4d0f-8488-ad507f05051d)

### Array of Y:
![img7](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/8d299634-5966-4a83-8125-1caa1f93e9a5)

### Training the dataset:
![img8](https://github.com/HariniBaskar/Ex.No.1---Data-Preprocessing/assets/93427253/377f7638-613a-4d54-8dee-bd12077ee5cf)

## RESULT
Thus the given data is been processed successfully.
