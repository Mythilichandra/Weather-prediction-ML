import pandas as pd
df=pd.read_csv("C:/Users/Praveen kumar/Downloads/archive/Weather Test Data.csv")
df.head()
df.tail()
df.dtypes
df.isnull().sum()
df.isnull().mean()
df ['row ID'].value_counts()
df.columns
new_df=df.drop(['row ID','Evaporation','Sunshine','Cloud9am', 'Cloud3pm'],axis=1)
new_df['Location'].value_counts()

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
new_df['Location']=label.fit_transform(new_df['Location'])
new_df.columns
new_df.dtypes

new_df['MinTemp']=new_df['MinTemp'].fillna(new_df ['MinTemp'].median())
new_df.isnull().sum()

new_df['MaxTemp']=new_df['MaxTemp'].fillna(new_df ['MaxTemp'].median())

Rainfall_mode=new_df['Rainfall'].mode()
new_df['Rainfall']=new_df['Rainfall'].fillna(0)

new_df['WindGustDir']=new_df['WindGustDir'].fillna('missing')
new_df['WindGustDir']=label.fit_transform(new_df['WindGustDir'])

new_df['WindGustSpeed']=new_df['WindGustSpeed'].fillna(new_df ['WindGustSpeed'].median())

new_df['WindDir9am']=new_df['WindDir9am'].fillna('missing')
new_df['WindDir9am']=label.fit_transform(new_df['WindDir9am'])

new_df['WindDir3pm']=new_df['WindDir3pm'].fillna('missing')
new_df['WindDir3pm']=label.fit_transform(new_df['WindDir3pm'])

new_df['WindSpeed9am']=new_df['WindSpeed9am'].fillna(new_df ['WindSpeed9am'].median())

new_df['WindSpeed3pm']=new_df['WindSpeed3pm'].fillna(new_df ['WindSpeed3pm'].median())

new_df['Humidity9am'].value_counts()

new_df['Humidity9am']=new_df['Humidity9am'].fillna(new_df ['Humidity9am'].median())

new_df['Humidity3pm']=new_df['Humidity3pm'].fillna(new_df ['Humidity3pm'].median())

new_df['Pressure9am']=new_df['Pressure9am'].fillna(new_df ['Pressure9am'].median())

new_df['Pressure3pm']=new_df['Pressure3pm'].fillna(new_df ['Pressure3pm'].median())

new_df['Temp9am']=new_df['Temp9am'].fillna(new_df ['Temp9am'].median())

new_df['Temp3pm']=new_df['Temp3pm'].fillna(new_df ['Temp3pm'].median())

new_df=new_df[new_df['RainToday'].notnull()]
new_df['RainToday']=label.fit_transform(new_df['RainToday'])

import matplotlib.pyplot as plt
import seaborn as sns

new_df.hist(bins=30, figsize=(15,20), edgecolor='black')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
sns.boxplot(x=new_df['RainToday'], y=new_df['Location'])
plt.show()

new_df_columns=new_df.columns
new_df_columns=new_df_columns[:-1]
for i in new_df_columns:
    plt.figure(figsize=(8,8))
    sns.boxplot(x=new_df['RainToday'], y=new_df[i])
    plt.show()

new_corr=new_df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(new_corr,annot=True,fmt=".2f", cmap="coolwarm", square=True)
plt.title("HEATMAP")
plt.show()

final_df=new_df.drop(['Location','Temp3pm','Temp9am','Pressure9am','Pressure3pm','MaxTemp'],axis=1)

x=final_df.drop(['RainToday'],axis=1)
y=final_df['RainToday']

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print("ACCURACY SCORE",accuracy_score(y_pred,y_test))
print("PRECISION SCORE",precision_score(y_pred,y_test))
print("RECALL SCORE",recall_score(y_pred,y_test))
print("F1 SCORE",f1_score(y_pred,y_test))