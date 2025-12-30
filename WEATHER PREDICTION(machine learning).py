import pandas as pd
df=pd.read_csv("C:/Users/user/Downloads/archive/Weather Test Data.csv")
df.head(100)
# df.tail()
# df.columns
# df.info()
# df.isnull().sum()
# df.dtypes
# miss_percent=df.isnull().mean()*100
#new_df=df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','row ID'],axis=1)
df['Location'].value_counts()