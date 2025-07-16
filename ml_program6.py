import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


#load dataset

df=pd.read_csv('dataset.csv')




# 🧹 1. Handle Missing Data

df['Data_value']= df['Data_value'].fillna(df['Data_value'].mean())

df=df.drop(columns=['Series_title_4','Series_title_5'],errors='ignore')

print("missing Values",df.isnull().sum())


# 🔤 2. Encode Categorical Columns
label_enc=LabelEncoder()

for col in df.select_dtypes(include='object').columns:

    df[col]=label_enc.fit_transform(df[col])

print("encoded DataFrame:",df.head())




#feature Scaling


scaler=StandardScaler()
numeric_cols=['Period','Data_value','Magnitude']



df[numeric_cols]=scaler.fit_transform(df[numeric_cols])

print("scaed Numerical Features:",df[numeric_cols].head())


