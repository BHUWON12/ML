import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load datasets
df=pd.read_csv('dataset.csv')

#filling missing values
df['Data_value']=df['Data_value'].fillna(df['Data_value'].mean())
'''
#drom useless   values
df=df.drop(columns=['Series_title_4','Series_title_5'],errors='ignore')

# Encode all object (categorical) columns correctly
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col])

#feature scaling
scaler=StandardScaler()
numeric_cols=['Period','Data_value','Magnitude']
df[numeric_cols]=scaler.fit_transform(df[numeric_cols])
'''
#select features and target

X = df[['Period', 'Data_value', 'Magnitude']]  # Features

y=df['STATUS']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)

#Train K-NN classifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#predict
y_pred=knn.predict(X_test)
print("\nAccuracy:",accuracy_score(y_test,y_pred))
print("\n Classification Report:",classification_report(y_test,y_pred))
print("\n Confusion matrix:",confusion_matrix(y_test,y_pred))


