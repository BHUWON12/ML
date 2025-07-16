import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn. metrics import r2_score
import matplotlib.pyplot as plt


df=pd.read_csv('dataset.csv')
#filling missing values
df['Data_value']=df['Data_value'].fillna(df['Data_value'].mean())


#selecting features
X=df[['Period','Magnitude']]
y=df['Data_value']

#split data

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=0)


#traing the model
model=LinearRegression()
model.fit(X_train,y_train)


#predection

y_pred=model.predict(X_test)

print("R2 score:",r2_score(y_test,y_pred))


#visualiation

plt.figure(figsize=(6,4))

plt.scatter(y_test,y_pred,color='blue',alpha=0.5)
plt.xlabel("Actual data value")
plt.ylabel("Predected data values")
plt.title("Actual vs Predection -Linear Regression")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
plt.tight_layout()
plt.show()




