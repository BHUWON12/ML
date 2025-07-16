#  9. Write a program to Implement K-Means clustering and Visualize clusters.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#load dataset
df=pd.read_csv('dataset.csv')



#filling the missing values
df['Data_value']=df['Data_value'].fillna(df['Data_value'].mean())


#select features

X=df[['Period','Data_value']]


#scale features
scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)


#apply KMeans

kmeans=KMeans(n_clusters=3, random_state=42)

df['Cluster']=kmeans.fit_predict(X_scaled)


#visualization
plt.figure(figsize=(7,9))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=df['Cluster'],cmap='viridis',alpha=0.6)
plt.xlabel('Period (scaled)')
plt.ylabel('Data_value (scaled')
plt.title('K-Means Clustering')
plt.grid(True)
plt.show()
