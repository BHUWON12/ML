import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/eklavya/Desktop/ml LAB/dataset.csv')  # Make sure path is right

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Period', y='Data_value', data=df)
plt.title('Period vs Data Value')
plt.xlabel('Period')
plt.ylabel('Data Value')
plt.show()
