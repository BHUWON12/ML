import pandas as pd

df_csv = pd.read_csv('dataset.csv')  
print("CSV Head:\n", df_csv.head())
print("CSV Summary Info:")
print(df_csv.info())                 

print("\nFull CSV Data:")
print(df_csv)


df_excel = pd.read_excel('dataset2.xlsx')
print("\nExcel Head:\n", df_excel.head())   
print("Excel Summary Info:")
print(df_excel.info())                      

print("\nFull Excel Data:")
print(df_excel)
