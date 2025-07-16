#introduce scikit-learn as a machine learing library
from sklearn import datasets
iris=datasets.load_iris()
print("Features Names:",iris.feature_names)
print("first Row of data:",iris.data[0])
print("Target (label) of first Row",iris.target[0])
