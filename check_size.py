import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.decomposition import PCA

df = pd.read_csv("train.csv")
one_hot_encoder = OneHotEncoder()

numerical_cols = df.select_dtypes(include=['float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = one_hot_encoder.fit_transform(df[column].astype(str))

# print(df.dtypes)
for column in df.select_dtypes(include=['float64']).columns:
    df[column] = one_hot_encoder.fit_transform(df[column].astype(str))


corr_matrix = df.corr()

print(df.shape)
print(df.describe())
print(corr_matrix["Transported"])

