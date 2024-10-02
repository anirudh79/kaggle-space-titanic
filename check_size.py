import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

one_hot_encoder = OneHotEncoder()

numerical_cols = df_train.select_dtypes(include=['float64']).columns
categorical_cols = df_train.select_dtypes(include=['object']).columns

le = LabelEncoder()

for column in categorical_cols:
    df_train[column] = le.fit_transform(df_train[column])
    df_test[column] = le.fit_transform(df_test[column])

scaler = StandardScaler()

df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = scaler.fit_transform(df_test[numerical_cols])

# print(df.head)

corr_matrix = df_train.corr()

# print(corr_matrix["Transported"])
df_train = df_train.dropna()

X = df_train.drop(columns=["Transported"])
y = df_train["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, Y_pred)
confusion = confusion_matrix(y_test, Y_pred)
report = classification_report(y_test, Y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)



