import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
# Assuming you have already loaded and preprocessed your dataset and stored it in a variable df
df = pd.read_excel("potato2.xlsx")

df = df.iloc[:,:-1]

col = ['state', 'district','season','start', 'end', 'area', 'production']
df = df[col]
df = df.dropna()

numeric_features = [3, 4, 5]
categorical_features = [0, 1, 2]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor)])

X_train_scaled = clf.fit_transform(df)
dd= pd.DataFrame(X_train_scaled)




# Calculate the correlation matrix
correlation_matrix = dd.corr()


# Create a heatmap
plt.figure(figsize=(10, 8), dpi=200)  # Set figure size and DPI
sns.set(font='Times New Roman')  # Set font family
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 16, 'fontweight': 'bold'})  # Set title font properties
plt.show()
