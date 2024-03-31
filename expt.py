import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('fertility.csv')
df = df.dropna()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


#linear regression
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from xgboost import XGBRegressor


models = {
    'knn':KNeighborsRegressor(),
    'etr':ExtraTreesRegressor(),
    'br':BaggingRegressor(),
    'ar':AdaBoostRegressor(),
    'xgb':XGBRegressor(),
    'lr':LinearRegression(),
    'rf':RandomForestRegressor(),
    'lss':Lasso(),
    'Rid':Ridge(),
    'Dtr':DecisionTreeRegressor(),
    'gbr':GradientBoostingRegressor()
}
for name, md in models.items():
    md.fit(X_train,y_train)
    y_pred = md.predict(X_test)
    
    print(f"{name} : mae : {mean_absolute_error(y_test,y_pred)} score : {r2_score(y_test,y_pred)}")

#just for testing..
xgb = XGBRegressor()
xgb.fit(X_train,y_train)
xg_pred=xgb.predict(X_test)

etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_pred=etr.predict(X_test)

knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)

br = BaggingRegressor()
br.fit(X_train,y_train)
br_pred=br.predict(X_test)

ar = AdaBoostRegressor()
ar.fit(X_train,y_train)
ar_pred=ar.predict(X_test)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_pred=gbr.predict(X_test)

dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_pred=dtr.predict(X_test)


# import joblib
# joblib.dump(knn,'./static/knn.joblib')
joblib.dump(etr,'./static/etr_fertility.joblib')
# joblib.dump(br,'./static/br.joblib')
# joblib.dump(ar,'./static/ar.joblib')
# joblib.dump(xgb,'./static/xgb.joblib')
# joblib.dump(lr,'./static/lr.joblib')
# joblib.dump(rf,'./static/rf.joblib')
# joblib.dump(gbr,'./static/gbr.joblib')
# joblib.dump(dtr,'./static/dtr.joblib')

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# def create_neural_network_model(input_shape):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=input_shape),
#         Dense(32, activation='relu'),
#         Dense(1)  # Output layer (single neuron for regression)
#     ])

#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

#     return model

# # Instantiate the neural network model
# neural_network_model = create_neural_network_model(X_train_dummy.shape[1])

# # Train the model
# neural_network_model.fit(X_train_dummy, y_train, epochs=50, batch_size=32, validation_split=0.2)

# # Predict using the neural network model
# neural_network_pred = neural_network_model.predict(X_test_dummy)
# # Evaluate the neural network model
# neural_network_mae = mean_absolute_error(y_test, neural_network_pred)
# neural_network_score = r2_score(y_test, neural_network_pred)

# print(f"Neural Network: MAE: {neural_network_mae}, R2 Score: {neural_network_score}")
