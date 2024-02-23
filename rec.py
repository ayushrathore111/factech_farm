
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load the data with actual and predicted values
data = pd.read_excel("topic39.xlsx")
data = data.dropna()
X= data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Decision Tree
from sklearn.tree             import DecisionTreeRegressor
DT = DecisionTreeRegressor(random_state=0)
DT.fit(X_train, Y_train)
DTR_Pred = DT.predict(X_test)
from sklearn.ensemble import ExtraTreesRegressor
extra_trees_model = ExtraTreesRegressor()
extra_trees_model.fit(X_train, Y_train)
ETR_Pred = extra_trees_model.predict(X_test)
from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor()
bagging_model.fit(X_train, Y_train)
BAGGING_Pred = bagging_model.predict(X_test)
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
from sklearn.ensemble import AdaBoostRegressor

adaboost_model = AdaBoostRegressor()
adaboost_model.fit(X_train, Y_train)
ar_Pred = adaboost_model.predict(X_test)
# Make predictions using the trained model
y_pred = model.predict(X_test)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,Y_train)
knn_Pred = knn.predict(X_test)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
LR_Pred = lr.predict(X_test)
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
gbr = GradientBoostingRegressor()
rf = RandomForestRegressor()
gbr.fit(X_train,Y_train)
rf.fit(X_train,Y_train)
gbr_predict = gbr.predict(X_test)
rf_predict = rf.predict(X_test)
# Extract actual values
actual_values = data.iloc[:,-1]

# Extract model names
# KNN	ETR	BR	AR	XGB	LR	RF	DTR	SVR	GBR	MLP
# "KNN","ETR","BR","AR", "XGB","LR","RF","DTR","GBR"
model_names = ["KNN","BR","AR","XGB","LR","RF","GBR"]  # Modify with your model names
# models= []
# Initialize dictionaries to store accuracy and deviation
accuracy = {}
deviation = {}
lrxg= (LR_Pred+y_pred)/2
lrrf= (LR_Pred+rf_predict)/2
lrgbr= (LR_Pred+gbr_predict)/2
brgbr= (BAGGING_Pred+gbr_predict)/2
bretr = (BAGGING_Pred+ETR_Pred)/2
lrbr= (LR_Pred+BAGGING_Pred)/2
# Calculate absolute errors
absolute_errors = {}
# absolute_errors["LR&XG"]= np.abs(lrxg-Y_test)
# absolute_errors["LR&RF"]= np.abs(lrrf-Y_test)
# absolute_errors["LR&GBR"]= np.abs(lrgbr-Y_test)
# absolute_errors["BR&GBR"]= np.abs(brgbr-Y_test)
# absolute_errors["BR&ETR"]= np.abs(bretr-Y_test)
# absolute_errors["LR&BR"]= np.abs(lrbr -Y_test)
absolute_errors["KNN"] = np.abs(knn_Pred - Y_test)
# absolute_errors["ETR"] = np.abs(ETR_Pred - Y_test)
absolute_errors["BR"] = np.abs(BAGGING_Pred - Y_test)
absolute_errors["AR"]= np.abs(ar_Pred-Y_test)
absolute_errors["XGB"] = np.abs(y_pred - Y_test)
absolute_errors["LR"] = np.abs(LR_Pred - Y_test)
absolute_errors["RF"] = np.abs(rf_predict - Y_test)
# absolute_errors["DTR"] = np.abs(DTR_Pred - Y_test)
absolute_errors["GBR"] = np.abs(gbr_predict - Y_test)

# Calculate accuracy and deviation
for model, errors in absolute_errors.items():
    accuracy[model] = 1 - (errors / actual_values)
    deviation[model] = 1 - (errors / errors.mean())

# Calculate REC curve points
rec_curve_points = {}
for model, accuracy_values in accuracy.items():
    rec_curve_points[model] = []
    for threshold in np.linspace(0, 1, 101):
        rec_curve_points[model].append(np.mean(accuracy_values >= threshold))

# Plot REC curve with accuracy and deviation for each model
plt.figure(figsize=())
# lrxg_rec = rec_curve_points["LR&XG"]
# lrrf_rec = rec_curve_points["LR&RF"]
# lrgbr_rec = rec_curve_points["LR&GBR"]
# brgbr_rec = rec_curve_points["BR&GBR"]
# bretr_rec = rec_curve_points["BR&ETR"]
# lrbr_rec = rec_curve_points["LR&BR"]
knn_rec= rec_curve_points["KNN"]
# etr_rec= rec_curve_points["ETR"]
br_rec= rec_curve_points["BR"]
ar_rec= rec_curve_points["AR"]
xgb_rec= rec_curve_points["XGB"]
lr_rec= rec_curve_points["LR"]
rf_rec= rec_curve_points["RF"]
# dtr_rec= rec_curve_points["DTR"]
gbr_rec= rec_curve_points["GBR"]
datafile= {
    # "LR&XG":lrxg_rec,
    # "LR&RF":lrrf_rec,
    # "LR&GBR":lrgbr_rec,
    # "BR&GBR":brgbr_rec,
    # "BR&ETR":bretr_rec,
    # "LR&BR":lrbr_rec,
    "KNN":knn_rec,
    # "ETR":etr_rec,
    "BR":br_rec,
    "AR":ar_rec,
    "XGB":xgb_rec,
    "LR":lr_rec,
    "RF":rf_rec,
    # "DTR":dtr_rec,
    "GBR":gbr_rec
}
df= pd.DataFrame(datafile)
print(df)
# df.to_excel("rec_values.xlsx")
plt.figure(figsize=(8,6),dpi=300)
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12 
for model in model_names:
    plt.plot(np.linspace(0, 1, 101), rec_curve_points[model], marker='o', label=f"{model} REC Curve")
plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Comparison of REC Curves with Accuracy and Deviation")
plt.legend()
plt.grid(True)
plt.show()



