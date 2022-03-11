import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
regressor=xgboost.XGBRegressor()

def car_predictions(Present_price, kms, Fuel, Seller, Transmission, Owner, Age):
    car_data=pd.read_csv('car data.csv')
    lb=LabelEncoder()
    car_data['Fuel_Type']=lb.fit_transform(car_data['Fuel_Type'])
    car_data['Seller_Type']=lb.fit_transform(car_data['Seller_Type'])
    car_data['Transmission']=lb.fit_transform(car_data['Transmission'])
    car_data['Vehicle_Age']=2020- car_data['Year']
    car_data.drop(['Year'],axis=1,inplace=True)
    
    x=car_data.drop(columns=['Selling_Price','Car_Name','company'],axis=1)
    y=car_data['Selling_Price']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)

    # reinitializing the regressor object with the best probable estimators
    regressor=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                importance_type=None, interaction_constraints='',
                learning_rate=0.15, max_delta_step=0, max_depth=15,
                min_child_weight=3, missing=np.nan, monotone_constraints='()',
                n_estimators=1500, n_jobs=12, num_parallel_tree=1, random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                tree_method='exact', validate_parameters=1, verbosity=None)
    # fitting the xgbRegressor on our training data
    regressor.fit(X_train,y_train)
    x_car_test=np.array([Present_price, kms, Fuel, Seller, Transmission, Owner,Age])
    x_car_test=x_car_test.reshape((1,7))
    predictions=regressor.predict(x_car_test)
    # fetching the predictions on our test data
    return predictions