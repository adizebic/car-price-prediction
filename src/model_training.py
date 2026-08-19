import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from data_preprocessing import napravi_preprocesing_pipeline
def istreniraj_model():
    podaci=pd.read_csv("data/cars_features.csv")
    preprocessor,numericke_kolone,kategorijske_kolone=napravi_preprocesing_pipeline()
    x=podaci[numericke_kolone+kategorijske_kolone]
    y=podaci['priceUSD']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model_pipeline=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=30,max_depth=15, n_jobs=-1))
    ])
    model_pipeline.fit(x_train,y_train)
    print("Model je istreniran.")
    joblib.dump(model_pipeline,"models/car_price_model.joblib",compress=3)
    print("Model je sačuvan u 'models/car_price_model.joblib'.")
    return model_pipeline,x_test,y_test
if __name__=="__main__":
    istreniraj_model()