import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from model_training import istreniraj_model
def evaluiraj_model(model_pipeline,x_test,y_test):
    predikcije=model_pipeline.predict(x_test)
    mae=mean_absolute_error(y_test,predikcije)
    mse=mean_squared_error(y_test,predikcije)
    rmse=mse**0.5
    r2=r2_score(y_test,predikcije)
    print("---Rezultati evaluacije modela---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    primjeri=pd.DataFrame({
        'stvarna_cijena':y_test.values[:10],
        'predikcija_cijene':predikcije[:10].round(2)
    })
    primjeri['greska']=(primjeri['stvarna_cijena']-primjeri['predikcija_cijene']).abs().round(2)
    print("\n---Primjeri predikcija---")
    print(primjeri)
    return mae,mse,rmse,r2
if __name__=="__main__":
    model_pipeline,x_test,y_test=istreniraj_model()
    evaluiraj_model(model_pipeline,x_test,y_test)