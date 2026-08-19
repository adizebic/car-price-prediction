import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def napravi_preprocesing_pipeline():
    numericke_kolone=[
        'year','mileage(kilometers)','volume(cm3)',
        'car_age','mileage_per_year','engine_volume_liters',
        'is_newer_car','is_high_mileage'
    ]
    kategorijske_kolone=[
        'make','condition','fuel_type','color',
        'transmission','drive_unit','segment'
    ]
    
    numericki_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    kategorijski_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numericko', numericki_pipeline, numericke_kolone),
            ('kategorijsko', kategorijski_pipeline, kategorijske_kolone)
        ]
    )

    return preprocessor,numericke_kolone,kategorijske_kolone
if __name__=="__main__":
    podaci=pd.read_csv("data/cars_features.csv")
    preprocessor,numericke_kolone,kategorijske_kolone=napravi_preprocesing_pipeline()
    x=podaci[numericke_kolone+kategorijske_kolone]
    y=podaci['priceUSD']
    x_transformisan=preprocessor.fit_transform(x)
    print("Oblik podataka prije preprocesiranja:", x.shape)
    print("Oblik podataka nakon preprocesiranja:", x_transformisan.shape)
    print("\nBroj kolona je veci poslije preprocesiranja zbog one-hot enkodiranja kategorijskih kolona.")