import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import napravi_preprocesing_pipeline

def uporedi_modele():
    podaci = pd.read_csv('data/cars_features.csv')

    preprocessor, numericke_kolone, kategorijske_kolone = napravi_preprocesing_pipeline()

    X = podaci[numericke_kolone + kategorijske_kolone]
    y = podaci['priceUSD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    modeli = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=30, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    rezultati = []

    for ime_modela in modeli:
        print(f"Treniram: {ime_modela}...")

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', modeli[ime_modela])
        ])

        model_pipeline.fit(X_train, y_train)
        predikcije = model_pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predikcije)
        mse = mean_squared_error(y_test, predikcije)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predikcije)

        rezultati.append({
            'Model': ime_modela,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4)
        })

        print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

    tabela_rezultata = pd.DataFrame(rezultati)

    print("\n=== Poredjenje svih modela ===")
    print(tabela_rezultata.to_string(index=False))

    # model sa najnizim MAE se smatra najboljim (MAE je najlakse tumaciti)
    najbolji = tabela_rezultata.loc[tabela_rezultata['MAE'].idxmin(), 'Model']
    print(f"\nNajbolji model po MAE kriterijumu: {najbolji}")

    return tabela_rezultata


if __name__ == "__main__":
    uporedi_modele()