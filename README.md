# Car Price Prediction

Projekat za predvidjanje cijene polovnih automobila na osnovu
njihovih karakteristika (marka, model, godina, kilometraza,
tip goriva, itd.)

## Sadrzaj repozitorijuma
car-price-prediction/
├── data/
│ ├── cars.csv - originalni skup podataka
│ ├── cars_cleaned.csv - podaci nakon ciscenja
│ └── cars_features.csv - podaci sa dodatim karakteristikama
├── notebooks/
│ └── 01_eda.ipynb - exploratorna analiza podataka
├── src/
│ ├── data_cleaning.py - uklanja nevalidne/ekstremne redove
│ ├── feature_engineering.py - dodaje nove karakteristike
│ ├── data_preprocessing.py - priprema podatke za model (imputacija, skaliranje, encoding)
│ ├── model_training.py - trenira finalni model i cuva ga
│ ├── model_evaluation.py - racuna metrike i primjere predikcija
│ └── model_comparison.py - poredi vise modela
├── models/
│ └── car_price_model.joblib - sacuvan finalni model
├── requirements.txt
└── README.md

## Kako pokrenuti projekat

### 1. Instalacija potrebnih biblioteka

```bash
pip install -r requirements.txt
```

### 2. Pokretanje koraka redom

Svi skriptovi se pokrecu iz glavnog foldera (`car-price-prediction`):

```bash
python3 src/data_cleaning.py
python3 src/feature_engineering.py
python3 src/data_preprocessing.py
python3 src/model_training.py
python3 src/model_evaluation.py
python3 src/model_comparison.py
```

- `data_cleaning.py` cita `data/cars.csv`, uklanja nevalidne redove, cuva `data/cars_cleaned.csv`
- `feature_engineering.py` dodaje nove karakteristike, cuva `data/cars_features.csv`
- `data_preprocessing.py` prikazuje kako se podaci pripremaju za model (imputacija, skaliranje, one-hot encoding)
- `model_training.py` trenira finalni model (Random Forest) i cuva ga u `models/car_price_model.joblib`
- `model_evaluation.py` prikazuje metrike (MAE, MSE, RMSE, R2) i primjere predikcija
- `model_comparison.py` poredi 4 razlicita modela

## EDA i analiza

Detaljna exploratorna analiza podataka (raspodela cijena, uocavanje ekstremnih vrijednosti, analiza kategorijskih kolona) nalazi se u `notebooks/01_eda.ipynb`.

Kljucni problemi uoceni u podacima:
- automobili sa nerealnom cijenom (<=100$ ili >=100000$)
- automobili sa godinom proizvodnje prije 1950
- automobili sa kilometrazom preko milion km (ocigledne greske u unosu)
- automobili sa zapreminom motora preko 6000 cm3
- nedostajuce vrijednosti u volume(cm3), drive_unit i segment

## Nove karakteristike (feature engineering)

- `car_age` - starost automobila
- `mileage_per_year` - prosjecna kilometraza po godini starosti
- `engine_volume_liters` - zapremina motora u litrima
- `is_newer_car` - da li je automobil noviji od 10 godina
- `is_high_mileage` - da li automobil ima preko 300000 km

## Poredjenje modela

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 2462.40 | 4219.01 | 0.6870 |
| Decision Tree | 1476.66 | 3135.42 | 0.8271 |
| **Random Forest** | **1123.44** | **2262.25** | **0.9100** |
| Gradient Boosting | 1439.49 | 2573.85 | 0.8835 |

## Izabrani model

Izabran je **Random Forest Regressor**, jer ima najnizi MAE (u prosjeku
gresi za oko 1123$) i najvisi R2 (objasnjava 91% varijacije cijene).
Dodatna prednost u odnosu na Linear Regression je da ne predvidja
negativne cijene, sto se desavalo kod linearnog modela.

MAE od 1123$ znaci da model u prosjeku odstupa za oko 1123$ od stvarne
cijene automobila - sto je razumna greska s obzirom na to da cijena
polovnih automobila zavisi i od faktora koje ovaj skup podataka ne
sadrzi (stanje enterijera, istorija servisiranja, pregovaranje sa
prodavcem, itd.)