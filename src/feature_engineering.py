import pandas as pd
from datetime import datetime

def dodaj_karakteristike(podaci):
    trenutna_godina = datetime.now().year
    podaci['car_age'] = trenutna_godina - podaci['year']
    podaci['mileage_per_year'] = podaci['mileage(kilometers)'] /(podaci['car_age']+1)
    podaci['engine_volume_liters'] = podaci['volume(cm3)'] / 1000
    podaci['is_newer_car']=(podaci['car_age']<=10).astype(int)
    podaci['is_high_mileage']=(podaci['mileage(kilometers)']>300000).astype(int)
    print("Karakteristike su dodate.")
    return podaci

if __name__=="__main__":
    podaci=pd.read_csv("data/cars_cleaned.csv")
    podaci=dodaj_karakteristike(podaci)
    podaci.to_csv("data/cars_features.csv",index=False)
    print("\nDodavanje karakteristika završeno. Podaci sa dodanim karakteristikama su sačuvani u 'data/cars_features.csv'.")
    print("\nPrikaz novih kolona:")
    print(podaci[['year', 'car_age', 'mileage(kilometers)', 'mileage_per_year', 'volume(cm3)', 'engine_volume_liters', 'is_newer_car', 'is_high_mileage']].head())