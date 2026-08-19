import pandas as pd
def ocisti_podatke(putanja_do_fajla):
    podaci=pd.read_csv(putanja_do_fajla)
    broj_prije=len(podaci)
    podaci=podaci[(podaci['priceUSD']>100)&(podaci['priceUSD']<100000)]
    podaci=podaci[podaci['year']>=1950]
    podaci=podaci[podaci['mileage(kilometers)']<=1000000]
    podaci=podaci[(podaci['volume(cm3)'].isna())|(podaci['volume(cm3)']<=6000)]
    broj_poslije=len(podaci)
    print(f"Broj redova prije čišćenja: {broj_prije}")
    print(f"Broj redova poslije čišćenja: {broj_poslije}")
    print(f"Broj redova uklonjenih čišćenjem: {broj_prije-broj_poslije}")
    return podaci

if __name__=="__main__":
    ocisceni_podaci=ocisti_podatke("data/cars.csv")
    ocisceni_podaci.to_csv("data/cars_cleaned.csv",index=False)
    print("\nČišćenje podataka završeno. Očišćeni podaci su sačuvani u 'data/cars_cleaned.csv'.")