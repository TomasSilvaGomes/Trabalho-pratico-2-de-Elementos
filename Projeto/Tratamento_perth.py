import pandas as pd
from sklearn.preprocessing import MinMaxScaler

perth = "Ficheiros/perth_file.csv"
perth = pd.read_csv(perth)


#Remoção de linhas com valores nulos
def remover_linhas(melbourne, arquivo_csv):
    valores_falta = melbourne.isnull().sum(axis=1)
    indices_remover = valores_falta[valores_falta == 7].index
    df = melbourne.drop(indices_remover)
    df.to_csv(arquivo_csv, index=False)


colunas_em_falta = ["bedrooms", "bathrooms", "car_garage", "sqft_living", "price"]
for coluna in colunas_em_falta:
    perth[coluna] = perth[coluna].replace(0, None)

#Cálculo da mediana para preencher valores nulos do sqft_living

median_sqft_living = perth['sqft_living'].median()
perth['sqft_living'] = perth['sqft_living'].replace(0, median_sqft_living)

#Remoção de outliers usando a técnica IQR 5/95
Q1 = perth['sqft_living'].quantile(0.05)
Q3 = perth['sqft_living'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = perth[
    (perth['sqft_living'] >= Q1 - 1.5 * IQR) & (perth['sqft_living'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/perth_file.csv', index=False)

Q1 = perth['bedrooms'].quantile(0.05)
Q3 = perth['bedrooms'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = perth[(perth['bedrooms'] >= Q1 - 1.5 * IQR) & (perth['bedrooms'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/perth_file.csv', index=False)

Q1 = perth['bathrooms'].quantile(0.05)
Q3 = perth['bathrooms'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = perth[(perth['bathrooms'] >= Q1 - 1.5 * IQR) & (perth['bathrooms'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/perth_file.csv', index=False)

Q1 = perth['car_garage'].quantile(0.05)
Q3 = perth['car_garage'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = perth[(perth['car_garage'] >= Q1 - 1.5 * IQR) & (perth['car_garage'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/perth_file.csv', index=False)

Q1 = perth['price'].quantile(0.05)
Q3 = perth['price'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = perth[(perth['price'] >= Q1 - 1.5 * IQR) & (perth['price'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/perth_file.csv', index=False)

remover_linhas(perth, 'Ficheiros/perth_file.csv')


#Normalização dos dados
def normalizacao(perth, arquivo_csv='Ficheiros_Normalizados/perth_data_normalizado.csv'):
    scaler = MinMaxScaler()
    perth_normalizado = perth.copy()
    # just consider the ["bedrooms", "bathrooms", "car_garage", "sqft_living","price"] columns
    perth_normalizado = perth_normalizado[["bedrooms", "bathrooms", "car_garage", "sqft_living", "price"]]
    perth_normalizado = pd.DataFrame(scaler.fit_transform(perth_normalizado), columns=perth_normalizado.columns)
    perth_normalizado.to_csv(arquivo_csv, index=False)


normalizacao(perth)
