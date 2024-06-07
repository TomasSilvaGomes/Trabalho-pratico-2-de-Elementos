import pandas as pd
from sklearn.preprocessing import MinMaxScaler

melbourne = "Ficheiros/melb_data.csv"
melbourne = pd.read_csv(melbourne)

#Remoção de linhas com valores nulos
def remover_linhas(melbourne, arquivo_csv):
    valores_falta = melbourne.isnull().sum(axis=1)
    indices_remover = valores_falta[valores_falta == 7].index
    df = melbourne.drop(indices_remover)
    df.to_csv(arquivo_csv, index=False)

colunas_em_falta = ["bedrooms", "bathrooms", "car_garage", "sqft_living","price"]
for coluna in colunas_em_falta:
    melbourne[coluna] = melbourne[coluna].replace(0, None)

#Cálculo da mediana para preencher valores nulos do sqft_living

median_sqft_living = melbourne['sqft_living'].median()
melbourne['sqft_living'] = melbourne['sqft_living'].replace(0, median_sqft_living)

#Remoção de outliers usando a técnica IQR 5/95
Q1 = melbourne['sqft_living'].quantile(0.05)
Q3 = melbourne['sqft_living'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = melbourne[
    (melbourne['sqft_living'] >= Q1 - 1.5 * IQR) & (melbourne['sqft_living'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/melb_data.csv', index=False)

Q1 = melbourne['bedrooms'].quantile(0.05)
Q3 = melbourne['bedrooms'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = melbourne[(melbourne['bedrooms'] >= Q1 - 1.5 * IQR) & (melbourne['bedrooms'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/melb_data.csv', index=False)

Q1 = melbourne['bathrooms'].quantile(0.05)
Q3 = melbourne['bathrooms'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = melbourne[(melbourne['bathrooms'] >= Q1 - 1.5 * IQR) & (melbourne['bathrooms'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/melb_data.csv', index=False)

Q1 = melbourne['car_garage'].quantile(0.05)
Q3 = melbourne['car_garage'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = melbourne[(melbourne['car_garage'] >= Q1 - 1.5 * IQR) & (melbourne['car_garage'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/melb_data.csv', index=False)

Q1 = melbourne['price'].quantile(0.05)
Q3 = melbourne['price'].quantile(0.95)
IQR = Q3 - Q1
without_outliers = melbourne[(melbourne['price'] >= Q1 - 1.5 * IQR) & (melbourne['price'] <= Q3 + 1.5 * IQR)]
without_outliers.to_csv('Ficheiros/melb_data.csv', index=False)


remover_linhas(melbourne, 'Ficheiros/melb_data.csv')

#Normalização dos dados
def normalizacao(melbourne, arquivo_csv= 'Ficheiros_Normalizados/melb_data_normalizado.csv'):
    scaler = MinMaxScaler()
    melbourne_normalizado = melbourne.copy()
    # just consider the ["bedrooms", "bathrooms", "car_garage", "sqft_living","price"] columns
    melbourne_normalizado = melbourne_normalizado[["bedrooms", "bathrooms", "car_garage", "sqft_living","price"]]
    melbourne_normalizado = pd.DataFrame(scaler.fit_transform(melbourne_normalizado), columns=melbourne_normalizado.columns)
    melbourne_normalizado.to_csv(arquivo_csv, index=False)

normalizacao(melbourne)




