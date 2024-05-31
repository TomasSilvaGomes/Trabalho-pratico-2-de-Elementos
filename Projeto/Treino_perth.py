# train the data from Ficherios_Normalizados/perth_normalized.csv

# Path: Projeto/Treino_perth.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Carregar dados
# Assumindo que os dados estão em um arquivo CSV
perth = pd.read_csv('Ficheiros_Normalizados/perth_normalized.csv')

# Dividir os dados em variáveis independentes e dependentes
X = perth.drop('price', axis=1)
y = perth['price']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Prever os preços
y_pred = model.predict(X_test)

# Calcular o erro
error = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', error)
