import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
perth = pd.read_csv(perth)
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"

# Treino usamos perth e teste usamos melbourne
X1 = perth.drop('price', axis=1)
y1 = perth['price']
X2 = melbourne.drop('price', axis=1)
y2 = melbourne['price']

# Dividir os dados em treino e teste
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Prever os valores
y_pred = modelo.predict(X_test)

# Calcular o erro

