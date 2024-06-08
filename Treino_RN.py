import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"


# Treino usamos perth e teste usamos melbourne para isso podemos dar concat nos dois dataframes e depois dividir em X e y

melbourne = pd.read_csv(melbourne)

# Concatenar os dois dataframes
data = pd.concat([perth, melbourne], axis=0)

# Dividir em X e y
X = data.drop('price', axis=1)
y = data['price']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Prever
y_pred = model.predict(X_test)

# Avaliar
mse = mean_squared_error(y_test, y_pred) * 100   # mse = mean squared error that is mean of the squared of the differences between the predicted and the true values
mape = (abs((y_test - y_pred) / y_test).mean()) * 100 # mape = mean absolute percentage error that is average of absolute percentage differences between predictions and real values
mae = (abs(y_test - y_pred).mean()) * 100  # mae = mean absolute error that is the average difference between predicted values and actual values


print(f'MSE: {mse}')
print(f'MAPE: {mape}')
print(f'MAE: {mae}')
