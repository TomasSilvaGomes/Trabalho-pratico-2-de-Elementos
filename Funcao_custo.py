import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
# Carregar os dados


perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"

perth = pd.read_csv(perth)
melbourne = pd.read_csv(melbourne)

data = melbourne

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% treino, 20% teste

# Polynomial regression example
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)  # Predicting the prices using the model
y_pred_test = lr.predict(X_test)

# Plotting actual vs predicted prices for polynomial regression
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_pred_train, color='green', alpha=0.3)
z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

mean_squared_error_train = mean_squared_error(y_train, y_pred_train) * 100
mean_squared_error_test = mean_squared_error(y_test, y_pred_test) * 100
mean_absolute_error_train = mean_absolute_error(y_train, y_pred_train) * 100
mean_absolute_error_test = mean_absolute_error(y_test, y_pred_test) * 100

print('Mean Squared Error (train):', mean_squared_error_train)
print('Mean Squared Error (test):', mean_squared_error_test)
print('Mean Absolute Error (train):', mean_absolute_error_train)
print('Mean Absolute Error (test):', mean_absolute_error_test)
