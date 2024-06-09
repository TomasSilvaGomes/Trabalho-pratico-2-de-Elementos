import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"

perth = pd.read_csv(perth)
melbourne = pd.read_csv(melbourne)

data = pd.concat([perth, melbourne], axis=0)

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% treino, 20% teste

# Polynomial regression example
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)  # Predicting the prices using the model

# Plotting actual vs predicted prices for polynomial regression
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_poly, color='blue', s=60, alpha=0.6, edgecolor='w', label='Predicted vs Actual')
sns.regplot(x=y_test, y=y_pred_poly, scatter=False, color='red', line_kws={"linewidth": 2, "linestyle": "--"},
            label='Regression Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Polynomial Regression)')
plt.legend()
plt.show()

# Plotting residuals for polynomial regression
residuals = y_test - y_pred_poly
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution for Polynomial Regression')
plt.show()

# Calculate the mean squared error and mean absolute error for polynomial regression
mse_poly = mean_squared_error(y_test, y_pred_poly) * 100
mae_poly = mean_absolute_error(y_test, y_pred_poly) * 100
print(f'Mean Squared Error (Polynomial Regression): {mse_poly}')
print(f'Mean Absolute Error (Polynomial Regression): {mae_poly}')
