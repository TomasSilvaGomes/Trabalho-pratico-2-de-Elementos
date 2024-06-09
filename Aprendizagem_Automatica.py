import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Carregar os dados
perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"
new_york = "Ficheiros_Normalizados/NY_data_normalizado.csv"

perth = pd.read_csv(perth)
melbourne = pd.read_csv(melbourne)
new_york = pd.read_csv(new_york)
perth_melbourne = pd.concat([perth, melbourne]).drop(columns=['car_garage'])
# train the model using a concat of perth and melbourn and test it with new york
X = perth_melbourne
y = X.pop('price')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X_train, y_train)
print("R2 score:", model.score(X_test, y_test))

# test the model with new york data
X = new_york
y = X.pop('price')
print("R2 score:", model.score(X, y))

# plot the este with the new york data

plt.scatter(y, model.predict(X))
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("True Prices vs Predicted Prices")
plt.show()






