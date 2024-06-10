import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Carregar os dados


perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"

perth = pd.read_csv(perth)
melbourne = pd.read_csv(melbourne)


def regressao_linear(df_treino, df_teste):
    X_train = df_treino.drop(columns=["price"])
    y_train = df_treino["price"]

    X_test = df_teste.drop(columns=["price"])
    y_test = df_teste["price"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)


    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.title("Real Price vs Predicted Price")
    plt.show()

    return model


regressao_linear(melbourne, perth)

