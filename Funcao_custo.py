import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados


perth = "Ficheiros_Normalizados/perth_data_normalizado.csv"
melbourne = "Ficheiros_Normalizados/melb_data_normalizado.csv"

perth = pd.read_csv(perth)
melbourne = pd.read_csv(melbourne)


def regressao_linear(df_treino, df_teste):
    # Separar as características e a variável alvo para treino e teste
    X_train = df_treino.drop(columns=["price"])
    y_train = df_treino["price"]

    X_test = df_teste.drop(columns=["price"])
    y_test = df_teste["price"]

    # Definição do modelo e busca de hiperparâmetros
    model = LinearRegression()
    param_grid = {'fit_intercept': [True, False]}

    # Validação cruzada e busca em grade
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)

    # Visualização dos resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.title("Real Price vs Predicted Price")
    plt.show()

    # show a heatmap with the correlation between the features
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_treino.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_teste.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()



    return best_model


regressao_linear(melbourne, perth)




