import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv('Ficheiros_Originais/delaney_solubility_with_descriptors.csv')

#remove outliers

def remove_outliers(df, columns):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    return df

needed_columns = [["MolLogP","MolWt","NumRotatableBonds","AromaticProportion","logS"]]
df = remove_outliers(df, needed_columns)

# Normalization of data
def normalizacao(df):
    scaler = MinMaxScaler()
    df_normalizado = df
    df_normalizado = df_normalizado[["MolLogP","MolWt","NumRotatableBonds","AromaticProportion","logS"]]
    df_normalizado = pd.DataFrame(scaler.fit_transform(df_normalizado), columns=df_normalizado.columns)
    return df_normalizado


df = normalizacao(df)

df=df.dropna()


def regressao_linear(df_treino):
    # Separar as características e a variável alvo para treino e teste
    X_train = df_treino.drop(columns=["logS"])
    y_train = df_treino["logS"]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color='red')
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.title("Real Price vs Predicted Price")
    plt.show()

    return best_model

regressao_linear(df)

