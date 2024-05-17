import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Carregar dados
# Assumindo que os dados estão em um arquivo CSV
perth = pd.read_csv('Ficheiros/perth_file.csv')

def rede_neuronal(perth):
    # Prepare data
    X = perth.drop(
        ['address', 'suburb', 'garage', 'floor_area', 'cbd_dist', 'nearest_stn', 'nearest_stn_dist', 'nearest_sch',
         'nearest_sch_dist', 'nearest_sch_rank', 'post_code', 'landsize', 'latitude', 'longitude'], axis=1)
    y = ["price"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(16, activation="relu", input_shape=(X.shape[1],)),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    model.fit(x_train, y_train, epochs=50)
    yhat_test = model.predict(x_test)
    loss, mae = model.evaluate(x_test, y_test)

    print("Mean Absolute Error:", mae)

    return model

rede_neuronal(perth)