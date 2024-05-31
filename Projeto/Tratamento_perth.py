import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregar dados
# Assumindo que os dados estão em um arquivo CSV
perth = pd.read_csv('Ficheiros/perth_file.csv')
# delete nulls values and outliters from the dataset
perth = perth.dropna()
perth['price'] = perth['price'] * 0.61
# remove variables with str values , null values and outliers
perth = perth.drop(
    ['address', 'suburb', 'latitude', 'longitude', 'floor_area', 'cbd_dist', 'nearest_stn', 'nearest_stn_dist',
     'nearest_sch',
     'nearest_sch_dist', 'nearest_sch_rank', 'post_code', 'date_sold', 'yearbuilt'], axis=1)

# normalize the data
scaler = StandardScaler()
scaler.fit(perth)
perth_normalized = scaler.transform(perth)
perth_normalized = pd.DataFrame(perth_normalized, columns=perth.columns)
# create a new csv file with the normalized data
perth_normalized.to_csv('Ficheiros_Normalizados/perth_normalized.csv', index=False)
