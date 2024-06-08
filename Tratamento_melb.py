import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sweetviz as sv

melbourne = "Ficheiros_Originais/melb_data.csv"
melbourne = pd.read_csv(melbourne)
melbourne['price'] = melbourne['price'] * 0.61


# Removing lines with null values
# def null_lines(perth, arquivo_csv):
#     missing_values = perth.isnull().sum(axis=1)
#     remove_idx = missing_values[missing_values == 7].index  # remove all lines with 7 missing values
#     df = perth.drop(remove_idx)
#     df.to_csv(arquivo_csv, index=False)
#
#
# null_lines(melbourne, 'Ficheiros_sOutliers/melb_file_sOutliers.csv')
#
#
# # Using a median to fill in the missing values of the column 'square_meters'
# median_square_meters = melbourne['square_meters'].median()
# melbourne['square_meters'] = melbourne['square_meters'].replace(0, median_square_meters)
#
#
# def remove_outliers(df, column):
#     Q1 = df[column].quantile(0.05)
#     Q3 = df[column].quantile(0.95)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
#     return df
#
#
# melbourne_sOutliers = melbourne.copy()
# melbourne_sOutliers = remove_outliers(melbourne_sOutliers, 'bedrooms')
# melbourne_sOutliers = remove_outliers(melbourne_sOutliers, 'bathrooms')
# melbourne_sOutliers = remove_outliers(melbourne_sOutliers, 'car_garage')
# melbourne_sOutliers = remove_outliers(melbourne_sOutliers, 'square_meters')
# melbourne_sOutliers = remove_outliers(melbourne_sOutliers, 'price')
# melbourne_sOutliers.to_csv('Ficheiros_sOutliers/melb_file_sOutliers.csv', index=False)
#
#
# # Normalization of data
# def normalizacao(melbourne_sOutliers, arquivo_csv='Ficheiros_Normalizados/melb_data_normalizado.csv'):
#     scaler = MinMaxScaler()
#     melbourne_normalizado = melbourne_sOutliers.copy()
#     # just consider the ["bedrooms", "bathrooms", "car_garage", "square_meters","price"] columns
#     melbourne_normalizado = melbourne_normalizado[["bedrooms", "bathrooms", "car_garage", "square_meters", "price"]]
#     melbourne_normalizado = pd.DataFrame(scaler.fit_transform(melbourne_normalizado), columns=melbourne_normalizado.
#                                          columns)
#     melbourne_normalizado.to_csv(arquivo_csv, index=False)
#
#
# normalizacao(melbourne)
#
#
# def hist_plot(df, column, title):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     fig.suptitle(title)
#     axs[0].hist(df[column], bins=20, color='blue', alpha=0.7)
#     axs[0].set_title('Before')
#     axs[0].set_xlabel(column)
#     axs[0].set_ylabel('Frequency')
#     axs[1].hist(melbourne_sOutliers[column], bins=20, color='red', alpha=0.7)
#     axs[1].set_title('After')
#     axs[1].set_xlabel(column)
#     axs[1].set_ylabel('Frequency')
#     plt.show()
#
#
# hist_plot(melbourne, 'bedrooms', 'Bedrooms')
# hist_plot(melbourne, 'bathrooms', 'Bathrooms')
# hist_plot(melbourne, 'car_garage', 'Car Garage')
# hist_plot(melbourne, 'square_meters', 'Square Meters')
# hist_plot(melbourne, 'price', 'Price')
#
# # Using the Sweetviz library to generate a report
# report = sv.analyze(melbourne_sOutliers)
# report.show_html('Ficheiros_sOutliers/melb_sOutliers_report.html')

