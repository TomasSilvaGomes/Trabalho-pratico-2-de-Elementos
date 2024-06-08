import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
import sweetviz as sv


perth = "Ficheiros_Originais/perth_file.csv"
perth = pd.read_csv(perth)
perth['price'] = perth['price'] * 0.61

# Remoção de linhas com valores nulos
def null_lines(perth, arquivo_csv):
    missing_values = perth.isnull().sum(axis=1)
    remove_idx = missing_values[missing_values == 7].index  # remove all lines with 7 missing values
    df = perth.drop(remove_idx)
    df.to_csv(arquivo_csv, index=False)


null_lines(perth, 'Ficheiros_sOutliers/perth_file_sOutliers.csv')


# Using a median to fill in the missing values of the column 'square_meters'
median_square_meters = perth['square_meters'].median()
perth['square_meters'] = perth['square_meters'].replace(0, median_square_meters)
# using a median to fill in the missing values of the column 'car_garage'
median_car_garage = perth['car_garage'].median()
perth['car_garage'] = perth['car_garage'].replace(0, median_car_garage)


# use the IQR method to remove outliers for the columns 'bedrooms', 'bathrooms', 'car_garage', 'square_meters' and 'price'
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

perth_sOutliers = perth.copy()
perth_sOutliers = remove_outliers(perth_sOutliers, 'bedrooms')
perth_sOutliers = remove_outliers(perth_sOutliers, 'bathrooms')
perth_sOutliers = remove_outliers(perth_sOutliers, 'car_garage')
perth_sOutliers = remove_outliers(perth_sOutliers, 'square_meters')
perth_sOutliers = remove_outliers(perth_sOutliers, 'price')
# replace the Null values in the column 'car_garage' with the median
perth_sOutliers.to_csv('Ficheiros_sOutliers/perth_file_sOutliers.csv', index=False)
#
#
# Normalization of data
def normalizacao(perth_sOutliers, arquivo_csv='Ficheiros_Normalizados/perth_data_normalizado.csv'):
    scaler = MinMaxScaler()
    perth_normalizado = perth_sOutliers.copy()
    perth_normalizado = perth_normalizado[["bedrooms", "bathrooms", "car_garage", "square_meters", "price"]]
    median_car_garage = perth_sOutliers['car_garage'].median()
    perth_normalizado = pd.DataFrame(scaler.fit_transform(perth_normalizado), columns=perth_normalizado.columns)
    perth_normalizado['car_garage'] = perth_normalizado['car_garage'].replace(0, median_car_garage)
    perth_normalizado.to_csv(arquivo_csv, index=False)


normalizacao(perth)
#
#
# # use a hist to show the distribution of the columns 'bedrooms', 'bathrooms', 'car_garage', 'square_meters' and 'price' before and after removing the outliers
#
# def hist_plot(df, column, title):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     fig.suptitle(title)
#     axs[0].hist(df[column], bins=20, color='blue', alpha=0.7)
#     axs[0].set_title('Before')
#     axs[0].set_xlabel(column)
#     axs[0].set_ylabel('Frequency')
#     axs[1].hist(perth_sOutliers[column], bins=20, color='red', alpha=0.7)
#     axs[1].set_title('After')
#     axs[1].set_xlabel(column)
#     axs[1].set_ylabel('Frequency')
#     plt.show()
#
# hist_plot(perth, 'bedrooms', 'Bedrooms')
# hist_plot(perth, 'bathrooms', 'Bathrooms')
# hist_plot(perth, 'car_garage', 'Car Garage')
# hist_plot(perth, 'square_meters', 'Square Meters')
# hist_plot(perth, 'price', 'Price')



# perth_sOutliers = pd.read_csv('Ficheiros_sOutliers/perth_file_sOutliers.csv')
# # use sweetviz to generate a report for the perth_sOutliers dataset
# report = sv.analyze(perth_sOutliers)
# report.show_html('Ficheiros_sOutliers/perth_sOutliers_report.html')

