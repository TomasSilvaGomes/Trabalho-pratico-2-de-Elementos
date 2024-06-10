import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
import sweetviz as sv


perth = "Ficheiros_Originais/perth_file.csv"
perth = pd.read_csv(perth)
perth['price'] = perth['price'] * 0.61


# using a median to fill in the missing values of the column 'car_garage'
# median_car_garage = perth['car_garage'].median()
# perth['car_garage'].fillna(median_car_garage, inplace=True)
#
#
# # use the IQR method to remove outliers for the columns 'bedrooms', 'bathrooms', 'car_garage', 'square_meters' and 'price'
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

perth_sOutliers = perth
perth_sOutliers = remove_outliers(perth_sOutliers, 'bedrooms')
perth_sOutliers = remove_outliers(perth_sOutliers, 'bathrooms')
perth_sOutliers = remove_outliers(perth_sOutliers, 'car_garage')
perth_sOutliers = remove_outliers(perth_sOutliers, 'square_meters')
perth_sOutliers = remove_outliers(perth_sOutliers, 'price')
perth_sOutliers.to_csv('Ficheiros_sOutliers/perth_file_sOutliers.csv', index=False)
# #
# #
# # Normalization of data
def normalizacao(arquivo_csv='Ficheiros_Normalizados/perth_data_normalizado.csv'):
    scaler = MinMaxScaler()
    perth_sOutliers = pd.read_csv('Ficheiros_sOutliers/perth_file_sOutliers.csv')
    perth_normalizado = perth_sOutliers
    perth_normalizado = perth_normalizado[["bedrooms", "bathrooms", "car_garage", "square_meters", "price"]]
    perth_normalizado = pd.DataFrame(scaler.fit_transform(perth_normalizado), columns=perth_normalizado.columns)
    perth_normalizado.to_csv(arquivo_csv, index=False)


normalizacao()

# use a hist to show the distribution of the columns 'bedrooms', 'bathrooms', 'car_garage', 'square_meters' and 'price' before and after removing the outliers

def hist_plot(df, column, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    axs[0].hist(df[column], bins=20, color='blue', alpha=0.7)
    axs[0].set_title('Before')
    axs[0].set_xlabel(column)
    axs[0].set_ylabel('Frequency')
    axs[1].hist(perth_sOutliers[column], bins=20, color='red', alpha=0.7)
    axs[1].set_title('After')
    axs[1].set_xlabel(column)
    axs[1].set_ylabel('Frequency')
    plt.show()

hist_plot(perth, 'bedrooms', 'Bedrooms')
hist_plot(perth, 'bathrooms', 'Bathrooms')
hist_plot(perth, 'car_garage', 'Car Garage')
hist_plot(perth, 'square_meters', 'Square Meters')
hist_plot(perth, 'price', 'Price')



perth_sOutliers = pd.read_csv('Ficheiros_sOutliers/perth_file_sOutliers.csv')
# use sweetviz to generate a report for the perth_sOutliers dataset
report = sv.analyze(perth_sOutliers)
report.show_html('Ficheiros_sOutliers/perth_sOutliers_report.html')
