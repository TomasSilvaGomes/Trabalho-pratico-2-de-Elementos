import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sweetviz as sv

melbourne = "Ficheiros_Originais/melb_data.csv"
melbourne = pd.read_csv(melbourne)
melbourne['price'] = melbourne['price'] * 0.61


median_car_garage = melbourne['car_garage'].median()
melbourne['car_garage'].fillna(median_car_garage, inplace=True)


# use winsorization to remove outliers
def remove_outliers(df, columns):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    return df


needed_columns = ['bedrooms', 'bathrooms', 'car_garage', 'square_meters', 'price']
melbourne_sOutliers = remove_outliers(melbourne, needed_columns)
melbourne_sOutliers.to_csv('Ficheiros_sOutliers/melb_file_sOutliers.csv', index=False)

# Normalization of data
def normalizacao(arquivo_csv='Ficheiros_Normalizados/melb_data_normalizado.csv'):
    scaler = MinMaxScaler()
    melbourne_sOutliers = pd.read_csv('Ficheiros_sOutliers/melb_file_sOutliers.csv')
    melbourne_normalizado = melbourne_sOutliers
    melbourne_normalizado = melbourne_normalizado[["bedrooms", "bathrooms", "car_garage", "square_meters", "price"]]
    melbourne_normalizado = pd.DataFrame(scaler.fit_transform(melbourne_normalizado), columns=melbourne_normalizado.
                                         columns)
    melbourne_normalizado.to_csv(arquivo_csv, index=False)

normalizacao()

def hist_plot(df, column, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    axs[0].hist(df[column], bins=20, color='blue', alpha=0.7)
    axs[0].set_title('Before')
    axs[0].set_xlabel(column)
    axs[0].set_ylabel('Frequency')
    axs[1].hist(melbourne_sOutliers[column], bins=20, color='red', alpha=0.7)
    axs[1].set_title('After')
    axs[1].set_xlabel(column)
    axs[1].set_ylabel('Frequency')
    plt.show()


hist_plot(melbourne, 'bedrooms', 'Bedrooms')
hist_plot(melbourne, 'bathrooms', 'Bathrooms')
hist_plot(melbourne, 'car_garage', 'Car Garage')
hist_plot(melbourne, 'square_meters', 'Square Meters')
hist_plot(melbourne, 'price', 'Price')

# Using the Sweetviz library to generate a report
report = sv.analyze(melbourne_sOutliers)
report.show_html('Ficheiros_sOutliers/melb_sOutliers_report.html')
