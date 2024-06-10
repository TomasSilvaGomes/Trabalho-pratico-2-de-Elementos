import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sweetviz as sv
import matplotlib.pyplot as plt

new_york = "Ficheiros_Originais/NY.csv"
new_york = pd.read_csv(new_york)

new_york['price'] = new_york['price'] * 0.92
new_york.rename(columns={"beds": "bedrooms", "bath": "bathrooms", "propertysqft": "square_meters"}, inplace=True)

new_york['square_meters'] = new_york['square_meters'] * 0.092903


def remove_outliers(df, column):
    """This function removes outliers from the data."""
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


def normalizacao(arquivo_csv='Ficheiros_Normalizados/NY_data_normalizado.csv'):
    """This function normalizes the data."""
    scaler = MinMaxScaler()
    new_york_sOutliers = pd.read_csv('Ficheiros_sOutliers/NY_sOutliers.csv')
    new_york_normalizado = new_york_sOutliers
    new_york_normalizado = new_york_normalizado[["bedrooms", "bathrooms", "square_meters", "price"]]
    new_york_normalizado = pd.DataFrame(scaler.fit_transform(new_york_normalizado),
                                        columns=new_york_normalizado.columns)
    new_york_normalizado.to_csv(arquivo_csv, index=False)


# def hist_plot(df, column, title):
#     """This function creates a histogram of the column before and after removing the outliers."""
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     fig.suptitle(title)
#     axs[0].hist(df[column], bins=20, color='blue', alpha=0.7)
#     axs[0].set_title('Before')
#     axs[0].set_xlabel(column)
#     axs[0].set_ylabel('Frequency')
#     axs[1].hist(new_york_sOutliers[column], bins=20, color='red', alpha=0.7)
#     axs[1].set_title('After')
#     axs[1].set_xlabel(column)
#     axs[1].set_ylabel('Frequency')
#     plt.show()


new_york_sOutliers = new_york
new_york_sOutliers = remove_outliers(new_york_sOutliers, 'bedrooms')
new_york_sOutliers = remove_outliers(new_york_sOutliers, 'bathrooms')
new_york_sOutliers = remove_outliers(new_york_sOutliers, 'square_meters')
new_york_sOutliers = remove_outliers(new_york_sOutliers, 'price')
new_york_sOutliers.to_csv('Ficheiros_sOutliers/NY_sOutliers.csv', index=False)

normalizacao()
#
# hist_plot(new_york, 'bedrooms', 'bedrooms')
# hist_plot(new_york, 'bathrooms', 'bathrooms')
# hist_plot(new_york, 'square_meters', 'square_meters')
# hist_plot(new_york, 'price', 'price')

# new_york_sOutliers = pd.read_csv('Ficheiros_sOutliers/NY_sOutliers.csv')
# report = sv.analyze(new_york_sOutliers)
# report.show_html('Ficheiros_sOutliers/NY_sOutliers_report.html')
