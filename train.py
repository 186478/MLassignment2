import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    df = pd.read_csv("finn-annonser-data.csv")

    df.info()


    X = df.drop('pris', axis=1)
    y = df['pris']

    one_hot = pd.get_dummies(df['type'], prefix='type', drop_first=True)
    df = pd.concat([df, one_hot], axis=1)

    one_hot = pd.get_dummies(df['girkasse'], prefix='girkasse', drop_first=True)
    df = pd.concat([df, one_hot], axis=1)

    one_hot = pd.get_dummies(df['selger'], prefix='selger', drop_first=True)
    df = pd.concat([df, one_hot], axis=1)

    df.drop(['type', 'girkasse', 'selger'], axis=1, inplace=True)

    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("antall for trening:", len(X_train), "\nantall for testing:", len(X_test))

    num_cols = ['merke', 'aar', 'km', 'type', 'girkasse', 'rekkevidde', 'pris', 'selger']


    # plt.figure(figsize=(14,8))
    # sns.histplot(df.pris, kde=True)
    # plt.xlabel('Selling price')
    # plt.ylabel('Count')
    # plt.title('Distribution of selling prices')
    # plt.show()

    # plt.figure(figsize=(14,8))
    # sns.histplot(df.aar, kde=True)
    # plt.xlabel('Model years')
    # plt.ylabel('Count')
    # plt.title('Distribution of model years')
    # plt.show()

    # plt.figure(figsize=(14,8))
    # sns.histplot(data=df, x='pris', hue='type', kde=True)
    # plt.xlabel('Selling price')
    # plt.ylabel('Count')
    # plt.title('Distribution of selling prices by fuel type')
    # plt.show()

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
