import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar
from sklearn.impute import SimpleImputer


if __name__ == "__main__":
    df = pd.read_csv("home_advantage_matches.csv")

    X = df.drop('home_goal_diff', axis=1)
    y = df['home_goal_diff']

    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("antall for trening:", len(X_train), "\nantall for testing:", len(X_test))

    num_cols = ['elo_home', 'elo_away', 'elo_diff', 'crowd_size', 'travel_distance_km']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.trasform(X_test)
