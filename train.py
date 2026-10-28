import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

SEED = 42

if __name__ == "__main__":
    df = pd.read_csv("finn-annonser-data.csv")

    df = df.fillna(0)

    one_hot = pd.get_dummies(df["type"], prefix="type", drop_first=True)
    df = pd.concat([df, one_hot], axis=1)

    one_hot = pd.get_dummies(df["girkasse"], prefix="girkasse", drop_first=True)
    df = pd.concat([df, one_hot], axis=1)

    one_hot = pd.get_dummies(df["selger"], prefix="selger", drop_first=True)
    df = pd.concat([df, one_hot], axis=1)

    df.drop(["type", "girkasse", "selger"], axis=1, inplace=True)

    print(df.head())

    # Rens datasettet
    numeric_cols = df.select_dtypes(include=["number"])
    iso = IsolationForest(random_state=SEED)
    iso_pred = iso.fit_predict(numeric_cols)

    df_cleaned = df[iso_pred == 1]

    # Splitt
    X = df_cleaned.drop("pris", axis=1)
    y = df_cleaned["pris"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)

    print("antall for trening:", len(X_train), "\nantall for testing:", len(X_test))

    # Target encode
    t_encoder = TargetEncoder(smooth="auto", target_type="continuous")

    X_train[["merke_encoded", "modell_encoded"]] = t_encoder.fit_transform(
        X_train[["merke", "modell"]], y_train
    )
    X_test[["merke_encoded", "modell_encoded"]] = t_encoder.transform(
        X_test[["merke", "modell"]]
    )

    X_train = X_train.drop(columns=["merke", "modell"])
    X_test = X_test.drop(columns=["merke", "modell"])

    # Tren
    rf_reg = GradientBoostingRegressor(random_state=SEED)
    rf_reg.fit(X_train, y_train)

    y_pred = rf_reg.predict(X_test)

    print("Mean absolute error:", mean_absolute_error(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))
    print("Root mean squared error:", root_mean_squared_error(y_test, y_pred))
    print("R2 score:", r2_score(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.regplot(
        x=y_test,
        y=y_pred,
        scatter_kws={"color": "blue", "s": 10},
        line_kws={"color": "red", "linestyle": "--"},
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Regression Plot")
    plt.show()

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
