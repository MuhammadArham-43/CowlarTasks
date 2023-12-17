
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_car_data(
    data_path: str
) ->  (pd.DataFrame, pd.DataFrame, Pipeline):
    """
    Handles complete preprocessing of cars_data

    Args:
        data_path (str): Path to cars_data.csv

    Returns:
        X (pd.DataFrame): Dataframe attributes
        y (pd.DataFrame): Dataframe labels
        Preprocessor (Pipeline): Pipeline object for preprocessing data
    """
    assert os.path.exists(data_path), "Dataset not found"    
    
    df = pd.read_csv(data_path)
    
    # Remove all Non-numeric price values
    df["Price"] = pd.to_numeric(df['Price'], errors="coerce")
    df = df.dropna(subset=['Price'])
    
    y = df['Price']
    X = df.drop("Price", axis=1)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categoric_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    categoric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categoric_transformer, categoric_features)
        ]
    )

    return (X, y, preprocessor)