from preprocesser import preprocess_car_data
from model import get_decision_tree_regressor, get_random_forest_regressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np



if __name__ == "__main__":
    DATASET_PATH = "data/car_data.csv"
    X, y, preprocessor = preprocess_car_data(DATASET_PATH)
    
    regressor, param_grid = get_decision_tree_regressor()
      
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    y_scaled = StandardScaler().fit_transform(np.asarray(y).reshape(-1,1)).reshape(-1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )
    
    
    print("Performing Grid Search...")
    grid_search = GridSearchCV(
        pipe,
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    predictions = best_model.predict(X_test)
    
    r2_acc = r2_score(predictions, y_test)
    mse = mean_squared_error(predictions, y_test)
    
    print("\n-- MODEL ACCURACY SCORES --")
    print("MSE: ", mse)
    print("R2 Score: ", r2_acc)