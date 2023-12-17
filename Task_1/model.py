from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def get_decision_tree_regressor():
    regressor = DecisionTreeRegressor()
    param_grid = {
        'regressor__max_depth': [None, 5, 10, 15],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    return regressor, param_grid

def get_random_forest_regressor():
    regressor = RandomForestRegressor()
    param_grid = {
        'regressor__n_estimators': [5, 10, 20],
        'regressor__max_depth': [None, 5, 10],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [5, 10]
    }
    return regressor, param_grid


