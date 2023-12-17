# Cars Data Regression Model

Install dependencies from the requirements.txt in the root directoy.

```
pip install -r requirements.txt
```

Run the model using

```
python main.py
```

This creates a Preprocessor and Regression model that fits and predicts on the data.

We perform Grid Search for Hyperparameter tuning and finding the best model results. **Can take upto 5 mintues for tuning.**

The _model.py_ scripts provides two functions that return a regression model and corresponding parameter search grid.

## Dataset Preprocessing

Firstly, we remove non-numeric values from the Price column.

We then split data into attributes and labels.

### Categorical Preprocessing

We replace NaN values with most_frequent value and then one-hot encode the data.

### Numeric Preprocessing

We replace NaN values with mean of the data and then scale values using normalization.

## Model Selection

We use Decision Tree Regressor as it can handle NaN and missing values. For this reason, we do not have to remove NaN values from attributes. Alternatively, we use RandomForestRegressor as an ensemble method over DecisionTree that can further improve accuracy.

## Results

We use R2-Score and Mean Squared Error for analyzing accuracy of our trained models. We acheive excellent results with a high R2 Score and low MSE.

> MSE: 0.07491187886579026
> R2 Score: 0.9267928412349552
