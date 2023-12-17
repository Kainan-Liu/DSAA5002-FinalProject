import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional, Literal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc


def xgboost_wrapper(data_dir,
                    random_state: Optional[int] = 42,
                    dropna: Literal[True, False] = False,
                    remove_low_variance: Optional[int] = None):
    '''
    Grid Search with Cross validation find the best parameters
    '''
    if not os.path.exists(data_dir):
        raise FileNotFoundError
    
    params = {
        "learning_rate": [0.1, 0.05, 0.01, 0.2],
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 5, 7], 
    }
    
    train_file = os.path.join(data_dir, "Q5_train.xlsx")
    data = pd.read_excel(train_file)

    if dropna:
        data.dropna(axis=0, inplace=True)
    
    y = data.iloc[:, -1]
    X = data.iloc[:, 1:-1]
    
    drop_features = []
    if remove_low_variance:
        features = X.var(axis="rows").sort_values().index.tolist()
        drop_features = features[:remove_low_variance]
        X.drop(drop_features, axis=1, inplace=True)
        
        
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Create an XGBoost classifier
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=params, cv=10)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # ------------------------------------------------------------------------------------------------------

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    xgb.plot_importance(model)
    plt.show()

    # Make predictions on the validation data
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Get predicted probabilities on the test set
    y_probabilities = model.predict_proba(X_valid)[:, 1]  # Probability of the positive class
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_valid, y_probabilities)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    print("=======================================================")

    #-------------------------------------------------------------------------------------------------------

    test_file = os.path.join(data_dir, "Q5_test.csv")
    test_data = pd.read_csv(test_file)
    if remove_low_variance is not None:
        test_data.drop(drop_features, axis=1, inplace=True)
    id = test_data.iloc[:, 0].tolist()
    y_pred = model.predict(test_data.iloc[:, 1:]).tolist()

    submit = pd.DataFrame({
        "id": id,
        "smoking": y_pred
    })

    submit.to_csv("./Data/Q5/Q5_output.csv", index=False)
    print("Submit successfully!")
    print("=======================================================")

if __name__ == "__main__":
    xgboost_wrapper(data_dir="./Data/Q5/", dropna=True, remove_low_variance=5)