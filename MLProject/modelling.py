import mlflow
import pandas as pd
import random
from lightgbm import LGBMClassifier
from joblib import dump
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# set experiment name
mlflow.set_experiment("Eksperimen Loan Approval Model")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # read train and test set
    train_file_path = os.path.join(os.path.dirname(__file__), "train_data.csv")
    test_file_path = os.path.join(os.path.dirname(__file__), "test_data.csv")

    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)

    # split features and target variable
    X_train = df_train.drop(columns=['loan_status'])
    y_train = df_train['loan_status']
    X_test = df_test.drop(columns=['loan_status'])
    y_test = df_test['loan_status']

    # take input example
    input_example = X_train[20:25]

    # log datasets
    dataset_version = "v1.0"

    # log parameters
    n_estimators=90
    learning_rate=0.03
    num_leaves=32
    random_state=random.randint(0, 1000)

    # start MLflow run
    #with mlflow.start_run():
    # create and train the model
    model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state)
    
    model.fit(X_train, y_train)

    # save to local file
    dump(model, "LGBM_v3.joblib")

    # evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # log dataset
    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("dataset_path", train_file_path)

    # log model parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_leaves", num_leaves)
    mlflow.log_param("random_state", random_state)

    # log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("auc", auc)

    # log confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    random_int = np.random.randint(0, 100)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/test_confusion_matrix.png")
    plt.close()

    # log model
    mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example)

    # log artifacts
    mlflow.log_artifact(train_file_path, artifact_path="datasets")
    mlflow.log_artifact("LGBM_v3.joblib", artifact_path="model_artifacts")
    mlflow.log_artifact(f"plots/test_confusion_matrix.png")