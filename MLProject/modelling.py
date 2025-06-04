import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
import joblib
import os
import warnings

class MLModel:
        def __init__(self):
               self.model = LGBMClassifier()

        def load_data(self, train_path, test_path):
               """Load preprocessed data from a CSV file."""
               df_train = pd.read_csv(train_path)
               df_test = pd.read_csv(test_path)
               return df_train, df_test
        
        def split_X_y(self, df_train, df_test):
               """Split the DataFrame into features and target variable."""
               X_train = df_train.drop(columns=['loan_status'])
               y_train = df_train['loan_status']
               X_test = df_test.drop(columns=['loan_status'])
               y_test = df_test['loan_status']
               return X_train, y_train, X_test, y_test

        def train(self, X_train, y_train):
                """Train the model."""
                self.model.fit(X_train, y_train)
                return self.model

        def evaluate(self, X_test, y_test):
                """Evaluate the model."""
                return self.model.score(X_test, y_test)

        def predict(self, X_test):
                return self.model.predict(X_test)

        def save_model(self, path):
                joblib.dump(self.model, path)

        def load_model(self, path):
                self.model = joblib.load(path)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # train and test set paths
    train_path = os.path.join(os.path.dirname(__file__), "train_data.csv")
    test_path = os.path.join(os.path.dirname(__file__), "test_data.csv")

    # model initialization
    model = MLModel()

    # load data
    df_train, df_test = model.load_data(train_path, test_path)

    # split features and target variable
    X_train, y_train, X_test, y_test = model.split_X_y(df_train, df_test)

    # Check for existing MLflow run, or start a new one
    if mlflow.active_run() is None:
        mlflow_run = mlflow.start_run()
    else:
        mlflow_run = mlflow.active_run()

    with mlflow_run:
        # Log parameters
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("n_estimators", model.model.get_params().get("n_estimators", None))
        mlflow.log_param("learning_rate", model.model.get_params().get("learning_rate", None))
        mlflow.log_param("num_leaves", model.model.get_params().get("num_leaves", None))

        # Train and evaluate
        model.train(X_train, y_train)
        score = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {score}")
        mlflow.log_metric("accuracy", score)

        # Log model artifact (in MLflow format)
        mlflow.lightgbm.log_model(model.model, artifact_path="model")

        # Save local model
        model.log_artifact("LGBM_v3.joblib")

    # End run if we started it
    if mlflow_run and mlflow_run.info.run_id != mlflow.active_run().info.run_id:
        mlflow.end_run()



