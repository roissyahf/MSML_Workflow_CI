import pandas as pd
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
                """Train the model using the training data."""
                self.model.fit(X_train, y_train)
                return self.model
        
        def evaluate(self, X_test, y_test):
                """Evaluate the model"""
                return self.model.score(X_test, y_test)

        def predict(self, X_test):
                """Make predictions using the trained model."""
                return self.model.predict(X_test)

        def save_model(self, path):
                joblib.dump(self.model, path)

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

    # train the model
    model.train(X_train, y_train)

    # evaluate the model
    score = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {score}")

    # save the model to local file
    model.save_model("LGBM_v3.joblib")
