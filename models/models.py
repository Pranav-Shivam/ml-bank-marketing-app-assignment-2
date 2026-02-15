try:
    from helper import ClassifierHelper
    from utils.evaluation_metrics import EvaluationMetrics
except ImportError:
    from .helper import ClassifierHelper
    from .utils.evaluation_metrics import EvaluationMetrics

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


class ClassifierModels:
    def __init__(self):
        self.helper = ClassifierHelper()
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=200,
                n_jobs=-1,
            ),
        }
        self.evaluator = EvaluationMetrics()

    def get_models(self):
        return self.models

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

    def set_up_for_model_training(
        self,
        file_path=None,
        target_column="deposit",
        test_size=0.2,
        random_state=42,
        verbose=False,
    ):
        self.helper.load_data(file_path)
        if verbose:
            print("Data loaded successfully.")
            print(f"Dataset shape: {self.helper.data.shape}")
        self.helper.data_cleaning()
        if verbose:
            print("Data cleaning completed.")

        X, y = self.helper.split_data(target_column=target_column)
        if verbose:
            print(f"Feature shape (X): {X.shape}")
            print(f"Target shape (y): {y.shape}")

        categorical_cols, numerical_cols = self.helper.get_categorical_and_numerical_columns()
        categorical_cols = [col for col in categorical_cols if col != target_column]
        numerical_cols = [col for col in numerical_cols if col != target_column]
        if verbose:
            print(f"Categorical columns: {len(categorical_cols)}")
            print(f"Numerical columns: {len(numerical_cols)}")

        preprocessor_sparse, preprocessor_dense = self.helper.preprocessor_sparse_dense(
            categorical_cols,
            numerical_cols,
        )

        X_train, X_test, y_train, y_test = self.helper.split_train_test(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        if verbose:
            print(f"Train/Test split done. X_train: {X_train.shape}, X_test: {X_test.shape}")

        model_pipelines = {}
        for model_name, model in self.models.items():
            preprocessor = preprocessor_dense if model_name == "Naive Bayes" else preprocessor_sparse
            model_pipelines[model_name] = Pipeline(
                [("preprocessor", preprocessor), ("model", model)]
            )
        if verbose:
            print(f"Prepared pipelines for {len(model_pipelines)} models.")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "model_pipelines": model_pipelines,
            "categorical_columns": categorical_cols,
            "numerical_columns": numerical_cols,
        }

    def train_model(self, model_pipeline, X_train, y_train):
        model_pipeline.fit(X_train, y_train)
        return model_pipeline

    def predict_model(self, model_pipeline, X_test):
        y_pred = model_pipeline.predict(X_test)

        y_score = None
        if hasattr(model_pipeline, "predict_proba"):
            y_score = model_pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(model_pipeline, "decision_function"):
            y_score = model_pipeline.decision_function(X_test)

        return y_pred, y_score

    def evaluate_model(self, model_name, y_test, y_pred, y_score=None):
        return self.evaluator.calculate_metrics(model_name, y_test, y_pred, y_score)

    def train_predict_evaluate(self, model_name, model_pipeline, X_train, y_train, X_test, y_test):
        trained_pipeline = self.train_model(model_pipeline, X_train, y_train)
        y_pred, y_score = self.predict_model(trained_pipeline, X_test)
        return self.evaluate_model(model_name, y_test, y_pred, y_score)

    def train_predict_evaluate_all_models(
        self,
        file_path=None,
        target_column="deposit",
        test_size=0.2,
        random_state=42,
        verbose=False,
    ):
        setup_info = self.set_up_for_model_training(
            file_path=file_path,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            verbose=verbose,
        )

        X_train = setup_info["X_train"]
        y_train = setup_info["y_train"]
        X_test = setup_info["X_test"]
        y_test = setup_info["y_test"]

        all_results = []
        for model_name, model_pipeline in setup_info["model_pipelines"].items():
            if verbose:
                print(f"Training and evaluating: {model_name}")
            result = self.train_predict_evaluate(
                model_name,
                model_pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            all_results.append(result)

        return all_results


if __name__ == "__main__":
    classifier_models = ClassifierModels()
    all_results = classifier_models.train_predict_evaluate_all_models(verbose=True)

    for result in all_results:
        print(result)
