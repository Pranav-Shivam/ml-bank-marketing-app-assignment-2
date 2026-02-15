from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


class EvaluationMetrics:
    def calculate_metrics(self, model_name, y_test, y_pred, y_score=None):
        auc_score = None
        try:
            if y_score is not None:
                auc_score = roc_auc_score(y_test, y_score)
            else:
                auc_score = roc_auc_score(y_test, y_pred)
        except ValueError:
            auc_score = None

        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": auc_score,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC Score": matthews_corrcoef(y_test, y_pred),
        }
