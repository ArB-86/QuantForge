import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

y_true = df["TARGET_5D_CLASS"]
y_pred = df["PRED_CLASS"]

print("\n==============================")
print("CLASSIFICATION METRICS")
print("==============================")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_true,
            y_pred
        ),
        4
    )
)

print(
    "Macro Precision:",
    round(
        precision_score(
            y_true,
            y_pred,
            average="macro"
        ),
        4
    )
)

print(
    "Macro Recall:",
    round(
        recall_score(
            y_true,
            y_pred,
            average="macro"
        ),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            y_true,
            y_pred,
            average="macro"
        ),
        4
    )
)

print("\nConfusion Matrix\n")
print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred
    )
)
