"""Fraud Detection - handles a highly imbalanced dataset (1% fraud) with class weights and threshold tuning."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split


def main():
    X, y = make_classification(n_samples=20_000, n_features=12, n_informative=6,
                               weights=[0.99, 0.01], flip_y=0.002, random_state=42)
    print(f"Transactions: {len(y):,}   fraud: {y.sum()} ({y.mean():.2%})\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    naive_acc = 1 - y_test.mean()
    print(f"A model that always says 'not fraud' is {naive_acc:.2%} accurate, but catches 0 fraud.")
    print("So we look at precision and recall instead of accuracy.\n")

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Average precision (PR-AUC): {average_precision_score(y_test, proba):.3f}\n")

    print("Default threshold 0.5:")
    print(classification_report(y_test, (proba >= 0.5).astype(int), target_names=["legit", "fraud"], digits=3))

    # Choose the threshold that maximises F1 for the fraud class
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = np.argmax(f1[:-1])
    print(f"Best threshold {thresholds[best]:.2f}: precision {precision[best]:.2f}, recall {recall[best]:.2f}")
    print(classification_report(y_test, (proba >= thresholds[best]).astype(int),
                                target_names=["legit", "fraud"], digits=3))


if __name__ == "__main__":
    main()
