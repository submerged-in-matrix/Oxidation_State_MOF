from env.modules import *
from utils.scaler import X_train_clean, X_test_clean, X_holdout_clean, y_train, y_test, y_holdout
from src.tuned_ensemble import *

ensemble = VotingClassifier(
    estimators=[
        ('gb', clf_gb),
        ('et', clf_et),
        ('knn', clf_knn),
        ('sgd', clf_sgd)
    ],
    voting='soft',
    weights=[10.0, 10.0, 0.1, 0.1]  # GB, ET, kNN, SGD
)

# Train and evaluate
ensemble.fit(X_train_clean, y_train)

# Test set
y_pred = ensemble.predict(X_test_clean)
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Holdout set
y_pred_holdout = ensemble.predict(X_holdout_clean)
print("\nHoldout accuracy:", accuracy_score(y_holdout, y_pred_holdout))
print(classification_report(y_holdout, y_pred_holdout))