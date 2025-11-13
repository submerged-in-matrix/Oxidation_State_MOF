from env.modules import *
from utils.scaler import X_train_clean, X_test_clean, X_holdout_clean, y_train, y_test, y_holdout

# Inserted best parameters from your optimization results:
final_params_gb = {
    'learning_rate': 0.07015351058361705,
    'max_depth': 12,
    'max_features': 0.5114056705372313,
    'min_samples_leaf': 6,
    'min_samples_split': 12,
    'n_estimators': 129,
    'subsample': 0.9116608003829677
}
final_params_et = {
    'criterion': 'entropy',
    'max_features': 0.6917810676482335,
    'min_samples_leaf': 1,
    'min_samples_split': 4,
    'n_estimators': 59
}
final_params_knn = {
    'leaf_size': 48,
    'metric': 'manhattan',
    'n_neighbors': 4,
    'p': 2,
    'weights': 'distance'
}
final_params_sgd = {
    'alpha': 4.0212149576865536e-05,
    'eta0': 0.07050171813288647,
    'l1_ratio': 0.24236385892113504,
    'learning_rate': 'invscaling',
    'max_iter': 38000,
    'penalty': 'elasticnet',
    'tol': 4.582974403364687e-05
}

# Build base estimators
clf_gb = GradientBoostingClassifier(random_state=0, **final_params_gb)
clf_et = ExtraTreesClassifier(random_state=0, **final_params_et)
clf_knn = KNeighborsClassifier(**final_params_knn)
clf_sgd = SGDClassifier(loss='log', random_state=42, **final_params_sgd)

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('gb', clf_gb),
        ('et', clf_et),
        ('knn', clf_knn),
        ('sgd', clf_sgd)
    ],
    voting='soft'
)

# Train and evaluate
ensemble.fit(X_train_clean, y_train)

# Test set evaluation
y_pred = ensemble.predict(X_test_clean)
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Holdout set evaluation
y_pred_holdout = ensemble.predict(X_holdout_clean)
print("\nHoldout accuracy:", accuracy_score(y_holdout, y_pred_holdout))
print(classification_report(y_holdout, y_pred_holdout))