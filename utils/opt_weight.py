from env.modules import *
from utils.scaler import *
from src.tuned_ensemble import *
from utils.feature_ranking import shap_feature_order, feature_names

final_features_ensemble = shap_feature_order[:47]
# Prepare train and test sets
X_train_final = pd.DataFrame(X_train_clean, columns=feature_names)[final_features_ensemble]
X_test_final = pd.DataFrame(X_test_clean, columns=feature_names)[final_features_ensemble]
X_holdout_final = pd.DataFrame(X_holdout_clean, columns=feature_names)[final_features_ensemble]
X_train_opt = X_train_final
X_test_opt = X_test_final

# Fix tuned base models (already fit, just need to be passed in fresh for VotingClassifier)
clf_gb = GradientBoostingClassifier(random_state=0, **final_params_gb)
clf_et = ExtraTreesClassifier(random_state=0, **final_params_et)
clf_knn = KNeighborsClassifier(**final_params_knn)
clf_sgd = SGDClassifier(loss='log', random_state=42, **final_params_sgd)

# Define search space
space = {
    'w_gb': hp.uniform('w_gb', 1.0, 10.0),
    'w_et': hp.uniform('w_et', 1.0, 10.0),
    'w_knn': hp.uniform('w_knn', 0.1, 0.7),
    'w_sgd': hp.uniform('w_sgd', 0.1, 0.7),
    'voting_type': hp.choice('voting_type', ['soft', 'hard'])
}

def objective(params):
    weights = [params['w_gb'], params['w_et'], params['w_knn'], params['w_sgd']]
    ensemble = VotingClassifier(
        estimators=[
            ('gb', clf_gb),
            ('et', clf_et),
            ('knn', clf_knn),
            ('sgd', clf_sgd)
        ],
        voting=params['voting_type'],
        weights=weights if params['voting_type'] == 'soft' else None # weights only for 'soft'
    )
    ensemble.fit(X_train_opt, y_train)
    y_pred = ensemble.predict(X_test_opt)
    acc = accuracy_score(y_test, y_pred)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50, 
    trials=trials,
    rstate=np.random.default_rng(42)
)

# Map voting_type back to string
voting_types = ['soft', 'hard']
best['voting_type'] = voting_types[best['voting_type']]

print("\nBest VotingClassifier params from hyperopt:")
print(best)

# Re-train best model for reporting
best_weights = [best['w_gb'], best['w_et'], best['w_knn'], best['w_sgd']]
best_ensemble = VotingClassifier(
    estimators=[
        ('gb', clf_gb),
        ('et', clf_et),
        ('knn', clf_knn),
        ('sgd', clf_sgd)
    ],
    voting=best['voting_type'],
    weights=best_weights if best['voting_type']=='soft' else None
)
best_ensemble.fit(X_train_opt, y_train)

# Test
y_pred = best_ensemble.predict(X_test_opt)
print("\nTest accuracy (optimized VotingClassifier):", accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Holdout 
y_pred_holdout = best_ensemble.predict(X_holdout_final)
print("\nHoldout accuracy:", accuracy_score(y_holdout, y_pred_holdout))
print(classification_report(y_holdout, y_pred_holdout))