from env.modules import *
from utils.scaler import *
# Define search space as in the paper (SI Table 3)
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0)),
    'max_depth': hp.quniform('max_depth', 2, 12, 1),
    'n_estimators': hp.quniform('n_estimators', 10, 150, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'max_features': hp.uniform('max_features', 0.3, 1.0),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 8, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 12, 1),
}

def objective(params):
    # Convert float hyperopt params to int where needed
    params = params.copy()
    for k in ['max_depth', 'n_estimators', 'min_samples_leaf', 'min_samples_split']:
        params[k] = int(params[k])
    model = GradientBoostingClassifier(random_state=0, **params)
    score = cross_val_score(model, X_train_clean, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}

trials = Trials()
n_total = 500  # set 500 for paper fidelity, ( was used: 100 for speed in demo )
n_tpe = int(0.8 * n_total)
n_rand = int(0.1 * n_total)
n_anneal = n_total - n_tpe - n_rand

# 1. TPE phase
best_tpe = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=n_tpe,
    trials=trials,
    rstate=np.random.default_rng(42)
)
# 2. Random phase
best_rand = fmin(
    fn=objective,
    space=space,
    algo=rand.suggest,
    max_evals=n_tpe + n_rand,
    trials=trials,
    rstate=np.random.default_rng(43)
)
# 3. Annealing phase
best_anneal = fmin(
    fn=objective,
    space=space,
    algo=anneal.suggest,
    max_evals=n_total,
    trials=trials,
    rstate=np.random.default_rng(44)
)

# Find best parameters from all trials
best_idx = np.argmin([t['result']['loss'] for t in trials.trials])
best_params = trials.trials[best_idx]['misc']['vals']
final_params = {k: v[0] for k, v in best_params.items()}
# int-cast for integer params
for k in ['max_depth', 'n_estimators', 'min_samples_leaf', 'min_samples_split']:
    final_params[k] = int(final_params[k])
print("Best Hyperopt params for GradientBoosting:", final_params)

# Train/evaluate model on full train/test
gb_best = GradientBoostingClassifier(random_state=0, **final_params)
gb_best.fit(X_train_clean, y_train)
y_pred_gb = gb_best.predict(X_test_clean)

print("Test accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))