from env.modules import *
from utils.opt_weight import X_train_final
from utils.scaler import y_train

space_rf = {
    'n_estimators': hp.quniform('n_estimators', 10, 150, 1),
    'max_depth': hp.quniform('max_depth', 2, 14, 1),
    'max_features': hp.uniform('max_features', 0.3, 1.0),
    'min_samples_split': hp.quniform('min_samples_split', 2, 12, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 8, 1),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}

def objective_rf(params):
    params = params.copy()
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    # Robust mapping
    if isinstance(params['criterion'], int):
        params['criterion'] = ['gini', 'entropy'][params['criterion']]
    # If it's already a string, leave it
    model = RandomForestClassifier(random_state=0, n_jobs=-1, **params)
    score = cross_val_score(model, X_train_final, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}


trials_rf = Trials()
n_total = 500

n_tpe = int(0.8 * n_total)
n_rand = int(0.1 * n_total)
n_anneal = n_total - n_tpe - n_rand

# TPE
fmin(objective_rf, space_rf, algo=tpe.suggest, max_evals=n_tpe, trials=trials_rf, rstate=np.random.default_rng(42))
# Random
fmin(objective_rf, space_rf, algo=rand.suggest, max_evals=n_tpe + n_rand, trials=trials_rf, rstate=np.random.default_rng(43))
# Annealing
fmin(objective_rf, space_rf, algo=anneal.suggest, max_evals=n_total, trials=trials_rf, rstate=np.random.default_rng(44))

# Extract best params
best_idx = np.argmin([t['result']['loss'] for t in trials_rf.trials])
vals = trials_rf.trials[best_idx]['misc']['vals']
final_params_rf = {k: v[0] if isinstance(v, list) else v for k, v in vals.items()}
for k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
    final_params_rf[k] = int(final_params_rf[k])
final_params_rf['criterion'] = ['gini', 'entropy'][final_params_rf['criterion']]
print("Best Hyperopt params for RF:", final_params_rf)