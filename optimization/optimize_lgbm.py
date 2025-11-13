from env.modules import *
from utils.opt_weight import X_train_final
from utils.scaler import y_train

space_lgbm = {
    'num_leaves': hp.quniform('num_leaves', 10, 150, 1),
    'max_depth': hp.quniform('max_depth', 2, 14, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0)),
    'n_estimators': hp.quniform('n_estimators', 10, 150, 1),
    'min_child_samples': hp.quniform('min_child_samples', 1, 20, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
}

def objective_lgbm(params):
    params = params.copy()
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    params['min_child_samples'] = int(params['min_child_samples'])
    model = LGBMClassifier(random_state=0,
                           n_jobs=-1,
                           verbose=-1,
                           verbosity=-1,
                           force_col_wise=True,
                           **params)
    score = cross_val_score(model, X_train_final, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}

trials_lgbm = Trials()
n_total = 500  

n_tpe = int(0.8 * n_total)
n_rand = int(0.1 * n_total)
n_anneal = n_total - n_tpe - n_rand

# TPE
fmin(objective_lgbm, space_lgbm, algo=tpe.suggest, max_evals=n_tpe, trials=trials_lgbm, rstate=np.random.default_rng(42))
# Random
fmin(objective_lgbm, space_lgbm, algo=rand.suggest, max_evals=n_tpe + n_rand, trials=trials_lgbm, rstate=np.random.default_rng(43))
# Annealing
fmin(objective_lgbm, space_lgbm, algo=anneal.suggest, max_evals=n_total, trials=trials_lgbm, rstate=np.random.default_rng(44))

# Extract best params
best_idx = np.argmin([t['result']['loss'] for t in trials_lgbm.trials])
vals = trials_lgbm.trials[best_idx]['misc']['vals']
final_params_lgbm = {k: v[0] if isinstance(v, list) else v for k, v in vals.items()}
for k in ['num_leaves', 'max_depth', 'n_estimators', 'min_child_samples']:
    final_params_lgbm[k] = int(final_params_lgbm[k])
print("Best Hyperopt params for LGBM:", final_params_lgbm)