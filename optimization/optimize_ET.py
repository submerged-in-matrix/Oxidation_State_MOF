from env.modules import *
from utils.scaler import X_train_clean, y_train, X_test_clean, y_test

space_et = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
    'max_features': hp.uniform('max_features', 0.3, 1.0),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 8, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 12, 1),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
}

def objective_et(params):
    params = params.copy()
    params['n_estimators'] = int(params['n_estimators'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['min_samples_split'] = int(params['min_samples_split'])
    model = ExtraTreesClassifier(random_state=0, **params)
    score = cross_val_score(model, X_train_clean, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}

trials_et = Trials()
n_total = 500  # 500 in the paper

n_tpe = int(0.8 * n_total)
n_rand = int(0.1 * n_total)
n_anneal = n_total - n_tpe - n_rand

# TPE
fmin(objective_et, space_et, algo=tpe.suggest, max_evals=n_tpe, trials=trials_et, rstate=np.random.default_rng(42))
# Random
fmin(objective_et, space_et, algo=rand.suggest, max_evals=n_tpe + n_rand, trials=trials_et, rstate=np.random.default_rng(43))
# Annealing
fmin(objective_et, space_et, algo=anneal.suggest, max_evals=n_total, trials=trials_et, rstate=np.random.default_rng(44))

best_idx = np.argmin([t['result']['loss'] for t in trials_et.trials])
best_params = trials_et.trials[best_idx]['misc']['vals']
final_params = {k: v[0] if isinstance(v, list) else v for k, v in best_params.items()}
final_params['n_estimators'] = int(final_params['n_estimators'])
final_params['min_samples_leaf'] = int(final_params['min_samples_leaf'])
final_params['min_samples_split'] = int(final_params['min_samples_split'])
final_params['criterion'] = ['gini', 'entropy'][final_params['criterion']]
print("Best Hyperopt params for ExtraTrees:", final_params)

et_best = ExtraTreesClassifier(random_state=0, **final_params)
et_best.fit(X_train_clean, y_train)
y_pred_et = et_best.predict(X_test_clean)
print("Test accuracy:", accuracy_score(y_test, y_pred_et))
print(classification_report(y_test, y_pred_et))