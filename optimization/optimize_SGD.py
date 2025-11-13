from env.modules import *
from utils.scaler import X_train_clean, y_train, X_test_clean, y_test

space_sgd = {
    'alpha': hp.loguniform('alpha', np.log(1e-6), np.log(1e-2)),
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1.0),
    'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
    'eta0': hp.loguniform('eta0', np.log(1e-4), np.log(1e-1)),
    'max_iter': hp.quniform('max_iter', 1000, 50000, 1000),
    'tol': hp.loguniform('tol', np.log(1e-6), np.log(1e-2))
}

def objective_sgd(params):
    params = params.copy()
    params['max_iter'] = int(params['max_iter'])
    model = SGDClassifier(loss='log', random_state=42, **params)
    score = cross_val_score(model, X_train_clean, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}

trials_sgd = Trials()
n_total = 500
n_tpe = int(0.8 * n_total)
n_rand = int(0.1 * n_total)
n_anneal = n_total - n_tpe - n_rand

# TPE
fmin(objective_sgd, space_sgd, algo=tpe.suggest, max_evals=n_tpe, trials=trials_sgd, rstate=np.random.default_rng(42))
# Random
fmin(objective_sgd, space_sgd, algo=rand.suggest, max_evals=n_tpe + n_rand, trials=trials_sgd, rstate=np.random.default_rng(43))
# Annealing
fmin(objective_sgd, space_sgd, algo=anneal.suggest, max_evals=n_total, trials=trials_sgd, rstate=np.random.default_rng(44))

best_idx = np.argmin([t['result']['loss'] for t in trials_sgd.trials])
best_params = trials_sgd.trials[best_idx]['misc']['vals']
final_params = {k: v[0] if isinstance(v, list) else v for k, v in best_params.items()}
final_params['penalty'] = ['l1', 'l2', 'elasticnet'][final_params['penalty']]
final_params['learning_rate'] = ['constant', 'optimal', 'invscaling', 'adaptive'][final_params['learning_rate']]
final_params['max_iter'] = int(final_params['max_iter'])
print("Best Hyperopt params for SGDClassifier:", final_params)

sgd_best = SGDClassifier(loss='log', random_state=42, **final_params)
sgd_best.fit(X_train_clean, y_train)
y_pred_sgd = sgd_best.predict(X_test_clean)
print("Test accuracy:", accuracy_score(y_test, y_pred_sgd))
print(classification_report(y_test, y_pred_sgd))