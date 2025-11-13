from env.modules import *
from utils.scaler import X_train_clean, y_train, X_test_clean, y_test

space_knn = {
    'n_neighbors': hp.quniform('n_neighbors', 1, 20, 1),
    'leaf_size': hp.quniform('leaf_size', 10, 100, 1),
    'weights': hp.choice('weights', ['uniform', 'distance']),
    'p': hp.choice('p', [1, 2]),
    'metric': hp.choice('metric', ['manhattan', 'euclidean']),
}

def objective_knn(params):
    params = params.copy()
    params['n_neighbors'] = int(params['n_neighbors'])
    params['leaf_size'] = int(params['leaf_size'])
    model = KNeighborsClassifier(**params)
    score = cross_val_score(model, X_train_clean, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}

trials_knn = Trials()
n_total = 500
n_tpe = int(0.8 * n_total)
n_rand = int(0.1 * n_total)
n_anneal = n_total - n_tpe - n_rand

# TPE
fmin(objective_knn, space_knn, algo=tpe.suggest, max_evals=n_tpe, trials=trials_knn, rstate=np.random.default_rng(42))
# Random
fmin(objective_knn, space_knn, algo=rand.suggest, max_evals=n_tpe + n_rand, trials=trials_knn, rstate=np.random.default_rng(43))
# Annealing
fmin(objective_knn, space_knn, algo=anneal.suggest, max_evals=n_total, trials=trials_knn, rstate=np.random.default_rng(44))

best_idx = np.argmin([t['result']['loss'] for t in trials_knn.trials])
best_params = trials_knn.trials[best_idx]['misc']['vals']
final_params = {k: v[0] if isinstance(v, list) else v for k, v in best_params.items()}
final_params['n_neighbors'] = int(final_params['n_neighbors'])
final_params['leaf_size'] = int(final_params['leaf_size'])
final_params['weights'] = ['uniform', 'distance'][final_params['weights']]
final_params['p'] = [1, 2][final_params['p']]
final_params['metric'] = ['manhattan', 'euclidean'][final_params['metric']]
print("Best Hyperopt params for KNN:", final_params)

knn_best = KNeighborsClassifier(**final_params)
knn_best.fit(X_train_clean, y_train)
y_pred_knn = knn_best.predict(X_test_clean)
print("Test accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))