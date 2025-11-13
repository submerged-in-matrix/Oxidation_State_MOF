from env.modules import *
from utils.opt_weight import X_train_final, X_test_final, X_holdout_final
from utils.scaler import y_train, y_test, y_holdout
from optimization.optimize_GB import final_params_gb
from optimization.optimize_ET import final_params_et
from optimization.optimize_lgbm import final_params_lgbm
from optimization.optimize_rf import final_params_rf

# 1. Define base models with tuned hyperparameters
clf_gb = GradientBoostingClassifier(random_state=0, **final_params_gb)
clf_et = ExtraTreesClassifier(random_state=0, **final_params_et)
clf_lgbm = LGBMClassifier(random_state=0, n_jobs=-1, **final_params_lgbm)
clf_rf = RandomForestClassifier(random_state=0, n_jobs=-1, **final_params_rf )

# 2. Hyperopt weight optimization (soft voting)
space = {
    'w_gb': hp.uniform('w_gb', 1.0, 10.0),
    'w_et': hp.uniform('w_et', 1.0, 10.0),
    'w_lgbm': hp.uniform('w_lgbm', 1.0, 10.0),
    'w_rf': hp.uniform('w_rf', 1.0, 10.0)
}

def objective(params):
    weights = [params['w_gb'], params['w_et'], params['w_lgbm'], params['w_rf']]
    ensemble = VotingClassifier(
        estimators=[
            ('gb', clf_gb),
            ('et', clf_et),
            ('lgbm', clf_lgbm),
            ('rf', clf_rf)
        ],
        voting='soft',
        weights=weights
    )
    ensemble.fit(X_train_final, y_train)
    y_pred = ensemble.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # cost-effective number of evaluations
    trials=trials,
    rstate=np.random.default_rng(42)
)

# 3. Use best weights for final ensemble
best_weights = [best['w_gb'], best['w_et'], best['w_lgbm'], best['w_rf']]
print("\nOptimized weights:", best_weights)

ensemble_best = VotingClassifier(
    estimators=[
        ('gb', clf_gb),
        ('et', clf_et),
        ('lgbm', clf_lgbm),
        ('rf', clf_rf)
    ],
    voting='soft',
    weights=best_weights
)
ensemble_best.fit(X_train_final, y_train)

# Test
y_pred = ensemble_best.predict(X_test_final)
print("\nTest accuracy (optimized VotingClassifier):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Holdout
y_pred_holdout = ensemble_best.predict(X_holdout_final)
print("\nHoldout accuracy:", accuracy_score(y_holdout, y_pred_holdout))
print(classification_report(y_holdout, y_pred_holdout))