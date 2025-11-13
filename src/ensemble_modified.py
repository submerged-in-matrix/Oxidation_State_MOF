from env.modules import *
from utils.scaler import X_train_final, X_test_final, X_holdout_final, y_train, y_test, y_holdout
from src.tuned_ensemble import final_params_gb, final_params_et

# Use previously tuned GB and ET
clf_gb = GradientBoostingClassifier(random_state=0, **final_params_gb)
clf_et = ExtraTreesClassifier(random_state=0, **final_params_et)
clf_lgbm = LGBMClassifier(random_state=0, n_jobs=-1)
clf_rf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Use same features and data
X_train_base = X_train_final
X_test_base = X_test_final
X_holdout_base = X_holdout_final

# New ensemble: GB, ET, LGBM, RF
ensemble_new = VotingClassifier(
    estimators=[
        ('gb', clf_gb),
        ('et', clf_et),
        ('lgbm', clf_lgbm),
        ('rf', clf_rf)
    ],
    voting='soft',
    weights=[1, 1, 1, 1]  # Equal weights for a fair first test
)

ensemble_new.fit(X_train_base, y_train)

# Test
y_pred_new = ensemble_new.predict(X_test_base)
print("\nTest accuracy (ensemble: GB+ET+LGBM+RF):", accuracy_score(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))

# Holdout
y_pred_holdout_new = ensemble_new.predict(X_holdout_base)
print("\nHoldout accuracy:", accuracy_score(y_holdout, y_pred_holdout_new))
print(classification_report(y_holdout, y_pred_holdout_new))