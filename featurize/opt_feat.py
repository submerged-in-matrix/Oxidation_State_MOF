from env.modules import *
from utils.scaler import X_train_clean, X_test_clean, y_train, y_test
from src.tuned_ensemble import clf_gb, clf_et, clf_knn, clf_sgd
from utils.feature_ranking import shap_feature_order, feature_names

results_ens = []
steps = list(range(30, len(shap_feature_order)+1, 1))
if 44 not in steps:
    steps.append(44)
if 116 not in steps:
    steps.append(116)
if steps[-1] != len(shap_feature_order):
    steps.append(len(shap_feature_order))

steps = sorted(set(steps))

for top_n in steps:
    selected = shap_feature_order[:top_n]
    X_train_top = pd.DataFrame(X_train_clean, columns=feature_names)[selected]
    X_test_top = pd.DataFrame(X_test_clean, columns=feature_names)[selected]

    # Use  tuned ensemble with custom weights if desired
    ensemble = VotingClassifier(
        estimators=[
            ('gb', clf_gb),
            ('et', clf_et),
            ('knn', clf_knn),
            ('sgd', clf_sgd)
        ],
        voting='soft',
        weights=[0.7, 0.7, 0.4, 0.4]
    )

    ensemble.fit(X_train_top, y_train)
    y_pred_ens = ensemble.predict(X_test_top)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    results_ens.append((top_n, acc_ens))
    #print(f"Top {top_n} features: Ensemble test accuracy = {acc_ens:.4f}")

# Plotting
x_ens, y_ens = zip(*results_ens)
plt.figure(figsize=(7,4))
plt.plot(x_ens, y_ens, marker='o')
plt.xlabel("Number of Top Features")
plt.ylabel("Test Accuracy (Ensemble)")
plt.title("Ensemble Test Accuracy vs. Top SHAP Features")
plt.axvline(44, color='orange', linestyle='--', label='N=44 (ET max)')
plt.axvline(116, color='red', linestyle='--', label='N=116 (paper)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Best N for ensemble
best_n_ens, best_acc_ens = max(results_ens, key=lambda t: t[1])
print(f"\nBest ensemble accuracy {best_acc_ens:.4f} achieved with top {best_n_ens} features.")