from env.modules import *
from utils.scaler import X_train_clean, X_test_clean, y_train, X_full
from src.tuned_ensemble import clf_et

# Fit the best ExtraTreesClassifier ()
clf_et.fit(X_train_clean, y_train)

# Get SHAP values (multi-class supported)
explainer = shap.TreeExplainer(clf_et)
shap_values = explainer.shap_values(X_test_clean)
feature_names = X_full.columns

# Global importance: mean(|SHAP|) across classes
if isinstance(shap_values, list):
    mean_abs_shap = np.mean([np.abs(vals).mean(axis=0) for vals in shap_values], axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

sorted_idx = np.argsort(mean_abs_shap)[::-1]
shap_feature_order = feature_names[sorted_idx].tolist()

print("Top 10 SHAP-ranked features:", shap_feature_order[:10])
print("Total features ranked:", len(shap_feature_order))