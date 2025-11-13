from env.modules import *
from utils.scaler import *

# 1. Base models with default parameters
clf_et = ExtraTreesClassifier(random_state=42)
clf_gb = GradientBoostingClassifier(random_state=42)
clf_knn = KNeighborsClassifier()
clf_sgd = SGDClassifier(loss="log", random_state=42)  

# 2. Ensemble with soft voting
ensemble = VotingClassifier(
    estimators=[
        ('et', clf_et),
        ('gb', clf_gb),
        ('knn', clf_knn),
        ('sgd', clf_sgd)
    ],
    voting='soft'
)

# 3. Train ensemble on training data
ensemble.fit(X_train_clean, y_train)

# 4. Evaluate on test set
y_pred = ensemble.predict(X_test_clean)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))