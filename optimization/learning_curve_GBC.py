from env.modules import *
from optimization.optimize_GB import *

# Extract losses (negative accuracy, so best is min(loss))
losses = [-t['result']['loss'] for t in trials.trials]  

# Cumulative best (max so far)
cummax = np.maximum.accumulate(losses)

# Trial when best found
best_trial_idx = np.argmax(losses)
best_trial_score = losses[best_trial_idx]

# Plot learning curve
plt.figure(figsize=(8,4))
plt.plot(range(1, len(cummax)+1), cummax, label='Best accuracy so far')
plt.xlabel('Trial')
plt.ylabel('Cross-validated accuracy')
plt.title('Learning Curve for GB Hyperopt Tuning')
plt.axvline(best_trial_idx+1, color='r', linestyle='--', label=f'Best found at trial {best_trial_idx+1}')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Best score: {best_trial_score:.4f} at trial {best_trial_idx+1}")