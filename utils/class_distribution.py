from env.modules import *
from utils.scaler import *

plt.figure(figsize=(10, 5))
bar_width = 0.25
classes = sorted(set(y_full))
x = np.arange(len(classes))

bars_train = plt.bar(x - bar_width, [train_counts.get(i, 0) for i in classes], width=bar_width, label="Train")
bars_test = plt.bar(x, [test_counts.get(i, 0) for i in classes], width=bar_width, label="Test")
bars_holdout = plt.bar(x + bar_width, [holdout_counts.get(i, 0) for i in classes], width=bar_width, label="Holdout")

plt.xticks(x, classes)
plt.xlabel("Oxidation State")
plt.ylabel("Number of Samples")
plt.yscale('log')
plt.legend()
plt.title("Class Distribution in Train, Test, Holdout Sets")

# Annotate bar heights
for bars in [bars_train, bars_test, bars_holdout]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height, f"{int(height)}", 
                     ha='center', va='bottom', fontsize=8, rotation=90, clip_on=True)

plt.tight_layout()
plt.show()