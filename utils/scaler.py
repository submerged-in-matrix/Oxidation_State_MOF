from env.modules import *

# Load and filter
df = pd.read_csv("all_features_final.csv")
df = df[df["oxidationstate"].apply(float.is_integer)].copy()
df["oxidationstate"] = df["oxidationstate"].astype(int)

# Remove oxidation states with fewer than 8 samples
counts = df["oxidationstate"].value_counts()
valid_states = counts[counts >= 8].index
df = df[df["oxidationstate"].isin(valid_states)].copy()

# Save metadata as CSV
metadata_df = df.copy()
metadata_df.to_csv("cleaned_metadata_with_names.csv", index=False)
print("Metadata shape (rows, columns):", metadata_df.shape)

meta_counts = metadata_df["oxidationstate"].value_counts().sort_index()

# Prepare numeric features
X_full = df.drop(columns=["oxidationstate"]).select_dtypes(include=[float, int])
y_full = df["oxidationstate"]

# 1. Random (not stratified) holdout split (185 samples)
np.random.seed(42)
holdout_indices = np.random.choice(X_full.index, size=185, replace=False)
X_holdout = X_full.loc[holdout_indices].copy()
y_holdout = y_full.loc[holdout_indices].copy()

X_remain = X_full.drop(index=holdout_indices)
y_remain = y_full.drop(index=holdout_indices)

# 2. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_remain, y_remain, test_size=0.2, stratify=y_remain, random_state=42
)

# Class counts for plotting
train_counts = y_train.value_counts().sort_index()
test_counts = y_test.value_counts().sort_index()
holdout_counts = y_holdout.value_counts().sort_index()

# Scaling and imputation
scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_holdout_scaled = scaler.transform(X_holdout)
imputer = SimpleImputer(strategy="median").fit(X_train_scaled)
X_train_clean = imputer.transform(X_train_scaled)
X_test_clean = imputer.transform(X_test_scaled)
X_holdout_clean = imputer.transform(X_holdout_scaled)

# # Print shapes and unique classes
# print("X_train_clean shape:", X_train_clean.shape)
# print("X_test_clean shape:", X_test_clean.shape)
# print("X_holdout_clean shape:", X_holdout_clean.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)
# print("y_holdout shape:", y_holdout.shape)
# print("Unique oxidation states (after cleaning):", np.unique(y_train))