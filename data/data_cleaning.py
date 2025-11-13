from env.modules import *

df = pd.read_csv("oqmd_icsd_with_oxstate_labels.csv")

def has_none_oxstate(row):
    oxdict = eval(row['oxstate_label']) if isinstance(row['oxstate_label'], str) else row['oxstate_label']
    return any(v is None for v in oxdict.values())

# Filter out rows with None in any metal oxidation state
filtered_df = df[~df.apply(has_none_oxstate, axis=1)]

print("Filtered shape:", filtered_df.shape)
filtered_df.to_csv("oqmd_icsd_with_oxstate_labels_no_none.csv", index=False)
filtered_df.head