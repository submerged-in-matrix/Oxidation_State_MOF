from env.modules import *

# Load my original file
orig_df = pd.read_csv("oqmd_icsd_with_oxstate_labels_no_none.csv")

# These lists will be expanded
expanded_names = []
expanded_oxidationstates = []
expanded_stypes = []

for i, row in orig_df.iterrows():
    # Metals column as list
    metals = ast.literal_eval(row['metals']) if isinstance(row['metals'], str) else row['metals']
    # Oxidationstates as dict
    oxstates = ast.literal_eval(row['oxstate_label']) if isinstance(row['oxstate_label'], str) else row['oxstate_label']
    # Repeat for every metal in that entry
    for m in metals:
        expanded_names.append(row['name'])
        expanded_oxidationstates.append(oxstates[m])
        expanded_stypes.append(m)  # species type = metal symbol

print("Number of expanded rows:", len(expanded_names))
print("Number of expanded oxidationstates:", len(expanded_oxidationstates))
print("Number of expanded stypes:", len(expanded_stypes))

# Load existing features (the file with 5692 rows)
df_feat = pd.read_csv("all_features_replicated.csv")

# Rename 'group' to 'column' if present
if 'group' in df_feat.columns:
    df_feat.rename(columns={'group': 'column'}, inplace=True)

# Add columns
df_feat['name'] = expanded_names
df_feat['oxidationstate'] = expanded_oxidationstates
df_feat['stype'] = expanded_stypes

# Save
df_feat.to_csv("all_features_final.csv", index=False)
print("✅ Final file written with extra columns.")
print("Final DataFrame shape:", df_feat.shape)

# Show the first five values for name, column, oxidationstate, stype
print("\nFirst five rows [name, column, oxidationstate, stype]:")
print(df_feat[['name', 'column', 'oxidationstate', 'stype']].head(5))