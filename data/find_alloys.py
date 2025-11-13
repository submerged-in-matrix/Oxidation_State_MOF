from env.modules import *


df = pd.read_csv("oqmd_icsd_with_metals.csv")
# e.g., keep if generic composition has more than one unique symbol
df = df[df['composition_generic'].str.len() > 1]
df.to_csv("oqmd_icsd_with_compounds_only.csv", index=False)
df.shape