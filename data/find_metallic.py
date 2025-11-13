from env.modules import *
from data.parse import *

df = pd.read_csv("oqmd_icsd_unitcell_composition_name.csv")

def find_metals(formula):
    try:
        comp = Composition(formula)
        # List all elements in the composition that are metals
        return [el.symbol for el in comp.elements if Element(el.symbol).is_metal]
    except Exception:
        return []

df['metals'] = df['name'].apply(find_metals)
df = df[df['metals'].map(len) > 0]  # Filter to only those with at least one metal
df.to_csv("oqmd_icsd_with_metals.csv", index=False)
print("Done. File with 'metals' column saved.")
df.shape