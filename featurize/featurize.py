from env.modules import *
from utils.featurization_helpers import *

# --- LOAD YOUR DATA ---
df = pd.read_csv("oqmd_icsd_with_oxstate_labels_no_none.csv")
print("Loaded DataFrame shape:", df.shape)

# --- MAIN FEATURE LOOP ---
all_features = []
feature_names_generated = None

for i, row in df.iterrows():
    try:
        cell = np.array(ast.literal_eval(row['unit_cell']))
        sites = ast.literal_eval(row['sites'])
        if isinstance(sites[0], str):
            species = [s.split('@')[0].strip() for s in sites]
            coords = [list(map(float, s.split('@')[1].strip().split())) for s in sites]
        else:
            species = [site['element'] for site in sites]
            coords = [site['xyz'] for site in sites]
        struct = Structure(Lattice(cell), species, coords, coords_are_cartesian=True)
        metals = ast.literal_eval(row['metals']) if isinstance(row['metals'], str) else row['metals']
        oxstate = ast.literal_eval(row['oxstate_label']) if isinstance(row['oxstate_label'], str) else row['oxstate_label']
        # DEBUG PRINT:
        #print(f"ROW {i}: metals={metals} oxstate={oxstate} species={species}")
        for m in metals:
            if m in oxstate:
                for idx, site in enumerate(struct):
                    #print(f"    Checking site idx={idx} site={site.specie} vs m={m}")
                    if Element(str(site.specie)).symbol == m:
                        #print(f"    --> MATCH: site={site.specie} with metal={m}")
                        feat, names = [], []
                        f, n = metal_center_features(Element(m)); feat += f; names += n
                        f, n = geometry_features(struct, idx); feat += f; names += n
                        f, n = local_chemistry_features(struct, idx); feat += f; names += n
                        f, n = gaussian_symm_features(struct, idx); feat += f; names += n
                        all_features.append(feat)
                        if feature_names_generated is None:
                            print("[DEBUG] Metal center:", len(metal_center_features(Element(m))[0]))
                            print("[DEBUG] Geometry:", len(geometry_features(struct, idx)[0]))
                            print("[DEBUG] Chemistry:", len(local_chemistry_features(struct, idx)[0]))
                            print("[DEBUG] Gaussian:", len(gaussian_symm_features(struct, idx)[0]))
                            print("[DEBUG] TOTAL:", len(feat))
                            print("[DEBUG] Names:", len(names))
                            feature_names_generated = names
                        break
    except Exception as e:
        print(f"Error at row {i}: {e}")
        continue

all_features = np.array(all_features)
print("Final shape:", all_features.shape)
print("Feature names:", len(feature_names_generated))
assert all_features.shape[1] == len(feature_names_generated), "Mismatch between features and names!"

# Save
np.save("all_features_replicated.npy", all_features)
pd.DataFrame(all_features, columns=feature_names_generated).to_csv("all_features_replicated.csv", index=False)