from env.modules import *

magpie = MagpieData()
geom_featurizer = CrystalNNFingerprint.from_preset("ops")
gsf = GaussianSymmFunc(
    etas_g2=[0.05, 4.0, 20.0, 80.0],
    etas_g4=[0.005, 0.005, 0.005, 0.005],
    zetas_g4=[1.0, 1.0, 4.0, 4.0],
    gammas_g4=[-1.0, 1.0, -1.0, 1.0],
    cutoff=6.5
)

def metal_center_features(el):
    config = el.full_electronic_structure
    Z = el.Z
    group = el.group
    row = el.row
    valence = config[-1][2] if config else np.nan
    diffto18 = 18 - valence if valence is not None else np.nan
    n_s = sum(1 for o in config if o[1] == 's')
    n_p = sum(1 for o in config if o[1] == 'p')
    n_d = sum(1 for o in config if o[1] == 'd')
    sunfilled = 2 - n_s
    punfilled = 6 - n_p
    dunfilled = 10 - n_d
    feats = [group, row, Z, valence, diffto18, sunfilled, punfilled, dunfilled]
    names = ['group', 'row', 'Z', 'valenceelectrons', 'diffto18electrons', 'sunfilled', 'punfilled', 'dunfilled']
    return feats, names

def geometry_features(struct, idx):
    feats = geom_featurizer.featurize(struct, idx)
    names = geom_featurizer.feature_labels()
    return feats, names

def get_local_shell_props(el):
    config = el.full_electronic_structure
    n_s = sum(1 for o in config if o[1] == 's')
    n_p = sum(1 for o in config if o[1] == 'p')
    n_d = sum(1 for o in config if o[1] == 'd')
    n_f = sum(1 for o in config if o[1] == 'f')
    props = {
        'NsValence': n_s,
        'NpValence': n_p,
        'NdValence': n_d,
        'NfValence': n_f,
        'NsUnfilled': 2 - n_s,
        'NpUnfilled': 6 - n_p,
        'NdUnfilled': 10 - n_d,
        'NfUnfilled': 14 - n_f,
        'NUnfilled': (2 - n_s) + (6 - n_p) + (10 - n_d) + (14 - n_f)
    }
    return props

def local_chemistry_features(struct, idx):
    vnn = VoronoiNN()
    neighs = vnn.get_nn_info(struct, idx)
    center_site = struct[idx]
    elems = [n['site'].specie for n in neighs]
    weights = [n['weight'] for n in neighs]
    chem_properties = [
        "MendeleevNumber", "Column", "Row", "Electronegativity", "NsValence",
        "NpValence", "NdValence", "NfValence", "NValence",
        "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled", "GSbandgap"
    ]
    features, names = [], []
    for prop in chem_properties:
        try:
            if prop in ["MendeleevNumber", "Column", "Row", "Electronegativity", "NValence", "GSbandgap"]:
                if prop == "Column":
                    center_val = Element(str(center_site.specie)).group
                    n_props = [Element(str(e)).group for e in elems]
                else:
                    center_val = magpie.get_elemental_property(center_site.specie, prop)
                    n_props = magpie.get_elemental_properties(elems, prop)
            else:
                center_val = get_local_shell_props(center_site.specie)[prop]
                n_props = [get_local_shell_props(e)[prop] for e in elems]
        except Exception:
            center_val = np.nan
            n_props = [np.nan] * len(elems)
        arr = np.array([x - center_val for x in n_props])
        wts = np.array(weights)
        if len(arr) == 0 or np.all(np.isnan(arr)):
            vals = [np.nan]*4
        else:
            vals = [
                np.average(np.abs(arr), weights=wts) if not np.all(np.isnan(arr)) else np.nan,  # abs_mean
                np.nanmax(np.abs(arr)),
                np.nanmin(np.abs(arr)),
                np.average(arr, weights=wts) if not np.all(np.isnan(arr)) else np.nan          # signed_mean
            ]
        features.extend(vals)
        names += [
            f"local difference in {prop}",
            f"maximum local difference in {prop}",
            f"mimum local difference in {prop}",
            f"local signed difference in {prop}",
        ]
    return features, names

def gaussian_symm_features(struct, idx):
    all_feats = gsf.featurize(struct, idx)
    indices = [0, 1, 2, 3, 4, 5, 12, 13]
    labels = gsf.feature_labels()
    feats = [all_feats[i] for i in indices]
    names = [labels[i] for i in indices]
    return feats, names