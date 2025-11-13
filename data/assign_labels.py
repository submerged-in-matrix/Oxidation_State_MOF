from env.modules import *


# Example fallback ox states
common_ox = {
    "Fe": [2, 3], "Cu": [1, 2], "Cr": [2, 3, 6], "Mn": [2, 3, 4, 7], "Co": [2, 3], "Ni": [2, 3],
    "V": [3, 4, 5], "Ti": [3, 4], "Mo": [4, 6], "Ag": [1, 2], "Sn": [2, 4], "Pb": [2, 4], "Au": [1, 3],
    "Al": [3], "Mg": [2], "Na": [1], "K": [1], "Ca": [2], "Zn": [2],
    
}

def assign_oxstate_multimetal(row):
    formula = row['name']
    metals = eval(row['metals']) if isinstance(row['metals'], str) else row['metals']
    try:
        comp = Composition(formula)
        # 1. Try pymatgen's logic first
        guesses = comp.oxi_state_guesses()
        if guesses:
            guess = guesses[0]
            ox_assign = {m: guess[m] for m in metals if m in guess}
            if ox_assign: return ox_assign

        # 2. Multi-metal fallback: try all combinations of possible oxidation states
        el_amt = {el.symbol: comp[el] for el in comp.elements}
        nonmetals = [el for el in el_amt if el not in metals]
        # (Optionally) guess anion charge as 0 for unknowns or use standard table if you want more accuracy

        # Generate all metal state combinations
        ox_ranges = [common_ox.get(m, [None]) for m in metals]
        for ox_tuple in product(*ox_ranges):
            total_charge = sum(el_amt[m]*ox for m, ox in zip(metals, ox_tuple))
            # Assume nonmetal/anion charge = 0; for better accuracy, you can guess e.g. O=-2
            if total_charge == 0:
                # Return a dict: {metal: ox for each metal}
                return {m: ox for m, ox in zip(metals, ox_tuple)}
        # 3. If still not possible, assign first possible for each metal (fallback)
        return {m: common_ox.get(m, [None])[0] for m in metals}
    except Exception:
        return {m: common_ox.get(m, [None])[0] for m in metals}

df = pd.read_csv("oqmd_icsd_with_compounds_only.csv")
df['oxstate_label'] = df.apply(assign_oxstate_multimetal, axis=1)
df.to_csv("oqmd_icsd_with_oxstate_labels.csv", index=False)
print("Saved: oqmd_icsd_with_oxstate_labels.csv")
print(df.shape)