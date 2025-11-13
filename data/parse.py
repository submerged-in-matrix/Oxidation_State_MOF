from env.modules import *


base_url = "http://oqmd.org/oqmdapi/formationenergy"
fields = "name,entry_id,icsd_id,sites,unit_cell,composition_generic"
batch_size = 100
offset = 0
max_entries = 7000
all_entries = []

while len(all_entries) < max_entries:
    url = (
        f"{base_url}?fields={fields}"
        f"&format=json"
        f"&limit={batch_size}"
        f"&offset={offset}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        print("Error:", resp.text[:300])
        break
    data = resp.json()
    if not data or not data.get("data"):
        break
    # Only collect up to max_entries
    for row in data["data"]:
        if row.get("icsd_id") and str(row["icsd_id"]).strip() != "0":
            all_entries.append(row)
            if len(all_entries) >= max_entries:
                break
    print(f"Collected {len(all_entries)}/{max_entries} entries so far...")  # progress printout
    if len(data["data"]) < batch_size or len(all_entries) >= max_entries:
        break
    offset += batch_size

df = pd.DataFrame(all_entries[:max_entries])
df.to_csv("oqmd_icsd_unitcell_composition_name.csv", index=False)
print(f"Saved {len(df)} ICSD-tagged entries with unit cell and composition.")