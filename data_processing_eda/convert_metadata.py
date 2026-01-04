import json

nb_path = "EDA_Spoken_SQuAD.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove widget metadata ONLY
nb["metadata"].pop("widgets", None)

for cell in nb.get("cells", []):
    cell.get("metadata", {}).pop("widgets", None)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Widget metadata removed, outputs preserved")
