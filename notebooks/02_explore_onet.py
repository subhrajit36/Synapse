import pandas as pd

ONET = "data/taxonomies/onet/db_30_3_text"

occ = pd.read_csv(f"{ONET}/Occupation Data.txt", sep="\t")
sw  = pd.read_csv(f"{ONET}/Software Skills.txt", sep="\t")

print("Occupation columns:", list(occ.columns))
print("Software columns:  ", list(sw.columns))

# Keep only software-domain roles: SOC code starts with "15-".
# occ["col"].str.startswith("15-") gives a column of True/False;
# putting that inside occ[...] keeps only the True rows. This is "boolean filtering".
occ15 = occ[occ["O*NET-SOC Code"].str.startswith("15-")]
sw15  = sw[sw["O*NET-SOC Code"].str.startswith("15-")]

print("\nSoftware roles:", len(occ15))
print("Role -> tool rows:", len(sw15))

print("\n--- Software roles ---")
for title in sorted(occ15["Title"].unique()):     # unique() drops duplicates
    print(" ", title)

print("\n--- Top 15 tools across software roles ---")
# value_counts() counts how often each tool appears = a frequency table.
print(sw15["Workplace Example"].value_counts().head(15))
