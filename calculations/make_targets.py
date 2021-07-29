import pandas as pd
import hashlib
import common

smiles = list()

df = pd.read_csv("tmc1_recommendations.csv")
with open("xyz/targets.smi", "w+") as write_file:
    for smi in df["Recommendation"]:
        write_file.write(f"{smi}\n")
        smiles.append(smi)

targets = common.get_targets()
smiles = [target[0] for target in targets]
smi_hash = hashlib.md5("\n".join(smiles).encode()).hexdigest()
with open("target_md5", "w+") as write_file:
    write_file.write(smi_hash)
