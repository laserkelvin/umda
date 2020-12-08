
from tqdm.auto import tqdm
from umda.smi_vec import canonicize_smi

MAX_NUM = 3000000

canon_smi = list()
counter = 0

with open("../data/external/pubchem/pubchem_screened.smi", "w+") as write_file:
    with open("../data/external/pubchem/CID-SMILES", "r") as read_file:
        for line in tqdm(read_file):
            if counter == MAX_NUM:
                break
            _, smi = line.split()
            try:
                new_smi = canonicize_smi(smi)
                counter += 1
                canon_smi.append(new_smi)
            except:
                pass
    print("Filtering unique SMILES now and writing")
    counter = 0
    for smi in tqdm(set(canon_smi)):
        counter += 1
        write_file.write(f"{smi}\n")
    print(f"Wrote {counter} molecules from PubChem")
