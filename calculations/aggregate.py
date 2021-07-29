
import numpy as np
import pandas as pd
from pathlib import Path
from common import Molecule, parse_properties, get_targets, neu_make_image
from loguru import logger

"""
TODO: aggregate results, grab SMILES from target.smi,
parse data from properties results, populate fields
from Pubchem pulls
"""

def main():
    targets = get_targets()
    molecules = list()
    column_df = pd.read_csv("tmc1_recommendations.csv")
    for target in targets:
        smi, index, charge, multi = target
        smi = smi.replace("\n", "")
        props_path = Path(f"props/geom{index}.inp.dat")
        if props_path.exists():
            molecule = parse_properties(props_path)
            # grab data
            row = column_df.iloc[index - 1]
            if row["Distance"] >= 1e-3:
                smi = row["Recommendation"]
                pubchem_data = Molecule.from_pubchem(smi).__dict__
                # remove fields without data
                pubchem_data = {key: value for key, value in pubchem_data.items() if value}
                molecule.__dict__.update(**pubchem_data)
                # populate fields
                molecule.name = f"geom{index}"
                molecule.charge = charge
                molecule.multiplicity = multi
                molecule.internal_smi = smi
                molecule.formula = row["Formula"]
                molecule.mass = row["Mass"]
                molecule.anchor = row["TMC-1 match"]
                molecule.distance = row["Distance"]
                molecule.log_column = row["GPR Ncol"]
                molecule.log_column_unc = row["GPR unc"]
                # now make an image of the molecule
                neu_make_image(smi, f"png/geom{index}.png")
                molecules.append(molecule)
        else:
            logger.info(f"Properties calculation has been done yet for {geom}.")
            continue
    df = pd.DataFrame([datum.__dict__ for datum in molecules])
    df["not_opt"] = np.load("optimized_mask.npy")
    df.to_csv("summary.csv")

if __name__ == "__main__":
    main()
