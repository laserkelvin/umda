
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

formatted_text = """
SMILES: \\verb|{internal_smi}|

Nearest TMC-1 molecule (distance): \\verb|{anchor}| ({distance:.2f})

Is DFT optimized?: {not_opt}


| Property | Value |
|---|---|
| Formula | {formula} |
| Molecular weight | {mass:.3f} |
| IUPAC name | {iupac_name} |
| $\mu_{{a,b,c}}$ | {mu_a}, {mu_b}, {mu_c} |
| $A, B, C$ | {A}, {B}, {C} |
| $A_s, B_s, C_s$ | {A_s}, {B_s}, {C_s} |
| Charge, Multiplicity | {charge}, {multiplicity} |
| Predicted log column density | {log_column:.3f}\pm{log_column_unc:.3f} |
| Electronic energy | {e} |
"""

header = """---
title: {title}
author: {author}
date: {date}
titlepage: {titlepage}
header-includes:
    - '\\usepackage{{graphicx}}'
---

"""

figure = """
\\begin{{figure}}
\\centering
\\includegraphics[width=0.4\\textwidth]{{{fig_path}}}
\\end{{figure}}

"""


foreword = """
# Foreword

The following molecules were identified to be of astrochemical interest towards the dark
molecular cloud TMC-1 using the unsupervised machine learning methodology described in:

Lee _et al._, "Unsupervised Machine Learning of Interstellar Chemical Inventories" (2021)

This PDF presents preliminary data for 1510 molecules that are of interest to
the chemical inventory of TMC-1, as identified with machine learning. The
molecules are identified based on a Euclidean distance cutoff, providing up to
100 of the closest molecules to those already seen in TMC-1. Structures were generated
from their SMILES strings using `OpenBabel` and `rdkit`, and geometry optimization carried out
using the geomeTRIC program:

> Wang, L.-P.; Song, C.C. (2016), _J. Chem, Phys._ 144, 214108. http://dx.doi.org/10.1063/1.4952956

Electronic structure calculations were performed using `psi4`, with both
geometry optimization and dipole moments calculated at at the
$\omega$B97X-D/6-31+G(d) level of theory.  Equilibrium dipole moments and
rotational constants are reported in unsigned debye and MHz respectively; for
the latter, we provide effective scaled parameters as well that empirically
correct for vibration-rotation interactions. Please refer to "Lee, K. L. K. and
McCarthy, M. 2020, _J Phys Chem A_, 5, 898" for information regarding their
uncertainties. For molecules where SCF/geometry optimizations failed to
converge, we provide their dipole moments based on the molecular mechanics
structures. These molecules will be indicated by "Is DFT optimized?: False".

The predicted column densities and uncertainties are given with a simple
Gaussian Process with rational quadratic and white noise kernels. Simply put,
the predicted column densities of unseen molecules are given as functions of
distance in chemical space that decays naturally to zero for infinite distance
from other data points.  The reader is encouraged to look at the distances
between recommendations and TMC-1 molecules to develop an intuition for how the
predicted column density behaves roughly with distance, and interpret them with
the uncertainties accordingly: as a guide but not to rule out molecules
specifically. Molecules with particularly large uncertainties are likely to be
impactful in constraining the chemistry of the source, even if we provide just
an upper limit.

Finally, there is no real ordering of which the molecules are given. This is
quasi-random, although there are pockets of similar molecules based on how
similar the TMC-1 molecules are between searches.

\\newpage
"""


def format_row(row) -> str:
    cols = ["formula", "mass", "iupac_name", "internal_smi", "mu_a", "mu_b", "mu_c", 
            "A", "B", "C", "charge", "multiplicity", "e", "anchor", "distance", "log_column", "log_column_unc", "not_opt"]
    data = {key: row[key] for key in cols}
    data["not_opt"] = not data["not_opt"]
    for key in ["A", "B", "C"]:
        if data["not_opt"]:
            # do scaling
            value = data[key] * 0.9971
            if not np.isnan(value):
                data[f"{key}_s"] = f"{value:.4f}"
            else:
                data[f"{key}_s"] = "$\\infty$"
        else:
            data[f"{key}_s"] = " - "
        if np.isnan(data[key]):
            data[key] = "$\\infty$"
        else:
            data[key] = f"{data[key]:.4f}"
        dipole = data[f"mu_{key.lower()}"]
        if np.isnan(dipole):
            dipole = " - "
        else:
            dipole = f"{dipole:.1f}"
        data[f"mu_{key.lower()}"] = dipole
    e = data["e"]
    if np.isnan(e):
        data["e"] = " - "
    else:
        data["e"] = f"{e:.5f}"
    # escape special character
    #data["internal_smi"] = f"""\\begin{{verbatim}} {data["internal_smi"]} \\end{{verbatim}}"""
    #data["anchor"] = f"""\\begin{{verbatim}} {data["anchor"]} \\end{{verbatim}}"""
    return formatted_text.format_map(data)


def main():
    data = pd.read_csv("summary.csv", index_col=0)
    data.loc[data["iupac_name"].isna(), "iupac_name"] = ""

    with open("report.md", "w+") as write_file:
        time = datetime.datetime.now().strftime("%Y-%m-%d")
        format_dict = {
            "title": "Molecule recommendations for TMC-1",
            "author": ["Lee _et al._",],
            "date": time,
            "titlepage": "true",
        }
        write_file.write(header.format_map(format_dict))
        write_file.write(foreword)
        for key in ["mu_a", "mu_b", "mu_c"]:
            data.loc[:,key] = np.abs(data.loc[:,key])
        data["text"] = data.apply(format_row, axis=1)
        for index, row in data.iterrows():
            mol_name = row["name"]
            page = f"# {mol_name} \n\n"
            if Path(f"png/{mol_name}.png").exists():
                page += figure.format_map({"fig_path": f"png/{mol_name}.png"})
            page += row["text"]
            page += "\n\\newpage\n"
            write_file.write(page)


if __name__ == "__main__":
    main()
