
from pathlib import Path
from typing import List
from collections import Counter
import re
import os
from subprocess import Popen, PIPE, run
from dataclasses import dataclass

import numpy as np
from pubchempy import get_compounds
from pymoments import Molecule as PyMolecule
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


DrawingOptions.atomLabelFontSize = 55
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3.0


@dataclass
class Molecule(object):
    formula: str = None
    iupac_name: str = None
    pubchem_smi: str = None
    canonical_smi: str = None
    internal_smi: str = None
    weight: float = None
    mu_a: float = None
    mu_b: float = None
    mu_c: float = None
    A: float = None
    B: float = None
    C: float = None
    harm_freq: np.ndarray = None
    e: float = None
    zpe: float = None
    e_tot: float = None
    quad_xx: float = None
    quad_xy: float = None
    quad_xz: float = None
    quad_yy: float = None
    quad_yz: float = None
    quad_zz: float = None
    charge: int = None
    multiplicity: int = None

    @classmethod
    def from_pubchem(cls, smi: str):
        data = get_pubchem_info(smi)
        molecule = cls()
        molecule.iupac_name = data.get("iupac_name")
        molecule.canonical_smi = data.get("canonical_smiles")
        molecule.internal_smi = data.get("internal_smi")
        molecule.formula = data.get("molecular_formula")
        molecule.weight = data.get("molecular_weight")
        molecule.charge = data.get("charge")
        return molecule

    def __repr__(self) -> str:
        return f"Molecule {self.internal_smi}, formula: {self.formula}"


def run_optimize(input_file: str, nproc: int = 8) -> str:
    cmd = ["geometric-optimize", "--engine", "psi4", input_file, "--nt", f"{nproc}"]
    with Popen(cmd, stdout=PIPE, stderr=PIPE) as process:
        result = process.communicate()[0]
    return result.decode("utf-8")


def run_psi4(input_file: str, output_file: str = None, nproc: int = 8):
    cmd = ["psi4", "-n", f"{nproc}", "-i", input_file]
    if output_file:
        cmd.extend(["-o", output_file])
    with Popen(cmd, stdout=PIPE, stderr=PIPE) as process:
        result = process.communicate()[0]
    return result.decode("utf-8")


def check_output(file_path: str) -> bool:
    """
    This function checks an output file, either from geomeTRIC or
    from Psi4, for a successful completion keyword. Returns
    True if the calculation finished successfully, otherwise
    False.
    """
    with open(file_path, "r") as read_file:
        text = read_file.read()
    checks = ["Converged! =D", "Psi4 exiting successfully"]
    return any([check in text for check in checks])


def orient_molecule(geom: str) -> str:
    mol = PyMolecule.from_xyz(geom)
    _ = mol.orient()
    return str(mol)


def is_scf_bad(index: int) -> bool:
    target = Path(f"opt/geom{index}.tmp/output.dat")
    if target.exists():
        with open(target, "r") as read_file:
            contents = read_file.read()
        return "Could not converge SCF iterations" in contents
    return False


def parse_geometry(optim_xyz_path: str) -> str:
    with open(optim_xyz_path, "r") as read_file:
        contents = read_file.readlines()
    read_from = 0
    # find the line index corresponding to the last iteration
    # by going backwards in the file
    for index, line in enumerate(contents[::-1]):
        if "Iteration" in line:
            read_from = index
            # stop reading when we've found it
            break
    # grab the last geometry, which has converged
    geom = ''.join(contents[-index:])
    # now we rotate the geometry into the principal
    # axis coordinates
    mol = PyMolecule.from_xyz(geom)
    _ = mol.orient()
    symbols = ''.join([string[0] for string in str(mol).split()])
    formula = ''.join([f'{key}{value}' for key, value in Counter(symbols).items()])
    return mol, formula


def parse_properties(prop_path: str):
    # use a dataclass as a data structure
    molecule = Molecule()
    molecule.harm_freq = list()
    with open(prop_path, "r") as read_file:
        contents = read_file.readlines()
    for index, line in enumerate(contents):
        if "Rotational constants" in line:
            temp = line.split()
            values = [temp[index] for index in [4, 7, 10]]
            for value, field in zip(values, ["A", "B", "C"]):
                try:
                    setattr(molecule, field, float(value))
                except ValueError:
                    pass
        if "Dipole Moment: [D]" in line:
            temp = contents[index + 1].split()
            values = [temp[index] for index in [5, 3, 1]]
            values = [float(value) for value in values]
            for value, field in zip(values, ["mu_a", "mu_b", "mu_c"]):
                setattr(molecule, field, value)
        if "Quadrupole Moment: [D A]" in line:
            # combine the first and second lines, and then split into a list
            values = f"{contents[index + 1]}\t{contents[index + 2]}".split()
            for key, value in zip(values[::2], values[1::2]):
                # get rid of the colon and make lower case
                key = key.replace(":", "").lower()
                setattr(molecule, f"quad_{key}", float(value))
        if "Total Energy = " in line:
            value = float(line.split()[-1])
            setattr(molecule, "e", value)
        if "Total ZPE, Electronic energy" in line:
            value = float(line.split()[-2])
            setattr(molecule, "e_tot", value)
        if "Correction ZPE" in line:
            value = float(line.split()[-4])
            setattr(molecule, "zpe", value)
        if "Freq [cm^-1]" in line:
            values = list()
            for value in line.split()[-3:]:
                if value.isdigit():
                    values.append(float(value))
            molecule.harm_freq.extend(values)
    return molecule


def get_targets() -> List[str]:
    with open("xyz/targets.smi", "r") as read_file:
        data = list()
        for index, line in enumerate(read_file.readlines()):
            smi = line.strip()
            charge, num_elec, multi = get_mol_data(smi)
            # doublets for open shell, singlets otherwise
            #multi = 1 if num_elec % 2 == 0 else 2
            # smiles, geom, index, charge, multi
            data.append([smi, index + 1, charge, multi])
        # sort by the geometry index
        data = sorted(data, key=lambda x: x[1])
    return data


def get_mol_data(smi: str):
    mol = Chem.MolFromSmiles(smi)
    charge = Chem.GetFormalCharge(mol)
    num_rad_electrons = 0
    for atom in mol.GetAtoms():
        num_rad_electrons += atom.GetNumRadicalElectrons()
    total_spin = num_rad_electrons / 2
    multi = 2 * total_spin + 1
    valence = Descriptors.NumValenceElectrons(mol)
    #mol = Chem.MolFromSmiles(smi, sanitize=False)
    #mol.UpdatePropertyCache(strict=False)
    return charge, valence, int(multi)


def get_pubchem_info(smi: str):
    data = {"internal_smi": smi}
    try:
        data.update(**get_compounds(smi, "smiles").pop().to_dict())
    except Exception as error:
        print(f"{smi} pubchem search failed because of {error}")
    finally:
        return data


def neu_make_image(smi: str, output_path: str):
    mol = Chem.MolFromSmiles(smi)
    #mol.UpdatePropertyCache(strict=False)
    Draw.MolToFile(mol, output_path)


def make_image(optim_xyz_path: str):
    name = optim_xyz_path.split("/")[-1].split(".")[0]
    xyz, formula = parse_geometry(optim_xyz_path)
    num_atoms = len(xyz.split("\n"))
    try:
        os.mkdir("png")
    except:
        pass
    formatted = f"{num_atoms}\n{name}\n{xyz}"
    with open(f"png/{name}.xyz", "w+") as write_file:
        write_file.write(formatted)
    with open(f"png/{name}.png", "wb+") as write_file:
        proc = Popen(["/usr/bin/obabel", "-ixyz", f"png/{name}.xyz", "-O", f"png/{name}.png", "-d", "-t", "-xw", "1200", "-xh", "1200", "-xb", "none"])
        #proc = run(["/usr/bin/obabel", "-ixyz", f"png/{name}.xyz", "-O", f"png/{name}.svg", "-xS"])
        #proc = run(["inkscape", "-D", "-z", f"--file=png/{name}.svg", f"--export-pdf=png/{name}.pdf", "--export-latex"])
    return formula

