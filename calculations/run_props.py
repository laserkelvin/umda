from pathlib import Path
from ruamel.yaml import YAML
from loguru import logger
import hashlib
import common
from multiprocessing.pool import ThreadPool


def run_subprocess(target, template: str):
    smi, index, charge, multi = target
    ref = "rhf" if multi == 1 else "uhf"
    opt, done  = False, False
    # go through each index, check if the optimization was done
    if Path(f"opt/geom{index}.log").exists():
        opt = common.check_output(f"opt/geom{index}.log")
    if Path(f"props/geom{index}.inp.dat").exists():
        done = common.check_output(f"props/geom{index}.inp.dat")
    # we only run the properties calculation if they're optimized and
    # not already calculated
    if not done:
        optimized_path = Path(f"opt/geom{index}_optim.xyz")
        # only use the optimized structure if it was successful
        if optimized_path.exists() and opt:
            logger.info(f"Optimized structure used from geom{index}")
            mol, formula = common.parse_geometry(optimized_path)
            geom = str(mol)
        # if the SCF or optimization failed, we use the MM structure
        else:
            logger.info(f"MM structure used for geom{index}.")
            with open(f"xyz/geom{index}.xyz", "r") as read_file:
                data = read_file.readlines()
            # exclude the number of atoms and comment line
            geom = "".join(data[2:])
        with open(f"props/geom{index}.inp", "w+") as write_file:
            write_file.write(template.format_map({"geometry": geom, "charge": charge, "multi": multi, "ref": ref}))
        result = common.run_psi4(f"props/geom{index}.inp", nproc=6)
    elif opt and done:
        logger.info(f"Properties calculated for geometry {index}; skipping.")


def main():
    logger.add("properties.txt")

    with open("template_props.inp", "r") as read_file:
        template = read_file.read()
    logger.info(f"Using template:\n {template}")

    targets = common.get_targets()
    smiles = [target[0] for target in targets]
    smi_hash = hashlib.md5('\n'.join(smiles).encode()).hexdigest()
    logger.info(f"SMILES file MD5: {smi_hash}")
    logger.info(f"Running through {len(smiles)} SMILES codes.")
    CONCURRENT = 6
    tp = ThreadPool(CONCURRENT)
    for target in targets:
        tp.apply_async(run_subprocess, (target, template))
    tp.close()
    tp.join()


if __name__ == "__main__":
    main()
