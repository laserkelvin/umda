from pathlib import Path
from ruamel.yaml import YAML
from loguru import logger
import hashlib
import common
from multiprocessing.pool import ThreadPool


def run_subprocess(target, template: str):
    smi, index, charge, multi = target
    ref = "rhf" if multi == 1 else "uhf"
    done  = False
    # go through each index, check if the optimization was done
    if Path(f"props/geom{index}.inp.dat").exists():
        done = common.check_output(f"props/geom{index}.inp.dat")
    # we only run the properties calculation if they're optimized and
    # ndy calculated
    if not done:
        with open(f"xyz/geom{index}.xyz", "r") as read_file:
            data = read_file.readlines()
            # exclude the number of atoms and comment line
            geom = "".join(data[2:])
            geom = common.orient_molecule(geom)
        with open(f"props/geom{index}.inp", "w+") as write_file:
            write_file.write(template.format_map({"geometry": geom, "charge": charge, "multi": multi, "ref": ref}))
            logger.info(f"Calculating properties for geometry {index}.")
        result = common.run_psi4(f"props/geom{index}.inp", nproc=6)
    else:
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
