from pathlib import Path
from subprocess import Popen, PIPE
from ruamel.yaml import YAML
from loguru import logger
from multiprocessing.pool import ThreadPool
import hashlib
import common


def opt_subprocess(target, template):
    # unpack
    smi, index, charge, multi = target
    ref = "rhf" if multi == 1 else "uhf"
    done = False
    bad_scf = common.is_scf_bad(index)
    logger.info(f"{smi} - {index} - Charge: {charge}, Multiplicity: {multi}, bad SCF? {bad_scf}")
    with open(f"xyz/geom{index}.xyz", "r") as read_file:
        data = read_file.readlines()
    # exclude the number of atoms and comment line
    geom = "".join(data[2:])
    output_file = Path(f"opt/geom{index}_optim.xyz").exists()
    if output_file:
        done = common.check_output(f"opt/geom{index}.log")
    # if the calculation was run before, but didn't converge, grab the
    # last geometry and redo
    if output_file and not done:
        try:
            mol, _ = common.parse_geometry(f"opt/geom{index}_optim.xyz")
            # convert mol to coordinates
            geom = str(mol)
            logger.info("Restarting from last optimization attempt.")
        except FileNotFoundError:
            logger.info("Previous geometry not found; trying from scratch.")
        finally:
            done = False
    if not done and not bad_scf:
        with open(f"opt/geom{index}.inp", "w+") as write_file:
            write_file.write(
                template.format_map(
                    {"geometry": geom, "charge": charge, "multi": multi, "ref": ref}
                )
            )
        logger.info(f"Running geometry {index}.")
        result = common.run_optimize(f"opt/geom{index}.inp", nproc=8)
    elif done:
        logger.info(f"Skipping geometry {index} as it is already done.")
    else:
        logger.info(f"Geometry {index} SCF fails to converge. Skipping.")


def main():
    logger.add("astro_survey.txt")

    with open("template_opt.inp", "r") as read_file:
        template = read_file.read()
    logger.info(f"Using template:\n {template}")

    targets = common.get_targets()
    smiles = [target[0] for target in targets]
    smi_hash = hashlib.md5("\n".join(smiles).encode()).hexdigest()
    logger.info(f"SMILES file MD5: {smi_hash}")
    logger.info(f"Running through {len(smiles)} SMILES codes.")
    with open("ignore", "r") as read_file:
        ignore = read_file.readlines()
        ignore = [int(i.strip()) for i in ignore]
    targets = list(filter(lambda x: x[1] not in ignore, targets))

    CONCURRENT = 4
    tp = ThreadPool(CONCURRENT)
    for target in targets:
        tp.apply_async(opt_subprocess, (target, template))
    tp.close()
    tp.join()


if __name__ == "__main__":
    main()
