from pathlib import Path


ignore = list()
for path in Path("opt/").rglob("geom*.log"):
    index = path.stem.replace("geom", "")
    if "_" not in index:
        index = index.split("_")[0]
        with open(path) as read_file:
            if "Psi4EngineError" in read_file.read():
                ignore.append(index)

ignore = list(sorted(set(ignore), key=lambda x: int(x)))

with open("ignore", "w+") as write_file:
    write_file.write("\n".join(ignore))
