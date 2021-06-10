
from pathlib import Path


install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "processed": top.joinpath("data/processed"),
    "models": top.joinpath("models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts")
}