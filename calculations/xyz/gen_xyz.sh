#!/bin/sh

# This script uses OpenBabel to generate 3D coordinates
# from the SMILES recommendations

obabel targets.smi -O geom.xyz --gen3d -m
tar -cf tmc1_targets.tar *.xyz
