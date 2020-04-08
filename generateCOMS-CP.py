#!/usr/bin/env python3
#coding:utf-8
"""
  Author:   peregrine.warren<at>physics.ox.ac.uk
  Purpose:  Generates Gaussian input files (.com) from geometry files (.xyz) of 
            two molecules for a Counterpoise calculation.
            This is a rewrite of the bash scripts in https://github.com/Jenny-Nelson-Group/counterpoise-J
  Created: 07/04/20
"""

import numpy as np
import pandas as pd
from inspect import cleandoc # to ignore indents when printing to file
#import sys

# Set variables for top of com file
nprocshared = 1
mem = '14GB'
method = 'b3lyp/6-31g*'

def read_geom(file_in):
    """read .xzy to dataframe."""
    mol_xyz = pd.read_csv(file_in, sep=' ', skiprows=0, skipinitialspace=1, header=None, names=['atom', 'x', 'y', 'z'])
    return mol_xyz

def write_com(geom_in, com_name):
    """generate com file from geom file"""
    with open(com_name, "w") as text_file:
    	print(cleandoc(
    		f'''
    		%nprocshared={nprocshared}
    		%mem={mem}
    		#p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) {method} nosymm

    		autogen

    		0 1'''), file=text_file)
    	print(geom_in.to_string(header=False, index=False), file=text_file)
    return

def translate_molecule(mol_xyz, d, dr):
    """translate a molecule in x, y or z by distance d in direction dr."""
    mol_xyz[dr] = mol_xyz[dr] + d
    return mol_xyz

def rotate_molecule(mol_xyz, d, dr, cen):
    """rotate a molecule by degress deg in a clockwise direction around center cen."""
    mol_xyz[dr] = mol_xyz[dr] + d
    return mol_xyz

# define paths
geoms = 'geom_files/'
coms = 'com_files/'

# Read in molecules
znpc_xyz = geoms + 'ZnPc.xyz'
f6_xyz = geoms + 'F6TCNNQ.xyz'
znpc_geom = read_geom(znpc_xyz)
f6_geom = read_geom(f6_xyz)

# Write a com file
write_com(znpc_geom, 'znpc_test.com')

# Translate one molecule in Z-direction
# for Z in np.arange(3,6,0.5):	
	# print (Z)
	# znpc_Z = translate_molecule(znpc_xyz, Z, 'z')

